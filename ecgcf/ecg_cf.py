"""ecg_cf — physiology-grounded counterfactual perturbations of 12-lead ECGs.

Pipeline
    fid  = delineate(sig, fs)                       # fiducial points per beat
    sig2, info = perturb(sig, fs, fid, kind, level) # controlled edit of one wave
    ver  = verify(sig, sig2, fs, kind, level)       # independent re-delineation + measurement

Signals: array (n_samples, 12) in millivolts, lead order
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 (PTB-XL order).
"""
import numpy as np
from scipy.signal import butter, filtfilt

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
L = {name: i for i, name in enumerate(LEADS)}

# ----------------------------------------------------------------------------------
# Perturbation catalogue: 5 pathological edits + 3 controls, 3 intensity levels each
# (4 for qrs_wide)
# ----------------------------------------------------------------------------------
CATALOGUE = {
    "st_elev":  dict(levels=[0.10, 0.20, 0.30], unit="mV", wave="ST",
                     dx="anterior ST elevation (STEMI pattern)", leads=["V1", "V2", "V3", "V4"]),
    "t_inv":    dict(levels=[0.5, 1.0, 1.5], unit="x T amplitude", wave="T",
                     dx="T-wave inversion (ischaemia pattern)", leads=["V2", "V3", "V4", "V5"]),
    "qt_long":  dict(levels=[60, 100, 140], unit="ms", wave="QT",
                     dx="prolonged QT interval", leads=None),
    # 4th level (+80 ms) added after the real-data audit: with the conservative
    # independent re-measurement (~0.78x of the stretch), +60 ms met QRS >= 120 ms in only
    # 60% of real recordings, which left the upper end of the dose-response unanchored.
    "qrs_wide": dict(levels=[20, 40, 60, 80], unit="ms", wave="QRS",
                     dx="wide QRS (conduction delay / bundle branch block)", leads=None,
                     # QT includes the QRS: widening it lengthens QT by construction.
                     # This is an expected physiological consequence, not a side effect.
                     secondary=["long_QT"]),
    "af":       dict(levels=[1, 2, 3], unit="severity", wave="P/RR",
                     dx="atrial fibrillation (absent P, irregular RR)", leads=None),
    # added for the review: anterior ST elevation WITH reciprocal inferior ST depression
    # (half the amplitude in II, III, aVF), a more complete STEMI pattern than st_elev.
    "st_recip": dict(levels=[0.10, 0.20, 0.30], unit="mV", wave="ST",
                     dx="anterior ST elevation with reciprocal inferior ST depression",
                     leads=["V1", "V2", "V3", "V4"], recip_leads=["II", "III", "aVF"], recip_factor=0.5,
                     secondary=["ST_depression"]),
    "ctrl_amp":    dict(levels=[0.85, 1.15, 1.30], unit="x gain", wave=None, dx="control: no change", leads=None),
    "ctrl_noise":  dict(levels=[0.01, 0.02, 0.04], unit="mV sd in TP", wave=None, dx="control: no change", leads=None),
    "ctrl_wander": dict(levels=[0.05, 0.10, 0.20], unit="mV", wave=None, dx="control: no change", leads=None),
}
# (f-wave mV, target RR CV). Evidence levels calibrated on 300 real PTB-XL AFIB recordings
# (folds 1-9): measured RR CV and V1 4-9 Hz activity of the edits match the 25th/50th/75th
# percentiles of real AF (RR CV 0.137/0.182/0.221; V1 activity 0.086/0.132/0.204 mV).
# Level-1 f-wave activity is set by the recording's own noise floor (~0.10 mV, 36th pct);
# the level-3 amplitude is extrapolated beyond the calibration grid (max 0.15).
AF_SEVERITY = {1: (0.02, 0.124), 2: (0.11, 0.164), 3: (0.21, 0.202)}

# ---------------- clinical criteria (guideline-based) ----------------
# ST elevation at the J point, Fourth Universal Definition of MI (Thygesen et al., 2018):
#   >= 0.1 mV in 2 contiguous leads, except V2-V3: >= 0.2 mV men >= 40 y,
#   >= 0.25 mV men < 40 y, >= 0.15 mV women.
# ST depression: >= 0.05 mV in 2 contiguous leads (same source).
# Long QT: QTc (Bazett) >= 450 ms men, >= 460 ms women (AHA/ACCF/HRS 2009 standardisation).
# Contiguity: precordial V1..V6 and the Cabrera limb sequence aVL, I, -aVR, II, aVF, III.
ST_ELEV_MV = 0.10
ST_DEP_MV = -0.05
T_INV_MV = -0.10
QRS_WIDE_MS = 120
RR_CV_AF = 0.08
P_ABSENT_FRAC = 0.30
PHYSIO_HZ = 40          # edits act on the <40 Hz component; the noise residual is kept


def st_threshold(lead, sex=None, age=None):
    if lead not in ("V2", "V3"):
        return ST_ELEV_MV
    if sex == "F":
        return 0.15
    if sex == "M" and age is not None and age < 40:
        return 0.25
    return 0.20                                   # men >= 40 y, or unknown (conservative)


def qtc_threshold(sex=None):
    return 450 if sex == "M" else 460             # unknown sex -> the higher threshold


def _ms(fs, ms):
    return int(round(ms * fs / 1000.0))


def _filt(x, fs, lo=None, hi=None, order=2):
    nyq = fs / 2
    if lo and hi:
        b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    elif lo:
        b, a = butter(order, lo / nyq, btype="high")
    else:
        b, a = butter(order, hi / nyq, btype="low")
    return filtfilt(b, a, x, axis=0)


def _smooth(v, w):
    w = max(int(w), 1)
    return np.convolve(v, np.ones(w) / w, mode="same")


# ----------------------------------------------------------------------------------
# 1. Delineation (global, multi-lead)
# ----------------------------------------------------------------------------------
def detect_r(sig, fs):
    """R peaks from the spatial derivative energy of all 12 leads (polarity independent)."""
    xf = _filt(sig, fs, 0.5, 40)
    d = _smooth(np.sqrt((np.diff(xf, axis=0, prepend=xf[:1]) ** 2).sum(1)), _ms(fs, 20))
    thr = 0.35 * np.percentile(d, 99)
    refr = _ms(fs, 250)
    peaks, i, n = [], 0, len(d)
    while i < n:
        if d[i] > thr:
            j = min(i + _ms(fs, 120), n)
            k = i + int(np.argmax(d[i:j]))
            peaks.append(k)
            i = k + refr
        else:
            i += 1
    # refine to the extreme of the vector magnitude of the baseline-free signal
    mag = np.sqrt((xf ** 2).sum(1))
    out = []
    for k in peaks:
        a, b = max(k - _ms(fs, 60), 0), min(k + _ms(fs, 60), n)
        out.append(a + int(np.argmax(mag[a:b])))
    return np.array(sorted(set(out)), dtype=int)


def delineate(sig, fs, r=None):
    """Return dict of per-beat fiducials (sample indices, NaN if not found).

    Keys: R, QRS_on, QRS_off (=J), T_on, T_peak, T_off, P_on, P_peak, P_off.
    Uses global (all-lead) derivative and magnitude curves, like the Uni-G/12SL
    global-measurement approach, so it is independent of any single lead's polarity.
    """
    n = sig.shape[0]
    xf = _filt(sig, fs, 0.5, 40)
    r = detect_r(sig, fs) if r is None else np.asarray(r, int)
    d = _smooth(np.sqrt((np.diff(xf, axis=0, prepend=xf[:1]) ** 2).sum(1)), _ms(fs, 16))
    keys = ["R", "QRS_on", "QRS_off", "T_on", "T_peak", "T_off", "P_on", "P_peak", "P_off"]
    F = {k: np.full(len(r), np.nan) for k in keys}
    F["R"] = r.astype(float)
    rr = np.diff(r) if len(r) > 1 else np.array([fs])
    rr_med = float(np.median(rr))

    for i, ri in enumerate(r):
        # ---------------- QRS on/off from derivative energy ----------------
        a = max(ri - _ms(fs, 60), 1)
        pk_on = a + int(np.argmax(d[a:ri + 1]))
        b = min(ri + _ms(fs, 90), n - 1)
        pk_off = ri + int(np.argmax(d[ri:b + 1]))
        dpk = d[a:b + 1].max()
        lo_on = d[max(ri - _ms(fs, 260), 0):max(ri - _ms(fs, 80), 1)]
        lo_off = d[min(ri + _ms(fs, 120), n - 1):min(ri + _ms(fs, 260), n)]
        if len(lo_on) < 5 or len(lo_off) < 5:
            continue
        thr_on = np.percentile(lo_on, 20) + 0.12 * (dpk - np.percentile(lo_on, 20))
        thr_off = np.percentile(lo_off, 20) + 0.12 * (dpk - np.percentile(lo_off, 20))
        k = pk_on
        while k > ri - _ms(fs, 160) and k > 0 and d[k] > thr_on:
            k -= 1
        qon = k
        k = pk_off
        while k < ri + _ms(fs, 200) and k < n - 1 and d[k] > thr_off:
            k += 1
        qoff = k
        F["QRS_on"][i], F["QRS_off"][i] = qon, qoff

        # ---------------- baseline-free magnitude for P and T ----------------
        pr0, pr1 = qon - _ms(fs, 30), qon - _ms(fs, 10)
        if pr0 < 0:
            continue
        base = xf[pr0:pr1].mean(0)
        rr_i = rr[i] if i < len(rr) else rr_med
        t_end = min(ri + int(min(0.62 * rr_i, _ms(fs, 560))), n - 1)
        t_beg = qoff + _ms(fs, 60)
        if t_end - t_beg < _ms(fs, 60):
            continue
        seg = xf[t_beg:t_end] - base
        mag = _smooth(np.sqrt((seg ** 2).sum(1)), _ms(fs, 20))
        tp = int(np.argmax(mag))
        m_iso = np.sqrt(((xf[pr0:pr1] - base) ** 2).sum(1)).mean()
        dm = np.gradient(mag)
        # T offset: tangent at steepest descent intersecting the isoelectric level
        k1 = min(tp + _ms(fs, 200), len(mag) - 1)
        if k1 > tp + 2:
            ks = tp + int(np.argmin(dm[tp:k1]))
            if dm[ks] < 0:
                toff = ks + (m_iso - mag[ks]) / dm[ks]
                F["T_off"][i] = t_beg + min(max(toff, ks), len(mag) + _ms(fs, 150))
        # T onset: tangent at steepest ascent before the peak
        if tp > 2:
            ka = int(np.argmax(dm[:tp]))
            if dm[ka] > 0:
                ton = ka - (mag[ka] - m_iso) / dm[ka]
                F["T_on"][i] = t_beg + max(ton, 0)
        F["T_peak"][i] = t_beg + tp
        if not np.isnan(F["T_off"][i]):
            F["T_off"][i] = min(F["T_off"][i], n - 1)

        # ---------------- P wave ----------------
        p_hi = qon - _ms(fs, 35)
        p_lo = qon - _ms(fs, 280)
        if i > 0 and not np.isnan(F["T_off"][i - 1]):
            p_lo = max(p_lo, int(F["T_off"][i - 1]) + _ms(fs, 20))
        p_lo = max(p_lo, 0)
        if p_hi - p_lo < _ms(fs, 60):
            continue
        pseg = xf[p_lo:p_hi] - base
        pm = _smooth(np.sqrt((pseg ** 2).sum(1)), _ms(fs, 20))
        pp = int(np.argmax(pm))
        # noise reference: high-passed residual in the same window
        noise = np.sqrt((_filt(pseg, fs, lo=15) ** 2).sum(1)).std() + 1e-6
        if pm[pp] - m_iso < max(0.06, 4 * noise):  # no P wave
            continue
        lvl = m_iso + 0.25 * (pm[pp] - m_iso)
        a_, b_ = pp, pp
        while a_ > 0 and pm[a_] > lvl:
            a_ -= 1
        while b_ < len(pm) - 1 and pm[b_] > lvl:
            b_ += 1
        F["P_on"][i], F["P_peak"][i], F["P_off"][i] = p_lo + a_, p_lo + pp, p_lo + b_
    return F


# ----------------------------------------------------------------------------------
# 2. Measurements (used for verification)
# ----------------------------------------------------------------------------------
def _pre_qrs_windows(x, fs, F):
    """detrended pre-QRS windows [QRS_on-240 ms, QRS_on-20 ms], beats x time x leads"""
    a_off, b_off = _ms(fs, 240), _ms(fs, 20)
    wins = []
    for q in F["QRS_on"]:
        if np.isnan(q):
            continue
        q = int(q)
        if q - a_off < 0:
            continue
        w = x[q - a_off:q - b_off].copy()
        # remove linear trend per lead (baseline wander)
        u = np.linspace(-1, 1, len(w))[:, None]
        w -= w.mean(0)
        w -= u * (w * u).sum(0) / (u * u).sum()
        wins.append(w)
    return np.array(wins) if len(wins) >= 3 else None


def phase_locking_index(x, fs, F):
    """PLI of the pre-QRS window (see p_wave_presence); nan if < 3 usable beats"""
    W = _pre_qrs_windows(x, fs, F)
    if W is None:
        return np.nan
    return float((W.mean(0) ** 2).sum() / ((W ** 2).sum((1, 2)).mean() + 1e-12))


def p_wave_presence(x, fs, F, corr_thr=0.5, amp_thr=0.04, pli_thr=0.5):
    """Fraction of beats with a sinus P wave.

    A sinus P wave is phase-locked to the QRS: the pre-QRS window
    [QRS_on-240 ms, QRS_on-20 ms] repeats from beat to beat. We build the median
    template across beats (all 12 leads, detrended) and call a beat 'P present' if it
    correlates with the template (r >= corr_thr) and the template holds a real
    deflection (peak-to-peak magnitude >= amp_thr mV). Fibrillatory waves are not
    phase-locked, so the template averages out and correlations drop."""
    W = _pre_qrs_windows(x, fs, F)
    if W is None:
        return np.nan
    # phase-locking index: power of the mean window / mean power of single windows.
    # ~1 for a repeated (sinus) P wave, ~1/N for activity not locked to the QRS (f-waves).
    # Needed because smooth 4-9 Hz f-waves in a 220 ms window have so few degrees of
    # freedom that per-beat correlations alone are fooled by chance (audit v2 finding).
    pli = float((W.mean(0) ** 2).sum() / ((W ** 2).sum((1, 2)).mean() + 1e-12))
    if pli < pli_thr:
        return 0.0
    present = []
    for k in range(len(W)):                  # leave-one-out template (avoids self-correlation)
        tmpl = np.median(np.delete(W, k, 0), 0)
        amp = np.sqrt((tmpl ** 2).sum(1)).max()
        r = np.corrcoef(W[k].ravel(), tmpl.ravel())[0, 1]
        present.append(amp >= amp_thr and r >= corr_thr)
    return float(np.mean(present))


def measure(sig, fs, F=None):
    """Global and per-lead measurements, median over beats."""
    F = delineate(sig, fs) if F is None else F
    x = _filt(sig, fs, hi=40)                 # no high-pass: keeps true ST level
    n = x.shape[0]
    st, stj, tamp, qrs, qt = [], [], [], [], []
    for i in range(len(F["R"])):
        qon, qoff = F["QRS_on"][i], F["QRS_off"][i]
        if np.isnan(qon) or np.isnan(qoff):
            continue
        qon, qoff = int(qon), int(qoff)
        b0, b1 = qon - _ms(fs, 30), qon - _ms(fs, 10)
        j60 = qoff + _ms(fs, 60)
        if b0 < 0 or j60 >= n:
            continue
        base = x[b0:b1].mean(0)
        st.append(x[j60] - base)
        stj.append(x[qoff:qoff + _ms(fs, 20)].mean(0) - base)   # J point (0-20 ms)
        qrs.append((qoff - qon) * 1000 / fs)
        if not np.isnan(F["T_peak"][i]):
            tpk = int(F["T_peak"][i])
            w = _ms(fs, 20)
            tamp.append(x[max(tpk - w, 0):tpk + w].mean(0) - base)
        if not np.isnan(F["T_off"][i]):
            qt.append((F["T_off"][i] - qon) * 1000 / fs)
    rr = np.diff(F["R"]) * 1000 / fs
    rr_med = float(np.median(rr)) if len(rr) else np.nan
    qt_med = float(np.median(qt)) if qt else np.nan
    m = {
        "n_beats": len(F["R"]),
        "hr_bpm": 60000 / rr_med if rr_med == rr_med else np.nan,
        "rr_cv": float(np.std(rr) / np.mean(rr)) if len(rr) > 2 else np.nan,
        "qrs_ms": float(np.median(qrs)) if qrs else np.nan,
        "qt_ms": qt_med,
        "qtc_ms": qt_med / np.sqrt(rr_med / 1000) if qt else np.nan,             # Bazett
        "qtcf_ms": qt_med / np.cbrt(rr_med / 1000) if qt else np.nan,            # Fridericia
        "p_present_frac": p_wave_presence(x, fs, F),
    }
    st_l = np.median(np.array(st), 0) if st else np.full(12, np.nan)
    stj_l = np.median(np.array(stj), 0) if stj else np.full(12, np.nan)
    t_l = np.median(np.array(tamp), 0) if tamp else np.full(12, np.nan)
    for j, name in enumerate(LEADS):
        m[f"st_{name}"] = float(st_l[j])      # J+60 ms
        m[f"stj_{name}"] = float(stj_l[j])    # J point
        m[f"t_{name}"] = float(t_l[j])
    return m


def _contig(flags):
    """max run of consecutive True in an ordered lead group"""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"]
CABRERA = ["aVL", "I", "-aVR", "II", "aVF", "III"]


def _v(m, key, lead):
    return -m[f"{key}_aVR"] if lead == "-aVR" else m[f"{key}_{lead}"]


def findings(m, sex=None, age=None):
    """Guideline-based criteria from measurements (booleans). sex: 'M'/'F'/None."""
    st_e = max(_contig([_v(m, "stj", l) >= st_threshold(l, sex, age) for l in grp])
               for grp in (PRECORDIAL, CABRERA)) >= 2
    # ST depression must be horizontal or down-sloping (guideline wording): depressed at the
    # J point AND still depressed at J+60 ms. An up-sloping J-point depression is a normal
    # variant (v2 counted it: main source of false off-target findings on real PTB-XL).
    st_d = max(_contig([(_v(m, "stj", l) <= ST_DEP_MV) and (_v(m, "st", l) <= ST_DEP_MV) for l in grp])
               for grp in (PRECORDIAL, CABRERA)) >= 2
    # T may be negative in normals in V1, III (and aVR, handled as -aVR): excluded
    t_i = max(_contig([_v(m, "t", l) <= T_INV_MV for l in grp])
              for grp in (PRECORDIAL[1:], CABRERA[:5])) >= 2
    return {
        "ST_elevation": bool(st_e),
        "ST_depression": bool(st_d),
        "T_inversion": bool(t_i),
        "long_QT": bool(m["qtc_ms"] >= qtc_threshold(sex)),
        "wide_QRS": bool(m["qrs_ms"] >= QRS_WIDE_MS),
        "AF_pattern": bool((m["p_present_frac"] <= P_ABSENT_FRAC) and (m["rr_cv"] >= RR_CV_AF)),
    }


TARGET_FINDING = {"st_recip": "ST_elevation", "st_elev": "ST_elevation", "t_inv": "T_inversion", "qt_long": "long_QT",
                  "qrs_wide": "wide_QRS", "af": "AF_pattern"}


# ----------------------------------------------------------------------------------
# 3. Perturbations
# ----------------------------------------------------------------------------------
def _valid(F, i, *keys):
    return all(not np.isnan(F[k][i]) for k in keys)


def _half_cos(n, rising=True):
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, max(n, 1)))
    return w if rising else w[::-1]


def _warp(sig, segs):
    """Piecewise-linear time warp. segs: list of (start, end, delta) in samples on the
    original time axis, non-overlapping and sorted; each segment's length changes by
    delta and everything after it shifts. Total delta must be 0 (length preserved)."""
    n = sig.shape[0]
    ko, kn, c = [0.0], [0.0], 0.0
    for s, e, dl in segs:
        ko += [s, e]
        kn += [s + c, e + c + dl]
        c += dl
    ko.append(n - 1.0)
    kn.append(n - 1.0 + c)
    ko, kn = np.array(ko), np.array(kn)
    keep = np.concatenate([[True], np.diff(ko) > 0])
    ko, kn = ko[keep], kn[keep]
    src = np.interp(np.arange(n), kn, ko)
    idx = np.arange(n)
    return np.stack([np.interp(src, idx, sig[:, j]) for j in range(sig.shape[1])], 1)


def _split(sig, fs):
    """physiological component (<PHYSIO_HZ) and noise residual. Edits act on the first;
    the residual (muscle noise, mains, electrode noise) is added back untouched so the
    edited ECG keeps the recording's own noise floor."""
    s = _filt(sig, fs, hi=PHYSIO_HZ, order=4)
    return s, sig - s


def _warp_keep_noise(sig, segs, fs, qrs_windows=()):
    """Time-warp the physiological component only. Inside QRS windows (whose
    high-frequency content belongs to the QRS itself) the residual is warped too."""
    s, r = _split(sig, fs)
    out = _warp(s, segs)
    if not qrs_windows:
        return out + r
    mask = np.zeros((sig.shape[0], 1))
    for a, b in qrs_windows:
        a, b = max(int(a), 0), min(int(b), sig.shape[0] - 1)
        mask[a:b + 1] = 1.0
    mask = _filt(mask, fs, hi=25, order=2).clip(0, 1)      # soft edges
    m_new = _warp(mask, segs)
    return out + (1 - m_new) * r + m_new * _warp(r, segs)


def _next_boundary(F, i, fs, n):
    """End of the TP segment after beat i (just before the next P wave, or QRS if no P)."""
    if i + 1 < len(F["R"]):
        if not np.isnan(F["P_on"][i + 1]):
            return F["P_on"][i + 1] - _ms(fs, 15)
        if not np.isnan(F["QRS_on"][i + 1]):
            return F["QRS_on"][i + 1] - _ms(fs, 60)
        return None
    return n - 1 - _ms(fs, 5)


def st_elevation(sig, fs, F, amp_mv, leads):
    y = sig.copy()
    idx = [L[l] for l in leads]
    ramp = _ms(fs, 20)
    done = 0
    for i in range(len(F["R"])):
        if not _valid(F, i, "QRS_off", "T_peak", "T_off"):
            continue
        # elevation reaches its full value AT the J point (guideline measurement point):
        # ramp over the terminal 20 ms of the QRS, plateau to the T peak, taper to T end
        j, tp, to = int(F["QRS_off"][i]), int(F["T_peak"][i]), int(F["T_off"][i])
        a = j - ramp
        if not (a > 0 and j < tp < to <= len(y) - 1):
            continue
        w = np.zeros(to - a + 1)
        w[:ramp] = _half_cos(ramp, True)
        w[ramp:tp - a] = 1.0
        w[tp - a:] = _half_cos(to - tp + 1, False)
        y[a:to + 1, idx] += amp_mv * w[:, None]
        done += 1
    return y, {"beats_modified": done}


def _iso(sig, fs, F, i):
    """Isoelectric level per lead from the PR segment of beat i (None if unavailable)."""
    if np.isnan(F["QRS_on"][i]):
        return None
    q = int(F["QRS_on"][i])
    a, b = q - _ms(fs, 30), q - _ms(fs, 10)
    return sig[a:b].mean(0) if a >= 0 else None


def t_inversion(sig, fs, F, factor, leads):
    """Mirror the T wave: T -> -factor * T, on the physiological component.
    Reference line: from the ST level at the start of the edit window to the
    isoelectric level after the T wave (next beat's PR segment interpolated, robust to
    wander). Anchoring on the ST level leaves the ST segment unchanged, so no
    artificial ST depression is created when the ST segment is elevated (v1 bug)."""
    s, res = _split(sig, fs)
    sig = s
    y = s.copy()
    n = len(y)
    idx = [L[l] for l in leads]
    done = 0
    for i in range(len(F["R"])):
        if not _valid(F, i, "QRS_off", "T_on", "T_off"):
            continue
        iso0 = _iso(sig, fs, F, i)
        if iso0 is None:
            continue
        iso1 = _iso(sig, fs, F, i + 1) if i + 1 < len(F["R"]) else None
        iso1 = iso0 if iso1 is None else iso1
        a = max(int(F["T_on"][i]) - _ms(fs, 30), int(F["QRS_off"][i]) + _ms(fs, 20))
        c = _next_boundary(F, i, fs, n)
        b = int(F["T_off"][i]) + _ms(fs, 50)
        if c is not None:
            b = min(b, int(c))
        b = min(b, n - 1)
        if b - a < _ms(fs, 100):
            continue
        tap = _ms(fs, 30)
        w = np.ones(b - a + 1)
        w[:tap] = _half_cos(tap, True)
        w[-tap:] = _half_cos(tap, False)
        q0 = int(F["QRS_on"][i])
        q1 = int(F["QRS_on"][i + 1]) if (i + 1 < len(F["R"]) and not np.isnan(F["QRS_on"][i + 1])) else q0 + n
        # isoelectric level at the end of the window (interpolated between PR segments)
        u_b = (b - q0) / max(q1 - q0, 1)
        iso_b = iso0 + (iso1 - iso0) * u_b
        st_a = sig[max(a - _ms(fs, 5), 0):a + _ms(fs, 5)].mean(0)      # ST level at start
        v = np.linspace(0, 1, b - a + 1)[:, None]
        ref = st_a[None, :] + (iso_b - st_a)[None, :] * v
        for j in idx:
            dev = sig[a:b + 1, j] - ref[:, j]
            y[a:b + 1, j] = sig[a:b + 1, j] - (1 + factor) * w * dev
        done += 1
    return y + res, {"beats_modified": done}


def _stretch(sig, fs, F, delta_ms, start_key, end_key, min_tp_ms=30):
    n = sig.shape[0]
    dl = _ms(fs, delta_ms)
    segs, got = [], []
    for i in range(len(F["R"])):
        if not _valid(F, i, start_key, end_key, "T_off"):
            continue
        s, e, toff = F[start_key][i], F[end_key][i], F["T_off"][i]
        c = _next_boundary(F, i, fs, n)
        if c is None or c <= toff:
            continue
        d = min(dl, (c - toff) - _ms(fs, min_tp_ms))
        if d <= 0:
            continue
        segs += [(s, e, d), (toff, c, -d)]
        got.append(d * 1000 / fs)
    segs.sort(key=lambda t: t[0])
    # drop overlapping segments (e.g. when end_key == T_off)
    clean, last_end = [], -1
    for s, e, d in segs:
        if s >= last_end:
            clean.append((s, e, d))
            last_end = e
    return _warp_keep_noise(sig, clean, fs), {"beats_modified": len(got),
                                              "achieved_ms_mean": float(np.mean(got)) if got else 0.0}


def qt_prolong(sig, fs, F, delta_ms):
    return _stretch(sig, fs, F, delta_ms, "QRS_off", "T_off")


def qrs_widen(sig, fs, F, delta_ms):
    # stretch QRS, shift ST-T, compress the following TP segment
    n = sig.shape[0]
    dl = _ms(fs, delta_ms)
    segs, got = [], []
    for i in range(len(F["R"])):
        if not _valid(F, i, "QRS_on", "QRS_off", "T_off"):
            continue
        c = _next_boundary(F, i, fs, n)
        if c is None or c <= F["T_off"][i]:
            continue
        d = min(dl, (c - F["T_off"][i]) - _ms(fs, 30))
        if d <= 0:
            continue
        # include a 12 ms margin on both sides so the low-slope QRS tails are stretched too
        s0 = F["QRS_on"][i] - _ms(fs, 12)
        e0 = F["QRS_off"][i] + _ms(fs, 12)
        d = int(round(d * (e0 - s0) / (F["QRS_off"][i] - F["QRS_on"][i])))
        d = min(d, int((c - F["T_off"][i]) - _ms(fs, 30)))
        segs += [(s0, e0, d), (F["T_off"][i], c, -d)]
        got.append(d * (F["QRS_off"][i] - F["QRS_on"][i]) / (e0 - s0) * 1000 / fs)
    segs.sort(key=lambda t: t[0])
    qrs_win = [(s0 - _ms(fs, 10), e0 + _ms(fs, 10)) for s0, e0, d in segs if d > 0]
    return _warp_keep_noise(sig, segs, fs, qrs_win), {
        "beats_modified": len(got), "achieved_ms_mean": float(np.mean(got)) if got else 0.0}


def atrial_fibrillation(sig, fs, F, severity, rng):
    f_amp, cv = AF_SEVERITY[int(severity)]
    n = sig.shape[0]
    s, res = _split(sig, fs)
    y = s.copy()
    # (1) remove P waves from the physiological component (smooth bridge); the recording's
    #     own noise residual is kept, so no synthetic noise is injected (v1 injected 1.5x)
    removed = 0
    for i in range(len(F["R"])):
        if not _valid(F, i, "P_on", "P_off", "QRS_on"):
            continue
        a = max(int(F["P_on"][i]) - _ms(fs, 15), 0)
        b = min(int(F["P_off"][i]) + _ms(fs, 15), int(F["QRS_on"][i]) - _ms(fs, 5))
        if b - a < 5:
            continue
        u = _half_cos(b - a + 1, True)[:, None]
        y[a:b + 1] = y[a] + (y[b] - y[a]) * u
        removed += 1
    y = y + res
    # (2) irregular RR: stretch/compress the (now flat) T_off -> next QRS_on segments
    segs = []
    rr_mean = np.median(np.diff(F["R"])) if len(F["R"]) > 1 else fs
    for i in range(len(F["R"]) - 1):
        if not (_valid(F, i, "T_off") and not np.isnan(F["QRS_on"][i + 1])):
            continue
        s, e = F["T_off"][i] + _ms(fs, 10), F["QRS_on"][i + 1] - _ms(fs, 20)
        if e - s > _ms(fs, 60):
            segs.append([s, e, 0.0])
    if len(segs) >= 3:
        lens = np.array([e - s for s, e, _ in segs])
        lo = _ms(fs, 50) - lens                    # keep >= 50 ms of baseline
        d = rng.standard_normal(len(segs))
        for _ in range(60):                        # zero-sum deltas, exact target RR CV
            d = d - d.mean()
            d = d * (cv * 1.1 * rr_mean) / (d.std() + 1e-9)
            d = np.maximum(d, lo)
        d -= d.mean()
        for k in range(len(segs)):
            segs[k][2] = float(d[k])
        y = _warp_keep_noise(y, [tuple(sg) for sg in segs], fs)
    # (3) fibrillatory f-waves: irregular band-limited (4-9 Hz) activity, largest in V1
    #     and the inferior leads; f_amp = 95th percentile of |f| in V1
    t = np.arange(n) / fs
    f = _filt(rng.standard_normal(n + 2 * fs), fs, 4, 9, order=3)[fs:fs + n]
    f *= 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t)
    f = f_amp * f / np.percentile(np.abs(f), 95)
    wts = np.array([0.4, 0.7, 0.7, -0.5, -0.2, 0.7, 1.0, 0.8, 0.5, 0.4, 0.3, 0.3])
    y += f[:, None] * wts[None, :]
    return y, {"p_waves_removed": removed, "rr_segments_warped": len(segs)}


def control(sig, fs, F, kind, level, rng):
    n = sig.shape[0]
    if kind == "ctrl_amp":
        return sig * level, {}
    if kind == "ctrl_wander":
        t = np.arange(n) / fs
        ph = rng.uniform(0, 6.28, 12)
        return sig + level * np.sin(2 * np.pi * 0.25 * t[:, None] + ph[None, :]), {}
    if kind == "ctrl_noise":
        y = sig.copy()
        for i in range(len(F["R"])):
            if not _valid(F, i, "T_off"):
                continue
            c = _next_boundary(F, i, fs, n)
            if c is None:
                continue
            a, b = int(F["T_off"][i]) + _ms(fs, 20), int(c) - _ms(fs, 20)
            if b > a:
                y[a:b] += level * rng.standard_normal((b - a, 12))
        return y, {}
    raise ValueError(kind)


def perturb(sig, fs, F, kind, level, seed=0):
    rng = np.random.default_rng(seed)
    spec = CATALOGUE[kind]
    if kind == "st_elev":
        return st_elevation(sig, fs, F, level, spec["leads"])
    if kind == "st_recip":
        y, info = st_elevation(sig, fs, F, level, spec["leads"])
        y, _ = st_elevation(y, fs, F, -spec["recip_factor"] * level, spec["recip_leads"])
        return y, info
    if kind == "t_inv":
        return t_inversion(sig, fs, F, level, spec["leads"])
    if kind == "qt_long":
        return qt_prolong(sig, fs, F, level)
    if kind == "qrs_wide":
        return qrs_widen(sig, fs, F, level)
    if kind == "af":
        return atrial_fibrillation(sig, fs, F, level, rng)
    return control(sig, fs, F, kind, level, rng)


# ----------------------------------------------------------------------------------
# 4. Verification: re-delineate the edited signal from scratch and re-measure
# ----------------------------------------------------------------------------------
def verify(m0, sig1, fs, kind, level, sex=None, age=None, achieved=None):
    """m0: measurements of the original. Returns dict with measurement deltas,
    `impl_ok` (edit implemented as specified), `criterion_met` (guideline threshold of
    the target finding reached for this sex/age), off-target findings, and the final
    `keep` decision with an explicit `reject_reason`.
    achieved: for time-warp edits, the stretch actually applied (ms); it can be smaller
    than requested when the TP segment is too short (fast heart rate)."""
    m1 = measure(sig1, fs)
    f0, f1 = findings(m0, sex, age), findings(m1, sex, age)
    spec = CATALOGUE[kind]
    out = {"kind": kind, "level": level}
    new = {k: (f1[k] and not f0[k]) for k in f1}
    target = TARGET_FINDING.get(kind)
    allowed = set(spec.get("secondary", []))
    out["criterion_met"] = bool(f1[target]) if target else False
    out["off_target_new_findings"] = ",".join(k for k, v in new.items()
                                              if v and k != target and k not in allowed)
    out["expected_secondary_findings"] = ",".join(k for k, v in new.items() if v and k in allowed)
    ach = level if achieved is None else achieved

    def dl(key):
        return m1[key] - m0[key]
    out["d_qrs_ms"], out["d_qt_ms"], out["d_qtc_ms"] = dl("qrs_ms"), dl("qt_ms"), dl("qtc_ms")
    out["d_rr_cv"], out["d_p_present"] = dl("rr_cv"), dl("p_present_frac")
    out["rr_cv"], out["p_present_frac"] = m1["rr_cv"], m1["p_present_frac"]
    out["qtc_ms"], out["qrs_ms"] = m1["qtc_ms"], m1["qrs_ms"]
    if spec["leads"]:
        tl = spec["leads"]
        if kind in ("st_elev", "st_recip"):
            d = np.array([dl(f"stj_{l}") for l in tl])
            out["d_st_target_mv"] = float(d.mean())
            out["impl_ok"] = bool(abs(d.mean() - level) <= 0.03)
            if kind == "st_recip":
                dr = np.array([dl(f"stj_{l}") for l in spec["recip_leads"]])
                out["d_st_recip_mv"] = float(dr.mean())
                out["impl_ok"] = out["impl_ok"] and bool(abs(dr.mean() + spec["recip_factor"] * level) <= 0.03)
        if kind == "t_inv":
            t0 = np.array([m0[f"t_{l}"] for l in tl])
            t1 = np.array([m1[f"t_{l}"] for l in tl])
            ok = t0 > 0.1
            ratio = (t1[ok] / t0[ok]) if ok.any() else np.array([np.nan])
            out["t_ratio_target"] = float(np.median(ratio))
            out["impl_ok"] = bool(ok.any() and abs(np.median(ratio) + level) <= 0.35 * level)
    elif kind == "qt_long":
        # compare the measurement with the stretch actually applied
        out["impl_ok"] = bool(abs(out["d_qt_ms"] - ach) <= max(20, 0.25 * ach)
                              and abs(out["d_qrs_ms"]) <= 10)
    elif kind == "qrs_wide":
        # the time-warp is exact by construction; the independent re-measurement is
        # conservative for stretched waves (~0.75x), so accept 0.6x-1.2x of the stretch
        out["impl_ok"] = bool(0.6 * ach <= out["d_qrs_ms"] <= 1.2 * ach)
    elif kind == "af":
        out["impl_ok"] = bool(m1["p_present_frac"] <= P_ABSENT_FRAC and m1["rr_cv"] >= RR_CV_AF)
    else:  # controls: must not create any finding nor move key measurements
        # a gain change legitimately scales an existing ST offset: widen tolerance for it
        tol = 0.05 + (abs(level - 1) * max(abs(m0[f"stj_{l}"]) for l in PRECORDIAL)
                      if kind == "ctrl_amp" else 0)
        stab = (abs(out["d_qrs_ms"]) <= 10 and abs(out["d_qt_ms"]) <= 20 and
                max(abs(dl(f"stj_{l}")) for l in PRECORDIAL) <= tol)
        out["impl_ok"] = bool(stab and not any(new.values()))
    # ---- final decision, with an explicit reason (no guessing when analysing rejects)
    reasons = []
    if kind in ("qt_long", "qrs_wide") and ach < 0.8 * level:
        reasons.append(f"capped({ach:.0f}/{level:g}ms)")
    if not out.get("impl_ok", False):
        reasons.append("impl")
    if out["off_target_new_findings"]:
        reasons.append("off_target:" + out["off_target_new_findings"])
    out["achieved"] = ach
    out["reject_reason"] = ";".join(reasons)
    out["keep"] = not reasons
    return out, m1
