"""Audit of the ECG-CF toolkit (held-out data, stress conditions, independence, artifacts).

python audit.py --n 60 --out audit_results

Rules followed:
 * held-out: seeds 10000+ were never used while developing/tuning thresholds (0-39 were);
 * stress: 4 acquisition conditions, from clean to hard;
 * every rate is reported with a Wilson 95% confidence interval.
"""
import argparse
import inspect
import os
import warnings

import numpy as np
import pandas as pd

import ecg_cf as E
from synthetic import make_normal_ecg

warnings.filterwarnings("ignore")

CONDITIONS = {
    "clean":     dict(noise=0.008, wander=0.03, emg=0.0, powerline=0.0, t_scale=1.0),
    "realistic": dict(noise=0.008, wander=0.08, emg=0.015, powerline=0.01, t_scale=1.0),
    "hard":      dict(noise=0.010, wander=0.15, emg=0.030, powerline=0.02, t_scale=1.0),
    "low_T":     dict(noise=0.008, wander=0.08, emg=0.015, powerline=0.01, t_scale=0.45),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, c - h), min(1, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100 * p:5.1f}% [{100 * lo:4.1f}-{100 * hi:5.1f}] (n={n})"


def _kw(fn, **kw):
    """pass only the keyword args a function accepts (works with v1 and v2 APIs)"""
    ok = inspect.signature(fn).parameters
    return {k: v for k, v in kw.items() if k in ok}


def demographics(rng):
    sex = "M" if rng.random() < 0.52 else "F"
    age = int(rng.integers(18, 85))
    # benign early repolarisation: frequent in young men, rarer otherwise
    if sex == "M" and age < 40:
        er = rng.uniform(0.05, 0.20) if rng.random() < 0.6 else 0.0
    else:
        er = rng.uniform(0.03, 0.10) if rng.random() < 0.2 else 0.0
    return sex, age, er


# ------------------------------------------------------------------ A. held-out stress test
def run_stress(n, seed0=10000):
    rows, base_rows = [], []
    for cname, cond in CONDITIONS.items():
        for k in range(n):
            seed = seed0 + k
            rng = np.random.default_rng(seed)
            sex, age, er = demographics(rng)
            sig, _ = make_normal_ecg(seed, early_repol=er, hr_range=(50, 90), **cond)
            fs = 500
            F = E.delineate(sig, fs)
            m0 = E.measure(sig, fs, F)
            f0 = E.findings(m0, **_kw(E.findings, sex=sex, age=age))
            flagged = [a for a, v in f0.items() if v]
            base_rows.append(dict(cond=cname, seed=seed, sex=sex, age=age, early_repol=er,
                                  flagged=",".join(flagged)))
            if flagged:
                continue
            for kind, spec in E.CATALOGUE.items():
                for li, lv in enumerate(spec["levels"], 1):
                    y, info = E.perturb(sig, fs, F, kind, lv, seed=seed * 10 + li)
                    v, m1 = E.verify(m0, y, fs, kind, lv, **_kw(E.verify, sex=sex, age=age,
                                     achieved=info.get("achieved_ms_mean")))
                    rows.append(dict(cond=cname, seed=seed, sex=sex, age=age, kind=kind,
                                     level_idx=li, impl_ok=v["impl_ok"],
                                     criterion_met=v["criterion_met"],
                                     off_target=v["off_target_new_findings"] or "",
                                     keep=v.get("keep", np.nan),
                                     reason=v.get("reject_reason", "")))
    return pd.DataFrame(rows), pd.DataFrame(base_rows)


# ------------------------------------------------------------------ B. independence check
def run_independent(n, seed0=20000):
    """Re-measure QT and QRS changes with neurokit2's wavelet delineator on lead II
    (a different algorithm from ecg_cf's own multi-lead delineator)."""
    import neurokit2 as nk
    fs, rows = 500, []
    for k in range(n):
        sig, _ = make_normal_ecg(seed0 + k, **CONDITIONS["realistic"])
        F = E.delineate(sig, fs)

        def nk_meas(x):
            try:
                xc = nk.ecg_clean(x[:, 1], sampling_rate=fs)
                _, info = nk.ecg_peaks(xc, sampling_rate=fs)
                _, w = nk.ecg_delineate(xc, info["ECG_R_Peaks"], sampling_rate=fs, method="dwt")
                on, off = np.array(w["ECG_R_Onsets"], float), np.array(w["ECG_R_Offsets"], float)
                toff = np.array(w["ECG_T_Offsets"], float)
                qrs = np.nanmedian(off - on) * 1000 / fs
                qt = np.nanmedian(toff - on) * 1000 / fs
                return qrs, qt
            except Exception:
                return np.nan, np.nan
        q0, t0 = nk_meas(sig)
        for kind in ("qt_long", "qrs_wide"):
            for lv in E.CATALOGUE[kind]["levels"]:
                y, _ = E.perturb(sig, fs, F, kind, lv, seed=k)
                q1, t1 = nk_meas(y)
                m0, m1 = E.measure(sig, fs, F), E.measure(y, fs)
                rows.append(dict(seed=seed0 + k, kind=kind, level=lv,
                                 nk_d_qrs=q1 - q0, nk_d_qt=t1 - t0,
                                 own_d_qrs=m1["qrs_ms"] - m0["qrs_ms"],
                                 own_d_qt=m1["qt_ms"] - m0["qt_ms"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C. artifact leakage
def hf_features(x, fs=500):
    """generic, wave-agnostic signal statistics an ML model could pick up"""
    from scipy.signal import welch
    f, p = welch(x, fs=fs, nperseg=512, axis=0)
    tot = p.sum(0)
    hf = p[(f > 40) & (f < 100)].sum(0) / tot
    vhf = p[f >= 100].sum(0) / tot
    d2 = np.abs(np.diff(x, 2, axis=0))
    return np.concatenate([np.log(hf + 1e-12), np.log(vhf + 1e-12),
                           np.log(np.percentile(d2, 99, axis=0) + 1e-12),
                           np.log(np.median(d2, axis=0) + 1e-12)])


def run_artifacts(n, seed0=30000):
    """Non-physiological fingerprint: an edit of the ECG waveform should not change the
    recording's noise floor. We compare the >100 Hz residual (where ECG content is
    negligible) before/after, restricted to the samples the edit modified.
    1.00 = noise floor untouched. (A first version of this test used a classifier on
    generic spectral features; it flagged every edit, including controls, because any
    change is detectable - the question is whether the trace is non-physiological.)"""
    fs, rows = 500, []
    for k in range(n):
        sig, _ = make_normal_ecg(seed0 + k, **CONDITIONS["realistic"])
        F = E.delineate(sig, fs)
        h0 = E._filt(sig, fs, lo=100, order=4)
        for kind, spec in E.CATALOGUE.items():
            y, _ = E.perturb(sig, fs, F, kind, spec["levels"][1], seed=k)
            h1 = E._filt(y, fs, lo=100, order=4)
            ch = np.abs(y - sig).max(1) > 1e-6
            loc = h1[ch].std() / h0[ch].std() if ch.sum() > 50 else np.nan
            rows.append(dict(seed=seed0 + k, kind=kind, vhf_ratio_edited_samples=loc,
                             vhf_ratio_record=h1.std() / h0.std()))
    d = pd.DataFrame(rows)
    return d.groupby("kind", sort=False)[["vhf_ratio_edited_samples", "vhf_ratio_record"]].median().reset_index()


# ------------------------------------------------------------------ D. targeted side effects
def run_side_effects(n, seed0=40000):
    """(i) ST shift created by T inversion; (ii) noise level injected by AF P-removal."""
    fs, rows = 500, []
    for k in range(n):
        rng = np.random.default_rng(seed0 + k)
        er = rng.uniform(0.05, 0.15)   # the worst case: pre-existing ST elevation
        sig, _ = make_normal_ecg(seed0 + k, early_repol=er, **CONDITIONS["realistic"])
        F = E.delineate(sig, fs)
        m0 = E.measure(sig, fs, F)
        y, _ = E.perturb(sig, fs, F, "t_inv", 1.5, seed=k)
        m1 = E.measure(y, fs)
        key = "stj_" if "stj_V2" in m0 else "st_"
        dst = min(m1[f"{key}{l}"] - m0[f"{key}{l}"] for l in ["V2", "V3", "V4", "V5"])
        # AF: high-frequency noise inside the removed P window vs the TP segment
        ya, _ = E.perturb(sig, fs, F, "af", 1, seed=k)
        hp_o, hp_a = E._filt(sig, fs, lo=20), E._filt(ya, fs, lo=20)
        tp_sd, p_sd = [], []
        for i in range(1, len(F["R"]) - 1):
            if np.isnan(F["P_on"][i]) or np.isnan(F["T_off"][i - 1]):
                continue
            a, b = int(F["P_on"][i]), int(F["P_off"][i])
            tp_sd.append(hp_o[int(F["T_off"][i - 1]) + 20:a - 20].std())
            p_sd.append(hp_a[a:b].std())
        rows.append(dict(seed=seed0 + k, early_repol=er, tinv_max_st_drop_mv=dst,
                         af_noise_ratio=np.median(p_sd) / np.median(tp_sd) if tp_sd else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="audit_results")
    ap.add_argument("--seed_offset", type=int, default=0,
                    help="shift all seeds (use a new value for a fresh, never-seen test set)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = []

    df, base = run_stress(a.n, 10000 + a.seed_offset)
    df.to_csv(f"{a.out}/stress_rows.csv", index=False)
    base.to_csv(f"{a.out}/stress_base.csv", index=False)
    rep.append(f"## A. Held-out stress test (seed offset {a.seed_offset}; seeds never used for tuning)\n")
    rep.append("Baseline normals flagged as abnormal (-> excluded):")
    for c in CONDITIONS:
        b = base[base.cond == c]
        rep.append(f"  {c:9s}: {fmt((b.flagged != '').sum(), len(b))}")
    yer = base[(base.sex == "M") & (base.age < 40) & (base.early_repol > 0)]
    rep.append(f"  young men with benign early repolarisation: {fmt((yer.flagged != '').sum(), len(yer))}"
               f"  <- selection-bias check")
    rep.append("\nEdit implemented as specified (impl_ok), all conditions pooled:")
    for kind in E.CATALOGUE:
        for li in (1, 2, 3):
            s = df[(df.kind == kind) & (df.level_idx == li)]
            rep.append(f"  {kind:11s} L{li}: impl {fmt(s.impl_ok.sum(), len(s))} | "
                       f"criterion {fmt(s.criterion_met.sum(), len(s))} | "
                       f"off-target {fmt((s.off_target != '').sum(), len(s))}")
    if "keep" in df:
        rep.append("\nFinal keep decision (impl OK, not capped, no off-target) and reject reasons:")
        for kind in E.CATALOGUE:
            s_ = df[df.kind == kind]
            rr = s_.reason.fillna("").str.split(";").explode()
            rr = rr[rr != ""].str.split("(").str[0].value_counts().to_dict()
            rep.append(f"  {kind:11s}: keep {fmt(int(s_.keep.sum()), len(s_))} reasons={rr}")
    rep.append("\nimpl_ok by condition (pathological edits, all levels):")
    pat = df[~df.kind.str.startswith("ctrl")]
    for c in CONDITIONS:
        s = pat[pat.cond == c]
        rep.append(f"  {c:9s}: {fmt(s.impl_ok.sum(), len(s))}")
    ctl = df[df.kind.str.startswith("ctrl")]
    rep.append(f"controls, any false finding: {fmt((ctl.off_target != '').sum(), len(ctl))}")

    ind = run_independent(max(20, a.n // 3), 20000 + a.seed_offset)
    ind.to_csv(f"{a.out}/independent.csv", index=False)
    rep.append("\n## B. Independent re-measurement (neurokit2 wavelet delineator, lead II)\n")
    g = ind.groupby(["kind", "level"]).agg(
        nk_qrs=("nk_d_qrs", "median"), own_qrs=("own_d_qrs", "median"),
        nk_qt=("nk_d_qt", "median"), own_qt=("own_d_qt", "median"),
        nk_valid=("nk_d_qt", lambda s: s.notna().mean())).round(1)
    rep.append(g.to_string())

    art = run_artifacts(max(30, a.n // 2), 30000 + a.seed_offset)
    art.to_csv(f"{a.out}/artifacts.csv", index=False)
    rep.append("\n## C. Noise-floor fingerprint (>100 Hz residual, edited/original; 1.00 = untouched)\n"
               "ctrl_amp and ctrl_noise change the noise by design (gain, added noise).")
    rep.append(art.round(3).to_string(index=False))

    se = run_side_effects(max(20, a.n // 3), 40000 + a.seed_offset)
    se.to_csv(f"{a.out}/side_effects.csv", index=False)
    rep.append("\n## D. Targeted side effects\n")
    rep.append(f"  T inversion x1.5 on ECGs with early repolarisation: worst ST change in V2-V5 "
               f"median {se.tinv_max_st_drop_mv.median():.3f} mV, min {se.tinv_max_st_drop_mv.min():.3f} mV")
    rep.append(f"  AF P-window high-frequency noise / TP noise: median {se.af_noise_ratio.median():.2f} "
               f"(1.0 = matched)")
    txt = "\n".join(rep)
    open(f"{a.out}/audit_report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
