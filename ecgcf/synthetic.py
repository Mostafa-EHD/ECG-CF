"""Synthetic 12-lead normal ECG generator (sum-of-Gaussians beat model).

Used only to validate the perturbation toolkit when PTB-XL is not at hand.
Output format matches PTB-XL records500: shape (N, 12), millivolts, 500 Hz,
lead order I, II, III, aVR, aVL, aVF, V1..V6.
"""
import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# wave: (centre s relative to R at RR=1 s, sigma_rise s, sigma_fall s)
_WAVES = {
    "P": (-0.165, 0.022, 0.022),
    "Q": (-0.024, 0.008, 0.008),
    "R": (0.000, 0.009, 0.009),
    "S": (0.024, 0.009, 0.009),
    "T": (0.270, 0.055, 0.035),
}
# amplitudes (mV) for I, II and the precordial leads; other limb leads derived (Einthoven/Goldberger)
_AMP = {
    "I":  dict(P=0.08, Q=-0.05, R=0.70, S=-0.10, T=0.25),
    "II": dict(P=0.15, Q=-0.05, R=1.10, S=-0.15, T=0.35),
    "V1": dict(P=0.06, Q=0.00, R=0.25, S=-0.90, T=0.05),
    "V2": dict(P=0.07, Q=0.00, R=0.50, S=-1.30, T=0.45),
    "V3": dict(P=0.07, Q=-0.02, R=0.80, S=-0.90, T=0.45),
    "V4": dict(P=0.07, Q=-0.05, R=1.30, S=-0.50, T=0.40),
    "V5": dict(P=0.07, Q=-0.08, R=1.20, S=-0.25, T=0.35),
    "V6": dict(P=0.07, Q=-0.08, R=0.90, S=-0.10, T=0.28),
}


def _wave(t, c, s_up, s_dn):
    s = np.where(t < c, s_up, s_dn)
    return np.exp(-((t - c) ** 2) / (2 * s ** 2))


def make_normal_ecg(seed=0, fs=500, duration=10.0, hr=None, noise=0.008, wander=0.03,
                    emg=0.0, powerline=0.0, t_scale=1.0, early_repol=0.0, hr_range=(55, 80)):
    """noise: white noise sd (mV); wander: baseline wander amplitude (mV);
    emg: band-limited 20-150 Hz muscle noise sd; powerline: 50 Hz amplitude;
    t_scale: multiplies T-wave amplitude (low T < 1); early_repol: benign J-point /
    ST elevation (mV) in V2-V4, as seen in many healthy young men."""
    rng = np.random.default_rng(seed)
    if hr is None:
        hr = rng.uniform(*hr_range)
    n = int(fs * duration)
    t = np.arange(n) / fs
    rr_mean = 60.0 / hr
    # R-peak times with mild sinus variability
    r_times, tr = [], rng.uniform(0.25, 0.6)
    while tr < duration + 1.0:
        r_times.append(tr)
        tr += rr_mean * (1 + 0.02 * rng.standard_normal())
    scale = {k: rng.uniform(0.8, 1.2) for k in _WAVES}
    scale["T"] *= t_scale
    er_w = {"V2": 1.0, "V3": 1.0, "V4": 0.6}
    base = {}
    for lead in ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]:
        x = np.zeros(n)
        for r in r_times:
            for w, (c, su, sd) in _WAVES.items():
                cc = c * np.sqrt(rr_mean) if w == "T" else c  # T position follows Bazett
                sdd = sd * np.sqrt(rr_mean) if w == "T" else sd
                suu = su * np.sqrt(rr_mean) if w == "T" else su
                x += _AMP[lead][w] * scale[w] * _wave(t, r + cc, suu, sdd)
            if early_repol and lead in er_w:
                # benign concave ST elevation from the J point, merging into the T wave
                j, tp = r + 0.045, r + _WAVES["T"][0] * np.sqrt(rr_mean)
                seg = (t >= j - 0.01) & (t <= tp + 0.06)
                u = np.clip((t[seg] - (j - 0.01)) / 0.02, 0, 1)
                dn = np.clip((tp + 0.06 - t[seg]) / 0.08, 0, 1)
                x[seg] += early_repol * er_w[lead] * np.minimum(u, dn)
        base[lead] = x
    I, II = base["I"], base["II"]
    limb = {"I": I, "II": II, "III": II - I, "aVR": -(I + II) / 2,
            "aVL": I - II / 2, "aVF": II - I / 2}
    sig = np.stack([limb[l] if l in limb else base[l] for l in LEADS], axis=1)
    sig += noise * rng.standard_normal(sig.shape)
    for j in range(12):
        f, ph = rng.uniform(0.15, 0.3), rng.uniform(0, 2 * np.pi)
        sig[:, j] += wander * np.sin(2 * np.pi * f * t + ph)
    if emg:
        from scipy.signal import butter, filtfilt
        b, a = butter(3, [20 / (fs / 2), min(150, fs / 2 - 1) / (fs / 2)], btype="band")
        e = filtfilt(b, a, rng.standard_normal(sig.shape), axis=0)
        sig += emg * e / e.std()
    if powerline:
        sig += powerline * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 6.28))[:, None]
    return sig.astype(np.float64), {"hr": hr, "fs": fs}
