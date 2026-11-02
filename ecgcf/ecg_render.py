"""Render a 12-lead ECG as a standard paper printout (the layout PULSE was trained on).

3 x 4 layout (I/II/III | aVR/aVL/aVF | V1-V3 | V4-V6), 2.5 s per column, plus a 10-s
lead II rhythm strip; red millimetre grid; 25 mm/s, 10 mm/mV; 1-mV calibration pulses.
Only a constant offset is removed per displayed segment (no filtering), so ST levels and
wave amplitudes are preserved exactly as in the signal.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAYOUT = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MM_PER_S, MM_PER_MV = 25.0, 10.0
ROW_MM = 32.0          # vertical spacing between rows (3.2 mV)
CAL_MM = 5.0           # calibration pulse width (0.2 s)


def _offset(seg, fs, onset_idx=None):
    """baseline offset: median of the segment (robust, keeps relative ST levels)"""
    return float(np.median(seg))


def render(sig, fs, path, dpi=125, label=True):
    sig = np.asarray(sig, float)
    n_col_s = 2.5
    ncol = int(n_col_s * fs)
    width_mm = CAL_MM + 4 * n_col_s * MM_PER_S + 5
    height_mm = 4 * ROW_MM + 14
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.axis("off")
    # grid
    for x in np.arange(0, width_mm + 0.01, 1.0):
        ax.axvline(x, color="#f6c1c1" if x % 5 else "#e98888", lw=0.35 if x % 5 else 0.7, zorder=0)
    for y in np.arange(0, height_mm + 0.01, 1.0):
        ax.axhline(y, color="#f6c1c1" if y % 5 else "#e98888", lw=0.35 if y % 5 else 0.7, zorder=0)

    def cal(x0, y0):
        ax.plot([x0, x0 + 1, x0 + 1, x0 + 4, x0 + 4, x0 + CAL_MM],
                [y0, y0, y0 + MM_PER_MV, y0 + MM_PER_MV, y0, y0], color="k", lw=0.9, zorder=3)

    rows_y = [height_mm - 8 - ROW_MM * (r + 0.6) for r in range(4)]
    for r, row in enumerate(LAYOUT):
        y0 = rows_y[r]
        cal(0.5, y0)
        for c, lead in enumerate(row):
            seg = sig[c * ncol:(c + 1) * ncol, LEADS.index(lead)]
            x = CAL_MM + 1 + c * n_col_s * MM_PER_S + np.arange(len(seg)) / fs * MM_PER_S
            y = y0 + (seg - _offset(seg, fs)) * MM_PER_MV
            ax.plot(x, y, color="k", lw=0.8, zorder=3)
            if c > 0:
                ax.plot([x[0], x[0]], [y0 - 3, y0 + 3], color="k", lw=0.8, zorder=3)
            if label:
                ax.text(x[0] + 2, y0 + 12, lead, fontsize=9, zorder=4)
    # rhythm strip, lead II, 10 s
    y0 = rows_y[3]
    cal(0.5, y0)
    seg = sig[:4 * ncol, 1]
    x = CAL_MM + 1 + np.arange(len(seg)) / fs * MM_PER_S
    ax.plot(x, y0 + (seg - _offset(seg, fs)) * MM_PER_MV, color="k", lw=0.8, zorder=3)
    if label:
        ax.text(x[0] + 2, y0 + 12, "II", fontsize=9, zorder=4)
    ax.text(15, 3, "Speed: 25 mm/s      Limb: 10 mm/mV      Chest: 10 mm/mV", fontsize=8, zorder=4)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path
