"""Figure 2: verified edits on a real PTB-XL recording (ecg_id 1287, unexposed, all edits kept)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = ["coding", "analysis", "ecgcf"]
import ecg_cf as E
z = np.load("results/fig2/1287.npz"); sig = z["original"].astype(float); fs = 500
F = E.delineate(sig, fs)
recip, _ = E.perturb(sig, fs, F, "st_recip", 0.30, seed=1287 * 10 + 3)
INK, INK2, GRID, ORIG, EDIT = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95", "#2a78d6"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
P = [("1287_st_elev_L3", None, "V2", 2.2, "ST elevation +0.3 mV (V1–V4); lead V2"),
     (None, recip, "aVF", 2.2, "ST elevation + reciprocal depression; lead aVF"),
     ("1287_t_inv_L3", None, "V3", 2.2, "T inversion ×1.5 (V2–V5); lead V3"),
     ("1287_qt_long_L3", None, "II", 2.2, "QT prolongation +140 ms; lead II"),
     ("1287_qrs_wide_L4", None, "V5", 0.8, "QRS widening +80 ms; lead V5 (one beat)"),
     ("1287_af_L2", None, "V1", 5.0, "Atrial fibrillation, evidence level 2; lead V1")]
fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0))
r0 = int(F["R"][2]) - int(0.35 * fs)
for ax, (key, arr, lead, win, title) in zip(axes.ravel(), P):
    y = z[key].astype(float) if key else arr
    sl = slice(r0, r0 + int(win * fs)); t = np.arange(sl.stop - sl.start) / fs; j = E.L[lead]
    ax.plot(t, sig[sl, j], color=ORIG, lw=1.3, label="original")
    ax.plot(t, y[sl, j], color=EDIT, lw=1.3, label="edited (verified)")
    ax.set_title(title, fontsize=8, loc="left", color=INK)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("time (s)"); ax.set_ylabel("mV")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.99, 1.0), fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("figures/fig_edits_real.pdf"); fig.savefig("figures/fig_edits_real.png", dpi=250)
print("ok")
