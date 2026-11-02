"""Figure: QTc stated in the answer vs QTc measured on the input (originals and long-QT edits)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = pd.read_csv("results/qtc_stated_vs_measured.csv").dropna(subset=["stated", "meas"])
C = {"ECG-R1": "#2a78d6", "GEM": "#eda100"}
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b5b0",
                     "xtick.color": "#555", "ytick.color": "#555"})
fig, ax = plt.subplots(figsize=(3.4, 3.0))
rng = np.random.default_rng(0)
ax.plot([340, 620], [340, 620], color="#8a8a85", lw=1, ls="--", zorder=0)
ax.text(575, 560, "stated = measured", rotation=45, ha="right", va="bottom", fontsize=6.5, color="#8a8a85")
ax.axhspan(450, 460, color="#ecebe6", zorder=0, lw=0)
ax.text(345, 462, "long-QT threshold (450/460 ms)", fontsize=6.3, color="#6a6a65")
for model, mk in [("ECG-R1", "o"), ("GEM", "^")]:
    d = D[D.model == model]
    ax.scatter(d.meas, d.stated + rng.uniform(-3, 3, len(d)), s=5, color=C[model], alpha=0.18, lw=0, zorder=1)
    g = d.groupby("level").agg(mx=("meas", "median"), my=("stated", "median"),
                               lo=("stated", lambda x: x.quantile(.25)), hi=("stated", lambda x: x.quantile(.75)), n=("stated", "size"))
    g = g[g.n >= 20]
    ax.errorbar(g.mx, g.my, yerr=[g.my - g.lo, g.hi - g.my], fmt=mk + "-", color=C[model], ms=6, lw=2, elinewidth=1,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3, label=f"{model} (median, IQR)")
ax.set_xlim(340, 620); ax.set_ylim(340, 620)
ax.set_xlabel("QTc measured on the input (ms)"); ax.set_ylabel("QTc stated in the answer (ms)")
ax.legend(frameon=False, loc="upper left", fontsize=6.8)
fig.tight_layout()
fig.savefig("figures/fig_qtc.pdf"); fig.savefig("figures/fig_qtc.png", dpi=300)
print("ok")
