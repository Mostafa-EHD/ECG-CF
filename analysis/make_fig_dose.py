"""Figure 3: DIR vs edit intensity (ECG-R1 signal, all levels) + supra-threshold DIR of other models/inputs."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/"
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return 100*p, 100*max(0, c-h), 100*min(1, c+h)
KINDS = [("st_elev", "ST elevation", ["+0.1", "+0.2", "+0.3"], "mV"), ("t_inv", "T inversion", ["×0.5", "×1", "×1.5"], ""),
         ("qt_long", "Long QT", ["+60", "+100", "+140"], "ms"), ("qrs_wide", "Wide QRS", ["+20", "+40", "+60", "+80"], "ms"),
         ("af", "Atrial fibrillation", ["1", "2", "3"], "evidence level")]
SUPRA = {"st_elev": 3, "t_inv": 3, "qt_long": 3, "qrs_wide": 4, "af": 3}
C = {"r1": "#2a78d6", "r1img": "#1baf7a", "r1both": "#e87ba4", "pulse": "#eb6834", "gem": "#eda100", "ink": "#2b2b2b", "muted": "#8a8a85"}
PV = pd.read_csv(R + "v5/pairs_all.csv")
r1 = PV[PV.set == "R1_signal"]
others = [("ECG-R1 (image)", "R1_image", C["r1img"], "s", -0.18),
          ("ECG-R1 (both)", "R1_both", C["r1both"], "D", -0.06),
          ("GEM (signal+image)", "GEM", C["gem"], "^", 0.06),
          ("PULSE (image)", "PULSE_report", C["pulse"], "o", 0.18)]
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b5b0",
                     "xtick.color": "#555", "ytick.color": "#555", "axes.labelcolor": C["ink"]})
fig, axs = plt.subplots(1, 5, figsize=(7.2, 2.35), sharey=True)
for ax, (k, title, labs, unit) in zip(axs, KINDS):
    e = r1[(r1.kind == k) & ~r1.d0]
    lv = sorted(e.level.unique()); x = np.arange(1, len(lv)+1)
    st = [wilson(e[e.level == l].I.sum(), (e.level == l).sum()) for l in lv]
    crit = [100*e[e.level == l].crit.mean() for l in lv]
    ax.bar(x, crit, width=0.55, color="#ecebe6", zorder=0, label="Edits meeting criterion (%)")
    ax.fill_between(x, [s[1] for s in st], [s[2] for s in st], color=C["r1"], alpha=0.15, lw=0)
    ax.plot(x, [s[0] for s in st], "-o", color=C["r1"], lw=2, ms=4.5, label="ECG-R1 (signal)", zorder=3)
    for name, f, col, mk, dx in others:
        d = PV[PV.set == f]; d = d[(d.kind == k) & ~d.d0 & (d.level == SUPRA[k])]
        if len(d) == 0: continue
        p, lo, hi = wilson(d.I.sum(), len(d)); xs = len(lv) + dx
        ax.errorbar(xs, p, yerr=[[max(p-lo,0)], [max(hi-p,0)]], fmt=mk, color=col, ms=4.5, elinewidth=1, capsize=0,
                    markeredgecolor="white", markeredgewidth=0.8, label=name, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_title(title, fontsize=8, color=C["ink"])
    ax.set_xlabel(unit, fontsize=7, color="#555"); ax.set_ylim(0, 103); ax.set_xlim(0.45, len(lv)+0.55)
    ax.yaxis.grid(True, color="#ecebe6", lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel("Decision influence rate (%)")
h, l = axs[0].get_legend_handles_labels(); h2, l2 = axs[4].get_legend_handles_labels()
seen = dict(zip(l, h)); seen.update({a: b for a, b in zip(l2, h2)})
order = ["ECG-R1 (signal)", "ECG-R1 (image)", "ECG-R1 (both)", "GEM (signal+image)", "PULSE (image)", "Edits meeting criterion (%)"]
fig.legend([seen[o] for o in order if o in seen], [o for o in order if o in seen], loc="lower center", ncol=6, frameon=False,
           fontsize=6.8, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
fig.savefig("figures/fig_dose.pdf"); fig.savefig("figures/fig_dose.png", dpi=300)
print("ok")
