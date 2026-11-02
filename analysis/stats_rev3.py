"""Confirmatory statistics (coder v5): H1-H4, modality, reciprocal ST; Holm correction."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest, spearmanr
sys.path[:0] = ["coding", "analysis", "ecgcf"]
from faith_parse import decision
R = "results/"
P = pd.read_csv(R + "v5/pairs_all.csv"); T = pd.read_csv(R + "v5/main_table.csv"); C = pd.read_csv(R + "v5/controls_all.csv")
SUPRA = {"st_elev": 3, "t_inv": 3, "qt_long": 3, "qrs_wide": 4, "af": 3}
rng = np.random.default_rng(7); tests = []  # rev3: B=50,000 (bootstrap floor 2e-5)

def cboot(d, col, ref, B=50000):
    """cluster bootstrap over base ECGs: 95% CI and one-sided p for mean > ref"""
    g = d.groupby("ecg_id")[col].agg(["sum", "size"]).values
    idx = rng.integers(0, len(g), (B, len(g))); S_ = g[idx].sum(1); est = S_[:, 0] / S_[:, 1]
    return np.percentile(est, 2.5), np.percentile(est, 97.5), max(np.mean(est <= ref), 1 / B)


def mcnemar(a, b):  # paired booleans
    n01 = int(((~a) & b).sum()); n10 = int((a & ~b).sum())
    return n01, n10, (binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0)

# H1: amplitude (ST, T) vs duration (QT, QRS), supra-threshold, cluster bootstrap on base ECGs
for s in ["R1_signal", "PULSE_report", "GEM"]:
    d = P[(P.set == s) & ~P.d0]; d = d[d.level == d.kind.map(SUPRA)]
    d = d.assign(amp=d.kind.isin(["st_elev", "t_inv", "af"]), dur=d.kind.isin(["qt_long", "qrs_wide"]))
    agg = d.groupby("ecg_id").apply(lambda x: pd.Series(dict(ai=x[x.amp].I.sum(), an=x.amp.sum(), di=x[x.dur].I.sum(), dn=x.dur.sum())))
    V = agg.values; diff = V[:, 0].sum() / V[:, 1].sum() - V[:, 2].sum() / V[:, 3].sum()
    idx = rng.integers(0, len(V), (50000, len(V))); S_ = V[idx].sum(1)
    bs = S_[:, 0] / S_[:, 1] - S_[:, 2] / S_[:, 3]
    p = 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))
    tests.append(dict(h="H1", test=f"{s}: DIR amplitude/rhythm (ST,T,AF) - duration (QT,QRS)", est=diff, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p=max(p, 1/50000)))
# H2: omission given influence > 10% (one-sided binomial), per model pooled over edits
for s in ["R1_signal", "PULSE_report", "GEM"]:
    d = P[(P.set == s) & ~P.d0 & P.I]; d = d[d.level == d.kind.map(SUPRA)]
    k = int((~d.evidence).sum()); n = len(d)
    lo, hi, p = cboot(d.assign(x=(~d.evidence).astype(int)), "x", 0.10)
    tests.append(dict(h="H2", test=f"{s}: omission given influence > 10% (cluster bootstrap)", est=k / n, lo=lo, hi=hi, p=p, k=k, n=n))
# H3: confabulation on controls > 5% (one-sided), pooled over the 3 controls
for s in ["R1_signal", "PULSE_report", "GEM"]:
    d = C[C.set == s]; k = int(d.new.sum()); n = len(d)
    lo, hi, p = cboot(d.assign(x=d.new.astype(int)), "x", 0.05)
    tests.append(dict(h="H3", test=f"{s}: new finding on controls > 5% (cluster bootstrap)", est=k / n, lo=lo, hi=hi, p=p, k=k, n=n))
# H4: faithfulness follows accuracy: Spearman between sensitivity on real recordings and F1 across model x edit cells
t = T[T.set.isin(["R1_signal", "PULSE_report", "GEM"]) & (T.kind != "st_elev")].dropna(subset=["sens", "F1"])
rho, p = spearmanr(t.sens, t.F1)
tests.append(dict(h="H4", test=f"Spearman(sensitivity, F1_faith) over {len(t)} model x edit cells", est=rho, lo=np.nan, hi=np.nan, p=p))
# modality (within ECG-R1, same 100 bases): signal vs image, signal vs both
A = pd.read_json(R + "r1/responses.jsonl", lines=True)
for k in SUPRA:
    lv = SUPRA[k]; sub = A[(A.kind == k) & (A.level_idx == lv)]
    Ob = A[A.role == "original"]
    ids = set(sub[sub.modality == "image"].ecg_id) & set(sub[(sub.block == "A")].ecg_id)
    def infl(mod, blk):
        o = Ob[(Ob.modality == mod) & (Ob.block == blk)].drop_duplicates("ecg_id").set_index("ecg_id").answer
        e = sub[(sub.modality == mod) & (sub.block == blk)].drop_duplicates("ecg_id").set_index("ecg_id").answer
        return {i: decision(e[i], k) and not decision(o[i], k) for i in ids if i in e.index and i in o.index}
    sg, im, bo = infl("signal", "A"), infl("image", "C"), infl("both", "C")
    for nm, other in [("image", im), ("both", bo)]:
        c = sorted(set(sg) & set(other))
        a = np.array([sg[i] for i in c]); b = np.array([other[i] for i in c])
        n01, n10, p = mcnemar(a, b)
        tests.append(dict(h="H4-modality", test=f"ECG-R1 {k}: signal {a.mean():.2f} vs {nm} {b.mean():.2f} (n={len(c)})", est=b.mean() - a.mean(), lo=np.nan, hi=np.nan, p=p, n01=n01, n10=n10))
# reciprocal ST (PTB-XL, L3) within model
for name, main_f, ext_f, q in [("ECG-R1", "r1/responses.jsonl", "ext_r1/responses.jsonl", "block=='A' and modality=='signal'"), ("GEM", "gem/responses.jsonl", "ext_gem/responses.jsonl", None), ("PULSE", "pulse/responses.jsonl", "ext_pulse/responses.jsonl", "block=='A'")]:
    M = pd.read_json(R + main_f, lines=True); M = M.query(q) if q else M
    X = pd.read_json(R + ext_f, lines=True); X = X[(X.block == "R") & (X.level_idx == 3)].set_index("ecg_id").answer
    O = M[M.role == "original"].drop_duplicates("ecg_id").set_index("ecg_id").answer
    pl = M[(M.kind == "st_elev") & (M.level_idx == 3)].drop_duplicates("ecg_id").set_index("ecg_id").answer
    c = [i for i in X.index if i in pl.index and i in O.index and not decision(O[i], "st_elev")]
    a = np.array([decision(pl[i], "st_elev") for i in c]); b = np.array([decision(X[i], "st_elev") for i in c])
    n01, n10, p = mcnemar(a, b)
    tests.append(dict(h="realism", test=f"{name}: ST plain {a.mean():.2f} vs reciprocal {b.mean():.2f} (n={len(c)})", est=b.mean() - a.mean(), lo=np.nan, hi=np.nan, p=p, n01=n01, n10=n10))
S = pd.DataFrame(tests)
# Holm over all tests
order = np.argsort(S.p.values); m = len(S); adj = np.empty(m); run = 0
for r, i in enumerate(order):
    run = max(run, min(1, (m - r) * S.p.values[i])); adj[i] = run
S["p_holm"] = adj
S.to_csv(R + "rev3/stats.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
print(S[["h", "test", "est", "lo", "hi", "p", "p_holm"]].round(4).to_string(index=False))
