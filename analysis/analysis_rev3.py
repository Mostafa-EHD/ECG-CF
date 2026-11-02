"""Revision 3 analyses (after the third audit, 14 Oct 2026). Exploratory, added after the main runs:
 (1) ST reading of influenced answers: ischaemic/injury vs benign (early repolarisation) vs unspecified;
 (2) contradiction rate: the explanation states that the edited component is normal although a
     verified supra-threshold edit is present (ST: anterior denial v7; QT: QT/QTc normal);
 (3) template identity of the ECG-R1 anterior ST denial sentence: controls vs non-influenced ST edits.
Writes results/rev3/*.csv and prints a report."""
import re, sys
import numpy as np, pandas as pd
sys.path[:0] = ["coding", "analysis", "ecgcf"]
from faith_parse import decision, final_answer, _sentences, _asserted
from denial import denies_st_anterior, denies_st_anterior_v7 as denies_st_anterior_v6, denies_qt_v7 as denies_qt
from analyze import wilson
exec(open("analysis/denial_template.py").read().split("def run")[0])   # st_denial_sentence (v5)
from denial import denies_st_anterior_v7 as denies_st_anterior_v6, denies_qt_v7 as denies_qt   # re-bind after the exec above (rev4: v7 rules)
R = "results/"; OUT = R + "rev4/"
rng = np.random.default_rng(2026)
man = pd.read_csv(R + "r1/manifest.csv").set_index("sample_id")
ISCH = [r"isch\w*emi", r"\binjur", r"infarct", r"\bstemi\b", r"acute coronary", r"pericarditis"]
BENIGN = [r"early repolari[sz]ation", r"\bbenign\b", r"normal variant"]

def st_reading(text):
    ss = _sentences(final_answer(text))
    if any(_asserted(p, s) for s in ss for p in ISCH): return "ischaemic"
    if any(_asserted(p, s) for s in ss for p in BENIGN): return "benign"
    return "unspecified"

def ci(df, col, B=10000):
    g = df.groupby("ecg_id")[col].agg(["sum", "size"]); s, n = g["sum"].values, g["size"].values
    if len(g) == 0: return (np.nan, np.nan)
    idx = rng.integers(0, len(g), (B, len(g)))
    return tuple(np.percentile(s[idx].sum(1) / n[idx].sum(1), [2.5, 97.5]))

def fmt(x, df, col):
    lo, hi = ci(df, col); return f"{100*x:.0f}% [{100*lo:.0f}-{100*hi:.0f}]"

SETS = {"ECG-R1": ("r1/responses.jsonl", lambda D: D[(D.block == "A") & (D.modality == "signal")]),
        "GEM": ("gem/responses.jsonl", lambda D: D), "PULSE": ("pulse/responses.jsonl", lambda D: D[D.block == "A"])}
EXT = {"ECG-R1": "ext_r1", "GEM": "ext_gem", "PULSE": "ext_pulse"}
rows_st, rows_con, rep = [], [], []
for m, (f, flt) in SETS.items():
    D = flt(pd.read_json(R + f, lines=True))
    O = D[D.role == "original"].drop_duplicates("ecg_id").set_index("ecg_id").answer
    X = pd.read_json(R + EXT[m] + "/responses.jsonl", lines=True)
    G = X[(X.dataset == "georgia")]; GO = G[G.role == "original"].set_index("ecg_id").answer
    arms = [("PTB-XL", D[(D.role == "edit") & (D.kind == "st_elev") & (D.level_idx == 3)], O),
            ("PTB-XL reciprocal", X[(X.dataset == "ptbxl") & (X.kind == "st_recip") & (X.level_idx == 3)], O),
            ("Georgia", G[(G.role == "edit") & (G.kind == "st_elev") & (G.level_idx == 3)], GO),
            ("Georgia reciprocal", G[(G.role == "edit") & (G.kind == "st_recip") & (G.level_idx == 3)], GO)]
    for arm, E, OO in arms:
        for _, r in E.iterrows():
            if r.ecg_id not in OO.index or decision(OO[r.ecg_id], "st_elev"): continue
            I = decision(r.answer, "st_elev"); rd = st_reading(r.answer) if I else "none"
            rows_st.append(dict(model=m, arm=arm, ecg_id=r.ecg_id, I=I, isch=rd == "ischaemic", benign=rd == "benign", unspec=rd == "unspecified",
                                deny5=denies_st_anterior(r.answer), deny6=denies_st_anterior_v6(r.answer)))
    # contradiction on verified supra-threshold ST and QT edits (PTB-XL main set) + originals as reference
    for k, den in [("st_elev", denies_st_anterior_v6), ("qt_long", denies_qt)]:
        E = D[(D.role == "edit") & (D.kind == k) & (D.level_idx == 3)]
        for _, r in E.iterrows():
            if r.ecg_id not in O.index or decision(O[r.ecg_id], k) or not bool(man.criterion_met.get(r.sample_id, False)): continue
            rows_con.append(dict(model=m, kind=k, ecg_id=r.ecg_id, I=decision(r.answer, k), deny=den(r.answer), deny_v5=denies_st_anterior(r.answer) if k == "st_elev" else den(r.answer)))
        for e, t in O.items():
            rows_con.append(dict(model=m, kind=k + "_orig", ecg_id=e, I=False, deny=den(t), deny_v5=denies_st_anterior(t) if k == "st_elev" else den(t)))
ST = pd.DataFrame(rows_st); CON = pd.DataFrame(rows_con)
ST.to_csv(OUT + "st_reading_pairs.csv", index=False); CON.to_csv(OUT + "contradiction_pairs.csv", index=False)

print("=== (1) ST supra-threshold edits (level 3, finding absent on original): DIR split by reading ===")
tab = []
for (m, arm), g in ST.groupby(["model", "arm"], sort=False):
    n = len(g); d = dict(model=m, arm=arm, n=n)
    for c in ["I", "isch", "benign", "unspec"]:
        d[c] = g[c].mean(); d[c + "_lo"], d[c + "_hi"] = ci(g.assign(x=g[c].astype(int)), "x")
    tab.append(d)
    print(f"{m:7s} {arm:19s} n={n:3d} DIR {fmt(g.I.mean(), g.assign(x=g.I.astype(int)), 'x')} | ischaemic {fmt(g.isch.mean(), g.assign(x=g.isch.astype(int)), 'x')}"
          f" | benign {fmt(g.benign.mean(), g.assign(x=g.benign.astype(int)), 'x')} | unspecified {100*g.unspec.mean():.0f}%")
pd.DataFrame(tab).to_csv(OUT + "st_reading.csv", index=False)

print("\n=== (2) contradiction: explanation states the edited component is normal (verified supra-threshold edit, criterion met) ===")
tab = []
for (m, k), g in CON.groupby(["model", "kind"], sort=False):
    g = g.assign(x=g.deny.astype(int)); inf = g[g.I]; non = g[~g.I]
    d = dict(model=m, kind=k, n=len(g), contra=g.deny.mean(), contra_v5=g.deny_v5.mean(), n_inf=len(inf), contra_inf=inf.deny.mean() if len(inf) else np.nan,
             n_non=len(non), contra_non=non.deny.mean() if len(non) else np.nan)
    d["lo"], d["hi"] = ci(g, "x"); tab.append(d)
    print(f"{m:7s} {k:13s} n={len(g):3d} contradiction {fmt(g.deny.mean(), g, 'x')} (v5 rule {100*g.deny_v5.mean():.0f}%)"
          + (f" | among influenced {int(inf.deny.sum())}/{len(inf)} | among non-influenced {int(non.deny.sum())}/{len(non)}" if not k.endswith("orig") else ""))
pd.DataFrame(tab).to_csv(OUT + "contradiction.csv", index=False)

print("\n=== (3) ECG-R1 anterior ST denial sentence identical to the original's ===")
r1 = pd.read_json(R + "r1/responses.jsonl", lines=True); D = r1[(r1.block == "A") & (r1.modality == "signal")]
O = D[D.role == "original"].drop_duplicates("ecg_id").set_index("ecg_id").answer
def same(df):
    k = [bool(st_denial_sentence(r.answer)) and st_denial_sentence(r.answer) == st_denial_sentence(O[r.ecg_id]) for _, r in df.iterrows() if r.ecg_id in O.index]
    return sum(k), len(k)
C = D[D.role == "control"]; s, n = same(C); print(f"controls: {s}/{n} ({100*s/n:.0f}%)")
E = D[(D.kind == "st_elev") & (D.level_idx == 3) & (D.role == "edit")]
E = E[[e in O.index and not decision(O[e], "st_elev") and not decision(a, "st_elev") for e, a in zip(E.ecg_id, E.answer)]]
s2, n2 = same(E); print(f"non-influenced supra-threshold ST edits: {s2}/{n2} ({100*s2/n2:.0f}%)")
glob = sum(1 for a in E.answer if denies_st_anterior_v6(a) or re.search(r"st[- ]segments? (are|is) isoelectric|no (significant )?st[- ]segment (elevation|deviation)", a.lower()))
print(f"non-influenced: anterior denial v7 {sum(map(denies_st_anterior_v6, E.answer))}/{n2}; anterior or global ST denial {glob}/{n2}")
