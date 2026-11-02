"""Full re-analysis with coder v5 (sha256 in data/faith_parse_v5.sha256). Writes results/v5/*.csv.
Adds strict T inversion, atrial tachyarrhythmia (AF or flutter) and component-named omission."""
import json, re, sys
import numpy as np, pandas as pd
sys.path[:0] = ["coding", "analysis", "ecgcf"]
from faith_parse import decision, evidence_code, says_normal, final_answer, decision_t_strict, decision_atrial
from analyze import wilson
R = "results/"; OUT = R + "v5/"
K = ["st_elev", "t_inv", "qt_long", "qrs_wide", "af"]
NAMED = {"st_elev": r"\bst\b|st[- ]segment", "t_inv": r"\bt[- ]?waves?\b|\bt[- ]?wave|repolari", "qt_long": r"\bqtc?\b",
         "qrs_wide": r"\bqrs\b|bundle[- ]branch|conduction (delay|defect|disturbance|abnormal)", "af": r"p[- ]?waves?|irregular|fibrillat|f[- ]?waves?"}
SUPRA = {"st_elev": 3, "t_inv": 3, "qt_long": 3, "qrs_wide": 4, "af": 3}
rng = np.random.default_rng(2026)
man = pd.read_csv(R + "r1/manifest.csv").set_index("sample_id")
NUM = {}

def boot_ci(df, col, B=2000):
    """cluster bootstrap over base ECGs of a mean"""
    g = df.groupby("ecg_id")[col].agg(["sum", "size"]); s, n = g["sum"].values, g["size"].values
    if len(g) == 0: return (np.nan, np.nan)
    idx = rng.integers(0, len(g), (B, len(g)))
    est = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return tuple(np.percentile(est, [2.5, 97.5]))

def load(f, q=None):
    D = pd.read_json(R + f, lines=True)
    return D.query(q) if q else D

SETS = {  # name: (file, filter for originals+edits+controls, all levels?)
    "R1_signal": ("r1/responses.jsonl", "block=='A' and modality=='signal'", True),
    "R1_image": ("r1/responses.jsonl", "block=='C' and modality=='image'", False),
    "R1_both": ("r1/responses.jsonl", "block=='C' and modality=='both'", False),
    "R1_exposed": ("r1/responses.jsonl", "block=='D'", False),
    "PULSE_report": ("pulse/responses.jsonl", "block=='A'", False),
    "PULSE_evidence": ("pulse/responses.jsonl", "block=='C'", False),
    "GEM": ("gem/responses.jsonl", None, False),
}
pair_rows, ctrl_rows, orig_rows = [], [], []
for name, (f, q, full) in SETS.items():
    D = load(f, q); D = D[D.role.isin(["original", "edit", "control"])]
    O = D[D.role == "original"].drop_duplicates("ecg_id").set_index("ecg_id").answer
    f0 = {e: {k: decision(t, k) for k in K} for e, t in O.items()}
    f0s = {e: {k: decision(t, k, strict=True) for k in K} for e, t in O.items()}
    f0x = {e: dict(t_strict=decision_t_strict(t), atrial=decision_atrial(t)) for e, t in O.items()}
    for e, t in O.items():
        orig_rows.append(dict(set=name, ecg_id=e, any=any(f0[e].values()), normal=says_normal(t), **{f"o_{k}": f0[e][k] for k in K}))
    for _, r in D[D.role == "edit"].iterrows():
        if r.ecg_id not in O.index: continue
        d0 = f0[r.ecg_id][r.kind]; d1 = decision(r.answer, r.kind)
        ev = evidence_code(r.answer, r.kind)
        other = [k for k in K if k != r.kind and decision(r.answer, k, strict=True) and not f0s[r.ecg_id][k]]
        named = bool(re.search(NAMED[r.kind], r.answer.lower()))
        xs = dict(d0_strict=f0x[r.ecg_id]["t_strict"], I_strict=decision_t_strict(r.answer) and not f0x[r.ecg_id]["t_strict"]) if r.kind == "t_inv" else {}
        if r.kind == "af":
            xs = dict(d0_atrial=f0x[r.ecg_id]["atrial"], I_atrial=decision_atrial(r.answer) and not f0x[r.ecg_id]["atrial"],
                      flutter=bool(re.search(r"flutter", final_answer(r.answer).lower())))
        pair_rows.append(dict(set=name, ecg_id=r.ecg_id, kind=r.kind, level=r.level_idx, crit=man.criterion_met.get(r.sample_id, np.nan),
                              d0=d0, I=(d1 and not d0), other=bool(other), other_which=",".join(other), named=named, **xs, **ev))
    for _, r in D[D.role == "control"].iterrows():
        if r.ecg_id not in O.index: continue
        new = [k for k in K if decision(r.answer, k) and not f0[r.ecg_id][k]]
        ctrl_rows.append(dict(set=name, ecg_id=r.ecg_id, kind=r.kind, new=bool(new), which=",".join(new),
                              identical=r.answer.strip() == O[r.ecg_id].strip(), **{f"n_{k}": (k in new) for k in K}))
P = pd.DataFrame(pair_rows); C = pd.DataFrame(ctrl_rows); Og = pd.DataFrame(orig_rows)
P.to_csv(OUT + "pairs_all.csv", index=False); C.to_csv(OUT + "controls_all.csv", index=False); Og.to_csv(OUT + "originals_all.csv", index=False)

# ---------------- positives (block B + extra) ----------------
pos = []
for name, f, extra, grp in [("R1", "r1/responses.jsonl", "r1_extrapos/responses_extrapos.jsonl", "modality"),
                            ("PULSE", "pulse/responses.jsonl", "pulse_extrapos/responses.jsonl", "prompt"),
                            ("GEM", "gem/responses.jsonl", "gem_extrapos/responses_extrapos.jsonl", "modality")]:
    A = load(f); A = A[A.role == "positive"]; X = pd.concat([A, load(extra)]).drop_duplicates(["sample_id", grp])
    for _, r in X.iterrows():
        pos.append(dict(model=name, cond=r[grp], kind=r.kind, ecg_id=r.ecg_id, det=decision(r.answer, r.kind)))
POS = pd.DataFrame(pos); POS.to_csv(OUT + "positives_all.csv", index=False)
sens = POS.groupby(["model", "cond", "kind"]).det.agg(["sum", "size"])

# ---------------- main table ----------------
SENS_KEY = {"R1_signal": ("R1", "signal"), "R1_image": ("R1", "image"), "R1_both": ("R1", "both"), "R1_exposed": ("R1", "signal"),
            "PULSE_report": ("PULSE", "report"), "PULSE_evidence": ("PULSE", "evidence"), "GEM": ("GEM", "both")}
rows = []
for name in SETS:
    Ps = P[P.set == name]; Cs = C[C.set == name]
    for k in K:
        e = Ps[(Ps.kind == k) & ~Ps.d0]
        sup = e[e.level == SUPRA[k]]
        if len(sup) == 0: continue
        inf = sup[sup.I]; non = sup[~sup.I]
        lo, hi = boot_ci(sup.assign(x=sup.I.astype(int)), "x")
        ctrl_flip = Cs[f"n_{k}"].mean() if len(Cs) else np.nan
        tp = (sup.I & sup.evidence).sum(); fp = (~sup.I & sup.evidence).sum(); fn = (sup.I & ~sup.evidence).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
        ta = np.nan
        if SETS[name][2]:
            ek = e.dropna(subset=["crit"])
            a, b = ek[ek.crit == True], ek[ek.crit == False]
            if len(a) and len(b): ta = a.I.mean() - b.I.mean()
        mk, cond = SENS_KEY[name]
        s = sens.loc[(mk, cond, k)] if (mk, cond, k) in sens.index else None
        se = s["sum"] / s["size"] if s is not None else np.nan
        rows.append(dict(set=name, kind=k, n=len(sup), DIR=sup.I.mean(), DIR_lo=lo, DIR_hi=hi, n_inf=len(inf),
                         omission=(~inf.evidence).mean() if len(inf) else np.nan,
                         grounded=inf.grounded.mean() if len(inf) else np.nan,
                         misloc=inf.misloc.mean() if len(inf) and k in ("st_elev", "t_inv") else np.nan,
                         conf=non.evidence.mean() if len(non) else np.nan, n_non=len(non), F1=f1, TA=ta,
                         wrong=sup.other.mean(), ctrl_flip=ctrl_flip, netDIR=sup.I.mean() - ctrl_flip,
                         sens=se, n_pos=s["size"] if s is not None else 0, gap=se - sup.I.mean() if s is not None else np.nan,
                         excluded_d0=int(((Ps.kind == k) & Ps.d0 & (Ps.level == SUPRA[k])).sum()),
                         omission_named=(~inf.named).mean() if len(inf) else np.nan,
                         DIR_t_strict=sup[~sup.d0_strict.astype(bool)].I_strict.mean() if k == "t_inv" else np.nan,
                         DIR_atrial=sup[~sup.d0_atrial.astype(bool)].I_atrial.mean() if k == "af" else np.nan,
                         flutter_in_noninf=non.flutter.mean() if k == "af" and len(non) else np.nan))
T = pd.DataFrame(rows); T.to_csv(OUT + "main_table.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.round(3).to_string(index=False))

# originals & controls
print("\nORIGINALS (any target finding / says normal):")
print(Og.groupby("set").agg(n=("any", "size"), any=("any", "sum"), normal=("normal", "sum"), **{k: (f"o_{k}", "sum") for k in K}).to_string())
print("\nCONTROLS new finding / identical:")
print(C.groupby(["set", "kind"]).agg(n=("new", "size"), new=("new", "mean"), ident=("identical", "mean")).round(3).to_string())
print("\nPOSITIVES:"); print(sens.to_string())
# dose-response R1
dr = P[(P.set == "R1_signal") & ~P.d0].groupby(["kind", "level"]).I.agg(["mean", "size"]).round(3)
print("\nR1 dose-response:"); print(dr.unstack(0).to_string())
dr.to_csv(OUT + "r1_dose.csv")
