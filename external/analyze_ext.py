"""Analysis of the review experiments (Georgia replication, reciprocal ST) for one model."""
import json, sys
import numpy as np, pandas as pd
sys.path[:0] = ["coding", "analysis", "ecgcf"]
from faith_parse import decision, evidence_code, says_normal, decision_atrial, decision_t_strict
from analyze import wilson
KINDS = ["st_elev", "t_inv", "qt_long", "qrs_wide", "af"]
DK = lambda k: "st_elev" if k == "st_recip" else k
def fmt(k, n):
    if n == 0: return "-"
    p, lo, hi = wilson(k, n); return f"{100*p:.0f} [{100*lo:.0f}-{100*hi:.0f}] ({k}/{n})"

def run(resp, man_g, main_resp, recip_man, tag, main_filter=None):
    R = pd.read_json(resp, lines=True)
    out = {}
    man = pd.read_csv(man_g).set_index("sample_id")
    G = R[R.block == "G"]
    O = G[G.role == "original"].set_index("ecg_id").answer
    f0 = {e: {k: decision(t, k) for k in KINDS} for e, t in O.items()}
    print(f"\n===== {tag} | Georgia, {len(O)} normal base ECGs =====")
    print("originals with no target finding:", fmt(sum(not any(v.values()) for v in f0.values()), len(O)))
    rows = []
    for _, r in G[G.role == "edit"].iterrows():
        k = DK(r.kind); d0 = f0[r.ecg_id][k]
        if d0: continue
        I = decision(r.answer, k)
        o = O[r.ecg_id]
        Ia = decision_atrial(r.answer) and not decision_atrial(o) if k == "af" else np.nan
        It = decision_t_strict(r.answer) and not decision_t_strict(o) if k == "t_inv" else np.nan
        rows.append(dict(kind=r.kind, level=r.level_idx, crit=man.criterion_met.get(r.sample_id), I=I, I_atrial=Ia, I_tstrict=It, **evidence_code(r.answer, k)))
    E = pd.DataFrame(rows)
    tab = E.groupby(["kind", "level"]).I.agg(["sum", "size"])
    print("DIR by edit and level:")
    for (k, l), x in tab.iterrows(): print(f"  {k:9s} L{l}: {fmt(int(x['sum']), int(x['size']))}")
    print("AF edits, atrial tachyarrhythmia DIR:", E[E.kind == "af"].assign(x=lambda d: d.I_atrial.astype(float)).groupby("level").x.mean().round(2).to_dict(),
          "| T edits, strict T DIR:", E[E.kind == "t_inv"].assign(x=lambda d: d.I_tstrict.astype(float)).groupby("level").x.mean().round(2).to_dict())
    inf = E[E.I]
    if len(inf):
        print("given influence: evidence / grounded / misloc"); print(inf.groupby("kind")[["evidence", "grounded", "misloc"]].mean().round(2).to_string())
    C = G[G.role == "control"]; cn = []
    for _, r in C.iterrows():
        cn.append(dict(kind=r.kind, new=any(decision(r.answer, k) and not f0[r.ecg_id][k] for k in KINDS)))
    cn = pd.DataFrame(cn)
    for k, g in cn.groupby("kind"): print(f"control {k}: new finding {fmt(int(g.new.sum()), len(g))}")
    P = R[R.block == "GP"]
    print("real Georgia recordings detected:")
    for k, g in P.groupby("kind"): print(f"  {k:9s}: {fmt(sum(decision(a, k) for a in g.answer), len(g))}" + (f" | atrial {sum(decision_atrial(a) for a in g.answer)}" if k == "af" else ""))
    out["E"] = E
    # reciprocal ST on PTB-XL vs plain ST
    M = pd.read_json(main_resp, lines=True)
    if main_filter is not None: M = main_filter(M)
    MO = M[M.role == "original"].drop_duplicates("ecg_id").set_index("ecg_id").answer
    Rr = R[R.block == "R"]; rr = []
    for _, r in Rr.iterrows():
        if r.ecg_id not in MO.index or decision(MO[r.ecg_id], "st_elev"): continue
        rr.append(dict(level=r.level_idx, I=decision(r.answer, "st_elev"), **evidence_code(r.answer, "st_elev")))
    rr = pd.DataFrame(rr)
    pl = []
    for _, r in M[M.kind == "st_elev"].iterrows():
        if r.ecg_id not in MO.index or decision(MO[r.ecg_id], "st_elev") or r.ecg_id not in set(Rr.ecg_id): continue
        pl.append(dict(level=r.level_idx, I=decision(r.answer, "st_elev")))
    pl = pd.DataFrame(pl)
    print(f"\n===== {tag} | PTB-XL: ST elevation alone vs with reciprocal depression =====")
    for l in sorted(rr.level.unique()):
        a = rr[rr.level == l]; b = pl[pl.level == l] if len(pl) else pl
        print(f"  L{l}: plain {fmt(int(b.I.sum()), len(b)) if len(b) else '-'} | reciprocal {fmt(int(a.I.sum()), len(a))}")
    out["rr"] = rr
    return out
if __name__ == "__main__":
    pass
