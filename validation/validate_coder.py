"""Coder validation (paper, Sect. 'Validation of the coder', Table tab:valid).
Recomputes the coder's codes on the 100 validation answers and compares them with the consensus
coding of two authors (validation/consensus_v3.csv). Run from the repository root:
    python validation/validate_coder.py"""
import sys, numpy as np, pandas as pd
sys.path[:0] = ["coding", "analysis"]
from sklearn.metrics import cohen_kappa_score as ck
from faith_parse import decision, evidence_code, decision_t_strict, final_answer, _sentences, _asserted
from denial import denies_st_anterior_v7, denies_qt_v7
R = "results/"
K = pd.read_csv("validation/items_v3.csv").set_index("code_id")
H = pd.read_csv("validation/consensus_v3.csv", index_col=0)
src = {"ECG-R1": ("r1/responses.jsonl", "block=='A' and modality=='signal'"), "PULSE": ("pulse/responses.jsonl", "block=='A'"), "GEM": ("gem/responses.jsonl", None)}
ans = {}
for m, (f, q) in src.items():
    D = pd.read_json(R + f, lines=True); D = D.query(q) if q else D
    ans[m] = D.drop_duplicates("sample_id").set_index("sample_id").answer
def coder(cid):
    r = K.loc[cid]; a = ans[r.model][r.sample_id]; tg = r.target; ev = evidence_code(a, tg)
    return dict(Q1=int(decision(a, tg)), Q2=int(ev["evidence"]),
                Q3=(int(ev["grounded"]) if ev["evidence"] else np.nan) if tg in ("st_elev", "t_inv") else np.nan,
                Q4=int(denies_st_anterior_v7(a) if tg == "st_elev" else denies_qt_v7(a)) if tg in ("st_elev", "qt_long") else np.nan,
                Q5=(int(decision_t_strict(a)) if tg == "t_inv" else int(any(_asserted(r"flutter", s) for s in _sentences(final_answer(a))))) if tg in ("t_inv", "af") else np.nan)
C = pd.DataFrame({c: coder(c) for c in K.index}).T
rng = np.random.default_rng(0); out = []
for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    m = H[q].notna() & C[q].notna(); a, c = H[q][m].astype(int), C[q][m].astype(int)
    bs = [ck(a.values[i], c.values[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(2000)) if len(set(a.values[i]) | set(c.values[i])) > 1]
    tp, fp = int(((a == 1) & (c == 1)).sum()), int(((a == 0) & (c == 1)).sum()); fn, tn = int(((a == 1) & (c == 0)).sum()), int(((a == 0) & (c == 0)).sum())
    out.append(dict(code=q, n=int(m.sum()), agreement=(a == c).mean(), kappa=ck(a, c) if len(set(a) | set(c)) > 1 else np.nan,
                    k_lo=np.percentile(bs, 2.5), k_hi=np.percentile(bs, 97.5), tp=tp, fp=fp, fn=fn, tn=tn))
print(pd.DataFrame(out).round(3).to_string(index=False))
