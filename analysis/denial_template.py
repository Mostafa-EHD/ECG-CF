"""Is the ST/QT denial a fixed template? Rate on originals vs edits, and verbatim identity."""
import re, sys
import pandas as pd
sys.path[:0] = ["coding", "analysis", "ecgcf"]
from faith_parse import decision, _sentences
from denial import denies_st_anterior, denies_qt, ANT, ST_TERMS
from faith_parse import _asserted

def st_denial_sentence(text):
    for s in _sentences(text):
        if not re.search(ANT, s) or not re.search(r"\bst\b|st[- ]segment", s): continue
        if re.search(r"isoelectric|st[- ]?segments?[^.]{0,40}(within normal|normal|unremarkable|no (significant )?(deviation|elevation|change))", s) or \
           (any(re.search(p, s) for p in ST_TERMS) and not any(_asserted(p, s) for p in ST_TERMS)):
            return s
    return None

def run(name, f, q=None):
    D = pd.read_json(f, lines=True); D = D.query(q) if q else D
    O = D[D.role == "original"].drop_duplicates("ecg_id").set_index("ecg_id").answer
    E = D[(D.kind == "st_elev") & (D.level_idx == 3)]
    E = E[[e in O.index and not decision(O[e], "st_elev") for e in E.ecg_id]]
    non = E[[not decision(a, "st_elev") for a in E.answer]]
    o_rate = sum(denies_st_anterior(t) for t in O); e_rate = sum(denies_st_anterior(t) for t in non.answer)
    ident = sum(1 for _, r in non.iterrows() if st_denial_sentence(r.answer) and st_denial_sentence(r.answer) == st_denial_sentence(O[r.ecg_id]))
    whole = sum(1 for _, r in non.iterrows() if r.answer.strip() == O[r.ecg_id].strip())
    return dict(model=name, originals_deny=o_rate, n_orig=len(O), noninf_deny=e_rate, n_noninf=len(non), same_sentence=ident, same_answer=whole)
rows = [run("ECG-R1", "r1/responses.jsonl", "block=='A' and modality=='signal'"), run("GEM", "gem/responses.jsonl"), run("PULSE", "pulse/responses.jsonl", "block=='A'")]
T = pd.DataFrame(rows); T.to_csv("v5/denial_template.csv", index=False); print(T.to_string(index=False))
