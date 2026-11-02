"""Main analysis of ECG-CF model answers (one model at a time).

Usage: python analyze.py responses.jsonl manifest.csv --model PULSE [--group prompt|modality]
Outputs tables (CSV + printed) for: recognition on real recordings, decision influence,
faithfulness of the explanation, controls, pattern gap. CIs: Wilson for proportions,
cluster bootstrap (by base ECG) for differences.
"""
import argparse
import json

import numpy as np
import pandas as pd

from faith_parse import DECISION, decision, evidence_code, says_normal

KINDS = ["st_elev", "t_inv", "qt_long", "qrs_wide", "af"]
LABEL = {"st_elev": "ST elevation", "t_inv": "T inversion", "qt_long": "Long QT", "qrs_wide": "Wide QRS",
         "af": "Atrial fibrillation"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100*p:.0f} [{100*lo:.0f}-{100*hi:.0f}] ({k}/{n})" if n else "-"


def any_finding(text):
    return {k: decision(text, k) for k in KINDS}


def analyse(R, man, group_col, tag):
    out = {}
    crit = man.set_index("sample_id").criterion_met.to_dict()
    for g, Rg in R.groupby(group_col):
        A = Rg[Rg.role.isin(["original", "edit", "control"])]
        O = A[A.role == "original"].set_index("ecg_id").answer
        # 1. originals
        f0 = {e: any_finding(t) for e, t in O.items()}
        n_norm = sum(not any(v.values()) for v in f0.values())
        rows = [dict(item="Original normal ECG: no target finding (specificity)", value=fmt(n_norm, len(O)))]
        n_say = sum(says_normal(t) for t in O)
        rows.append(dict(item="Original normal ECG: explicitly called normal", value=fmt(n_say, len(O))))
        # 2. edits
        E = []
        for _, r in A[A.role == "edit"].iterrows():
            if r.ecg_id not in O.index:
                continue
            d1 = decision(r.answer, r.kind)
            d0 = f0[r.ecg_id][r.kind]
            new_other = [k for k in KINDS if k != r.kind and decision(r.answer, k, strict=True)
                         and not decision(O[r.ecg_id], k, strict=True)]
            ev = evidence_code(r.answer, r.kind)
            E.append(dict(ecg_id=r.ecg_id, kind=r.kind, level=r.level_idx, crit=crit.get(r.sample_id, np.nan),
                          d0=d0, I=d1 and not d0, other=bool(new_other), unchanged=r.answer.strip() == O[r.ecg_id].strip(),
                          **ev))
        E = pd.DataFrame(E)
        # 3. controls
        Cn = []
        for _, r in A[A.role == "control"].iterrows():
            if r.ecg_id not in O.index:
                continue
            new = [k for k in KINDS if decision(r.answer, k) and not f0[r.ecg_id][k]]
            Cn.append(dict(ecg_id=r.ecg_id, kind=r.kind, new=bool(new), which=",".join(new),
                           identical=r.answer.strip() == O[r.ecg_id].strip()))
        Cn = pd.DataFrame(Cn)
        # 4. positives
        P = Rg[Rg.role == "positive"]
        tab = []
        for k in KINDS:
            e = E[E.kind == k] if len(E) else E
            p = P[P.kind == k]
            sens = sum(decision(a, k) for a in p.answer)
            if len(e) == 0:
                continue
            e0 = e[~e.d0]
            infl = e0[e0.I]
            dir_p = e0.I.mean() if len(e0) else np.nan
            sens_p = sens / len(p) if len(p) else np.nan
            tab.append({
                "Edit": LABEL[k],
                "n pairs": len(e0),
                "DIR %": fmt(int(e0.I.sum()), len(e0)),
                "Evidence | I %": fmt(int(infl.evidence.sum()), len(infl)),
                "Omission %": fmt(int((~infl.evidence).sum()), len(infl)),
                "Grounded | I %": fmt(int(infl.grounded.sum()), len(infl)),
                "Misloc | I %": fmt(int(infl.misloc.sum()), len(infl)) if k in ("st_elev", "t_inv") else "-",
                "Wrong finding %": fmt(int(e0.other.sum()), len(e0)),
                "Evidence w/o I %": fmt(int(e0[~e0.I].evidence.sum()), int((~e0.I).sum())),
                "Real recordings detected %": fmt(sens, len(p)),
                "Pattern gap (pts)": f"{100*(sens_p - dir_p):.0f}" if len(p) else "-",
            })
        tab = pd.DataFrame(tab)
        ctrl = [dict(item=f"Control {c}: new target finding", value=fmt(int(g_.new.sum()), len(g_)))
                for c, g_ in Cn.groupby("kind")] if len(Cn) else []
        ctrl += [dict(item=f"Control {c}: answer identical to original", value=fmt(int(g_.identical.sum()), len(g_)))
                 for c, g_ in Cn.groupby("kind")] if len(Cn) else []
        print(f"\n=================== {tag} | {group_col} = {g} ===================")
        print(pd.DataFrame(rows + ctrl).to_string(index=False))
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(tab.to_string(index=False))
        if len(Cn) and Cn.new.any():
            print("new findings on controls:", Cn[Cn.new].which.str.split(",").explode().value_counts().to_dict())
        out[g] = dict(E=E, C=Cn, tab=tab, O=O)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("responses"); ap.add_argument("manifest")
    ap.add_argument("--model", default="model"); ap.add_argument("--group", default="prompt")
    ap.add_argument("--block", default=None)
    a = ap.parse_args()
    R = pd.DataFrame([json.loads(l) for l in open(a.responses)])
    if a.block:
        R = R[R.block.isin(a.block.split(",")) | (R.role == "positive")]
    man = pd.read_csv(a.manifest)
    res = analyse(R, man, a.group, a.model)
    for g, r in res.items():
        r["tab"].to_csv(f"table_{a.model}_{g}.csv", index=False)
        r["E"].to_csv(f"pairs_{a.model}_{g}.csv", index=False)
