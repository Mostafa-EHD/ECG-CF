"""Validate the perturbation toolkit on synthetic normal 12-lead ECGs.

python validate_synthetic.py --n 40 --out results
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from ecg_cf import CATALOGUE, delineate, findings, measure, perturb, verify
from synthetic import make_normal_ecg

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--fs", type=int, default=500)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for rec in range(a.n):
        sig, meta = make_normal_ecg(seed=rec, fs=a.fs)
        F = delineate(sig, a.fs)
        m0 = measure(sig, a.fs, F)
        if any(findings(m0).values()):
            print(f"record {rec}: baseline not normal, skipped", findings(m0))
            continue
        for kind, spec in CATALOGUE.items():
            for li, level in enumerate(spec["levels"]):
                sig1, info = perturb(sig, a.fs, F, kind, level, seed=1000 * rec + li)
                v, _ = verify(m0, sig1, a.fs, kind, level)
                v.update(record=rec, level_idx=li + 1, hr0=m0["hr_bpm"], **info)
                rows.append(v)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "synthetic_verification.csv"), index=False)
    summ = (df.groupby(["kind", "level_idx", "level"])
              .agg(n=("impl_ok", "size"), impl_ok=("impl_ok", "mean"),
                   criterion_met=("criterion_met", "mean"),
                   off_target=("off_target_new_findings", lambda s: (s.fillna("") != "").mean()))
              .reset_index())
    order = {k: i for i, k in enumerate(CATALOGUE)}
    summ = summ.sort_values(by=["kind", "level_idx"], key=lambda c: c.map(order) if c.name == "kind" else c)
    summ.to_csv(os.path.join(a.out, "synthetic_summary.csv"), index=False)
    with pd.option_context("display.width", 140):
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
