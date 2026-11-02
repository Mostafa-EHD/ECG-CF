"""Build the counterfactual ECG set from PTB-XL.

1. selects clean, human-validated normal sinus-rhythm ECGs (default: test fold 10),
2. applies the 8 edits x 3 levels,
3. re-measures every edited ECG independently and keeps only verified edits,
4. writes the signals (.npz, one per record) and a manifest (.csv) for model querying.

Usage (PTB-XL v1.0.3 from https://physionet.org/content/ptb-xl/1.0.3/):
    python run_ptbxl.py --ptbxl /path/to/ptb-xl-1.0.3 --n 500 --out cf_ptbxl
"""
import argparse
import ast
import os
import time
import warnings

import numpy as np
import pandas as pd
import wfdb

from ecg_cf import CATALOGUE, LEADS, TARGET_FINDING, delineate, findings, measure, perturb, verify

warnings.filterwarnings("ignore")
QUALITY_COLS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems",
                "extra_beats", "pacemaker"]


def select_records(db, folds, seed, n):
    codes = db.scp_codes.apply(ast.literal_eval)
    normal = codes.apply(lambda c: c.get("NORM", 0) == 100 and set(c) <= {"NORM", "SR"})
    clean = db[[c for c in QUALITY_COLS if c in db]].isna().all(1)
    ok = normal & clean & (db.validated_by_human == True) & db.strat_fold.isin(folds)  # noqa: E712
    sel = db[ok]
    print(f"{ok.sum()} clean validated NORM records in folds {folds}")
    return sel.sample(frac=1.0, random_state=seed)   # shuffled; we stop once n pass the checks


def infer_sex_coding(db):
    """PTB-XL documents 52% male / 48% female but not the 0/1 coding explicitly.
    Infer it from the data instead of assuming, and print it for the user to confirm."""
    share = db.sex.value_counts(normalize=True)
    male_code = (share - 0.52).abs().idxmin()
    print(f"sex coding inferred from data: {share.round(3).to_dict()} -> male = {male_code}"
          f" (PTB-XL paper: 52% male). Check this once.")
    return lambda v: "M" if v == male_code else ("F" if v == v else None)


def clean_age(a):
    """PTB-XL sets ages >= 90 to 300 (HIPAA)."""
    if a != a:
        return None
    return 90 if a >= 300 else int(a)


def baseline_reason(m0, sex, age, hr=(50, 90)):
    """why a candidate normal ECG is not used as a base recording ("" = included)"""
    if m0["n_beats"] < 6:
        return "too few beats"
    if not (hr[0] <= m0["hr_bpm"] <= hr[1]):
        return f"HR {m0['hr_bpm']:.0f}"
    if not (m0["p_present_frac"] >= 0.8):
        return "P waves not detected"
    f0 = findings(m0, sex, age)
    if any(f0.values()):
        return "baseline measurement not normal: " + ",".join(k for k, v in f0.items() if v)
    return ""


def load(ptbxl, fname):
    sig, meta = wfdb.rdsamp(os.path.join(ptbxl, fname))
    names = [s.upper() for s in meta["sig_name"]]
    order = [names.index(l.upper()) for l in LEADS]
    return sig[:, order].astype(np.float64), int(meta["fs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", required=True, help="folder containing ptbxl_database.csv")
    ap.add_argument("--n", type=int, default=500, help="number of base ECGs to keep")
    ap.add_argument("--folds", type=int, nargs="+", default=[10],
                    help="PTB-XL strat folds to draw from (10 = recommended test fold)")
    ap.add_argument("--hr", type=float, nargs=2, default=[50, 90],
                    help="heart-rate range (bpm); leaves room for QT/QRS stretching")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_ids", nargs="*", default=[],
                    help="JSON files with PTB-XL ecg_ids to exclude (e.g. records found in the "
                         "instruction-tuning data of the evaluated models)")
    ap.add_argument("--out", default="cf_ptbxl")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "signals"), exist_ok=True)
    db = pd.read_csv(os.path.join(a.ptbxl, "ptbxl_database.csv"), index_col="ecg_id")
    cand = select_records(db, a.folds, a.seed, a.n)
    excl = set()
    for f in a.exclude_ids:
        excl |= {int(i) for i in __import__("json").load(open(f))}
    if excl:
        before = len(cand)
        cand = cand[~cand.index.isin(excl)]
        print(f"excluded {before - len(cand)} candidates found in training data ({len(cand)} left)")
    to_sex = infer_sex_coding(db)

    base_rows, rows, kept, t0 = [], [], 0, time.time()
    for ecg_id, r in cand.iterrows():
        if kept >= a.n:
            break
        sex, age = to_sex(r.sex), clean_age(r.age)
        try:
            sig, fs = load(a.ptbxl, r.filename_hr)
            F = delineate(sig, fs)
            m0 = measure(sig, fs, F)
        except Exception as e:                       # corrupted / unreadable record
            print(f"  {ecg_id}: skipped ({e})")
            continue
        reason = baseline_reason(m0, sex, age, a.hr)
        base_rows.append(dict(ecg_id=ecg_id, age=age, sex=sex, fold=r.strat_fold,
                              excluded=reason or "", **m0))
        if reason:
            continue
        arrays = {"original": sig.astype(np.float32)}
        for kind, spec in CATALOGUE.items():
            for li, level in enumerate(spec["levels"], start=1):
                y, info = perturb(sig, fs, F, kind, level, seed=int(ecg_id) * 10 + li)
                v, _ = verify(m0, y, fs, kind, level, sex=sex, age=age,
                              achieved=info.get("achieved_ms_mean"))
                sid = f"{ecg_id}_{kind}_L{li}"
                arrays[sid] = y.astype(np.float32)
                rows.append(dict(
                    sample_id=sid, ecg_id=ecg_id, sex=sex, age=age, kind=kind, level=level, level_idx=li,
                    unit=spec["unit"], is_control=kind.startswith("ctrl"),
                    expected_change=spec["dx"], target_wave=spec["wave"] or "",
                    target_leads="-".join(spec["leads"]) if spec["leads"] else "global",
                    target_finding=TARGET_FINDING.get(kind, ""),
                    **{k: v[k] for k in v if k not in ("kind", "level")}, **info))
        np.savez_compressed(os.path.join(a.out, "signals", f"{ecg_id}.npz"), fs=fs, leads=LEADS, **arrays)
        kept += 1
        if kept % 25 == 0:
            print(f"  {kept}/{a.n} base ECGs done ({time.time() - t0:.0f}s)")

    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(a.out, "manifest.csv"), index=False)
    pd.DataFrame(base_rows).to_csv(os.path.join(a.out, "base_ecgs.csv"), index=False)
    s = (man.groupby(["kind", "level_idx", "level"])
            .agg(n=("keep", "size"), kept=("keep", "mean"), criterion_met=("criterion_met", "mean"))
            .reset_index())
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print(f"\n{kept} base ECGs, {len(man)} edited ECGs, {int(man.keep.sum())} verified")
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
