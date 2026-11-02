"""Build the counterfactuals for the review experiments (CPU, parallel).

A. Georgia (G12EC), external replication: normal candidates of selection.json in order,
   same inclusion rule as PTB-XL (HR 50-90 bpm, P waves, our measurements normal),
   until N_BASE are included; all edits (incl. st_recip) x all levels + controls, verified.
   -> $ECGCF_WORK/ext/cf_georgia/{signals/*.npz, manifest.csv, base_ecgs.csv}
B. PTB-XL, reciprocal ST edit on the 262 unexposed base recordings of the main run.
   -> $ECGCF_WORK/ext/cf_recip/{signals/*.npz, manifest.csv}

Deterministic (seeds from the record id), so PULSE on Kaggle can regenerate the same edits.
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import ecg_cf as E                                  # noqa: E402
from run_ptbxl import baseline_reason, load as load_ptbxl   # noqa: E402
import georgia as G                                 # noqa: E402

N_BASE = int(os.environ.get("N_BASE", 150))


def rows_for(rid, sig, fs, F, m0, sex, age, kinds, seed_base):
    arrays, rows = {}, []
    for kind in kinds:
        spec = E.CATALOGUE[kind]
        for li, level in enumerate(spec["levels"], start=1):
            y, info = E.perturb(sig, fs, F, kind, level, seed=seed_base * 10 + li)
            v, _ = E.verify(m0, y, fs, kind, level, sex=sex, age=age, achieved=info.get("achieved_ms_mean"))
            sid = f"{rid}_{kind}_L{li}"
            arrays[sid] = y.astype(np.float32)
            rows.append(dict(sample_id=sid, ecg_id=rid, sex=sex, age=age, kind=kind, level=level, level_idx=li,
                             unit=spec["unit"], is_control=kind.startswith("ctrl"),
                             target_finding=E.TARGET_FINDING.get(kind, ""),
                             **{k: v[k] for k in v if k not in ("kind", "level")}, **info))
    return arrays, rows


def georgia_one(args):
    rec, folder, out = args
    try:
        sig, fs, meta = G.load(folder, rec)
        sig = sig[:5000]
        F = E.delineate(sig, fs)
        m0 = E.measure(sig, fs, F)
    except Exception as e:                      # noqa: BLE001
        return dict(ecg_id=rec, excluded=f"unreadable: {str(e)[:60]}"), []
    reason = baseline_reason(m0, meta["sex"], meta["age"])
    base = dict(ecg_id=rec, age=meta["age"], sex=meta["sex"], excluded=reason, **m0)
    if reason:
        return base, []
    arrays, rows = rows_for(rec, sig, fs, F, m0, meta["sex"], meta["age"], list(E.CATALOGUE), int(rec[1:]))
    np.savez_compressed(os.path.join(out, "signals", f"{rec}.npz"), fs=fs, original=sig.astype(np.float32), **arrays)
    return base, rows


def recip_one(args):
    ecg_id, fname, sex, age, ptbxl, out = args
    sig, fs = load_ptbxl(ptbxl, fname)
    F = E.delineate(sig, fs)
    m0 = E.measure(sig, fs, F)
    arrays, rows = rows_for(int(ecg_id), sig, fs, F, m0, sex, age, ["st_recip"], int(ecg_id))
    np.savez_compressed(os.path.join(out, "signals", f"{ecg_id}.npz"), fs=fs, original=sig.astype(np.float32), **arrays)
    return rows


def main():
    work = os.environ["ECGCF_WORK"]
    gdir = os.environ["GEORGIA_DIR"]
    ptbxl = os.environ["PTBXL_DIR"]
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    t0 = time.time()

    # ---------------- A. Georgia ----------------
    out = f"{work}/ext/cf_georgia"
    if not os.path.exists(f"{out}/manifest.csv"):
        os.makedirs(f"{out}/signals", exist_ok=True)
        sel = json.load(open(os.path.join(gdir, "selection.json")))
        bases, rows, n_inc = [], [], 0
        cand = sel["normal"]
        with Pool(ncpu) as pool:
            for i in range(0, len(cand), ncpu * 4):          # chunks, in order, stop at N_BASE
                for b, r in pool.map(georgia_one, [(c, gdir, out) for c in cand[i:i + ncpu * 4]]):
                    if n_inc >= N_BASE and not b["excluded"]:
                        b["excluded"] = "not needed (N_BASE reached)"
                        os.remove(os.path.join(out, "signals", f"{b['ecg_id']}.npz"))
                        r = []
                    bases.append(b); rows += r
                    n_inc += int(not b["excluded"])
                print(f"Georgia: {len(bases)} candidates read, {min(n_inc, N_BASE)} included ({time.time()-t0:.0f} s)", flush=True)
                if n_inc >= N_BASE:
                    break
        pd.DataFrame(rows).to_csv(f"{out}/manifest.csv", index=False)
        pd.DataFrame(bases).to_csv(f"{out}/base_ecgs.csv", index=False)
    man = pd.read_csv(f"{out}/manifest.csv")
    b = pd.read_csv(f"{out}/base_ecgs.csv")
    print(f"\nGeorgia: {int((b.excluded.fillna('') == '').sum())} base ECGs included; exclusions:")
    print(b.excluded.fillna("").str.split(":").str[0].str.replace(r"HR \d+", "HR", regex=True).value_counts().to_string())
    print(man.groupby(["kind", "level_idx"]).agg(kept=("keep", "mean"), crit=("criterion_met", "mean")).round(2).unstack().to_string())

    # ---------------- B. PTB-XL reciprocal ST ----------------
    out = f"{work}/ext/cf_recip"
    if not os.path.exists(f"{out}/manifest.csv"):
        os.makedirs(f"{out}/signals", exist_ok=True)
        base = pd.read_csv(f"{work}/main_r1/cf/base_ecgs.csv")
        train = set(json.load(open("ptbxl_ids_in_training.json")))
        inc = base[(base.excluded.fillna("") == "") & ~base.ecg_id.isin(train)]
        db = pd.read_csv(os.path.join(ptbxl, "ptbxl_database.csv"), index_col="ecg_id")
        jobs = [(int(r.ecg_id), db.loc[int(r.ecg_id), "filename_hr"], r.sex, int(r.age) if r.age == r.age else None, ptbxl, out)
                for r in inc.itertuples()]
        print(f"\nPTB-XL ST + mirrored image: {len(jobs)} base ECGs", flush=True)
        with Pool(ncpu) as pool:
            rows = sum(pool.map(recip_one, jobs), [])
        pd.DataFrame(rows).to_csv(f"{out}/manifest.csv", index=False)
    man = pd.read_csv(f"{out}/manifest.csv")
    print(man.groupby("level_idx").agg(n=("keep", "size"), kept=("keep", "mean"), crit=("criterion_met", "mean"),
                                       st=("d_st_target_mv", "mean"), recip=("d_st_recip_mv", "mean")).round(3).to_string())
    # ---------------- C. Georgia positive controls, saved as numpy (no scipy needed later) ----------------
    out = f"{work}/ext/pos_georgia"
    os.makedirs(out, exist_ok=True)
    sel = json.load(open(os.path.join(gdir, "selection.json")))
    n = 0
    for kind, lst in sel["positive"].items():
        for p in lst:
            f = os.path.join(out, p["rec"] + ".npz")
            if not os.path.exists(f):
                sig, fs, _ = G.load(gdir, p["rec"])
                np.savez_compressed(f, fs=fs, original=sig[:5000].astype(np.float32))
            n += 1
    print(f"\nGeorgia positive controls saved: {n}")
    print(f"\nDONE ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    sys.exit(main())
