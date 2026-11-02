"""Shared queue for the review experiments (same items for ECG-R1, GEM and PULSE).

Blocks
  G   Georgia normals: original + edits + controls (middle level)
      full=True  (ECG-R1 signal): every level of every edit, incl. st_recip
      full=False (GEM, PULSE):     supra-threshold level of each edit + st_recip L3
  GP  Georgia positive controls (real recordings)
  R   PTB-XL reciprocal ST edit on the 262 unexposed base recordings
      (full: L1-L3; otherwise L3). Their originals were answered in the main runs.
"""
import json
import os

import numpy as np
import pandas as pd

SUPRA = {"st_elev": 3, "t_inv": 3, "qt_long": 3, "qrs_wide": 4, "af": 3, "st_recip": 3}
CONTROLS = [("ctrl_amp", 2), ("ctrl_noise", 2), ("ctrl_wander", 2)]
LEVELS = {"st_elev": 3, "t_inv": 3, "qt_long": 3, "qrs_wide": 4, "af": 3, "st_recip": 3}


def build_queue(work, gdir, full):
    q = []
    cf = f"{work}/ext/cf_georgia"
    man = pd.read_csv(f"{cf}/manifest.csv")
    keep = man.set_index("sample_id").keep.to_dict()
    base = pd.read_csv(f"{cf}/base_ecgs.csv")
    inc = base[base.excluded.fillna("") == ""].ecg_id.tolist()
    for e in inc:
        q.append(dict(sample_id=f"{e}_original", ecg_id=e, kind="original", level_idx=0, role="original", block="G",
                      dataset="georgia"))
        var = ([(k, li) for k, n in LEVELS.items() for li in range(1, n + 1)] if full else list(SUPRA.items())) + CONTROLS
        for k, li in var:
            sid = f"{e}_{k}_L{li}"
            if keep.get(sid, False):
                q.append(dict(sample_id=sid, ecg_id=e, kind=k, level_idx=li, block="G", dataset="georgia",
                              role="control" if k.startswith("ctrl") else "edit"))
    sel = json.load(open(os.path.join(gdir, "selection.json")))
    for kind, lst in sel["positive"].items():
        for p in lst:
            q.append(dict(sample_id=f"gpos_{kind}_{p['rec']}", ecg_id=p["rec"], kind=kind, level_idx=-1,
                          role="positive", block="GP", dataset="georgia", dx=",".join(p["dx"])))
    rman = pd.read_csv(f"{work}/ext/cf_recip/manifest.csv")
    for r in rman[rman.keep].itertuples():
        if full or r.level_idx == 3:
            q.append(dict(sample_id=r.sample_id, ecg_id=int(r.ecg_id), kind="st_recip", level_idx=int(r.level_idx),
                          role="edit", block="R", dataset="ptbxl"))
    return q


_cache = {}


def signal_of(it, work, gdir):
    """(5000, 12) mV, PTB-XL lead order"""
    if it["role"] == "positive":
        return np.load(f"{work}/ext/pos_georgia/{it['ecg_id']}.npz")["original"]
    path = (f"{work}/ext/cf_georgia/signals/{it['ecg_id']}.npz" if it["dataset"] == "georgia"
            else f"{work}/ext/cf_recip/signals/{it['ecg_id']}.npz")
    if path not in _cache:
        _cache.clear(); _cache[path] = np.load(path)
    z = _cache[path]
    return z["original"] if it["kind"] == "original" else z[it["sample_id"]]
