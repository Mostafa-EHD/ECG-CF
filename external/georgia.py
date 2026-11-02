"""Georgia 12-lead ECG database (G12EC), PhysioNet/CinC Challenge 2021 v1.0.3.

Records E00001..E10344, WFDB header (.hea) + MATLAB v4 signal (.mat, variable 'val',
12 x N int16, gain in the header). 500 Hz, 10 s. Diagnoses as SNOMED-CT codes on the
'#Dx:' header line.
"""
import os
import re

import numpy as np

BASE_URL = "https://physionet.org/files/challenge-2021/1.0.3/training/georgia"
N_RECORDS = 10344
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

SR = "426783006"
POS_SNOMED = {                       # ECG-CF edit kind -> SNOMED-CT codes (Challenge dx_mapping)
    "st_elev": ["164931005"],                      # ST elevation
    "qt_long": ["111975006"],                      # prolonged QT interval
    "t_inv": ["59931005"],                         # T-wave inversion
    "af": ["164889003"],                           # atrial fibrillation
    "qrs_wide": ["164909002", "733534002", "59118001", "713427006"],   # LBBB/CLBBB/RBBB/CRBBB
}
N_POS = {"st_elev": 40, "qt_long": 40, "t_inv": 20, "af": 20, "qrs_wide": 20}


def rec_name(i):
    return f"E{i:05d}"


def subfolder(rec):
    """records are stored by thousands: g1 = E00001-E01000, ..., g11 = E10001-E10344"""
    i = int(rec[1:])
    return f"g{(i - 1) // 1000 + 1}"


def parse_header(text):
    lines = [l for l in text.splitlines() if l.strip()]
    first = lines[0].split()
    n_sig, fs, n = int(first[1]), float(first[2]), int(first[3])
    names, gains, baselines = [], [], []
    for l in lines[1:1 + n_sig]:
        p = l.split()
        g = p[2]                                      # e.g. 1000/mV or 1000(0)/mV
        m = re.match(r"([\d.eE+-]+)(?:\(([-\d]+)\))?", g)
        gains.append(float(m.group(1)) or 1000.0)
        baselines.append(int(m.group(2)) if m.group(2) else int(p[4]) if len(p) > 4 and p[4].lstrip("-").isdigit() else 0)
        names.append(p[-1])
    meta = dict(fs=fs, n=n, names=names, gains=gains, baselines=baselines, dx=[], age=None, sex=None)
    for l in lines:
        if l.replace(" ", "").startswith("#Dx"):
            meta["dx"] = [c.strip() for c in l.split(":", 1)[1].split(",") if c.strip()]
        elif l.replace(" ", "").startswith("#Age"):
            v = l.split(":", 1)[1].strip()
            meta["age"] = int(float(v)) if re.match(r"^\d+(\.\d+)?$", v) else None
        elif l.replace(" ", "").startswith("#Sex"):
            v = l.split(":", 1)[1].strip().lower()
            meta["sex"] = "M" if v.startswith("m") else ("F" if v.startswith("f") else None)
    return meta


def load(folder, rec):
    """signal (n, 12) in mV, PTB-XL lead order I..V6, fs, meta"""
    from scipy.io import loadmat
    meta = parse_header(open(os.path.join(folder, rec + ".hea")).read())
    val = np.asarray(loadmat(os.path.join(folder, rec + ".mat"))["val"], dtype=np.float64)
    if val.shape[0] != len(meta["names"]):
        val = val.T
    sig = (val - np.asarray(meta["baselines"])[:, None]) / np.asarray(meta["gains"])[:, None]
    names = [s.upper() for s in meta["names"]]
    sig = sig[[names.index(l.upper()) for l in LEADS]].T
    return sig, int(meta["fs"]), meta
