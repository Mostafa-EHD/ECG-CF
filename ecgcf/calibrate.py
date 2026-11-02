"""Calibration of ECG-CF against real PTB-XL recordings (answers reviewer items).

All calibration uses folds 1-9 only, so nothing is tuned on the test fold 10 from which
the base recordings are drawn (except the selection-bias table, which describes fold 10).

  A. real_profiles     measurements of real NORM, AFIB and wide-QRS recordings
                       (RR CV, P presence, phase-locking index, f-wave amplitude, QRS, QT)
  B. af_edit_grid      the same measurements on our AF edits over a grid of
                       (f-wave amplitude, target RR CV) -> place edit levels on the real
                       AF distribution
  C. detector ROC      PLI / P presence / RR CV / AF rule, NORM vs AFIB
  D. selection table   fold-10 candidates: included vs excluded (residual selection bias)
  E. 12SL agreement    our QRS/QT vs GE 12SL (PTB-XL+), incl. wide-QRS recordings
"""
import ast
import os
import time

import numpy as np
import pandas as pd

import ecg_cf as E
from run_ptbxl import QUALITY_COLS, baseline_reason, clean_age, infer_sex_coding, load

AF_CODES = {"AFIB"}
WIDE_CODES = {"CLBBB", "CRBBB", "IVCD"}


def scp(db):
    return db.scp_codes.apply(ast.literal_eval)


def pick(db, group, folds, n, seed=0):
    c = scp(db)
    clean_noise = db[[k for k in ["static_noise", "burst_noise", "electrodes_problems"] if k in db]].isna().all(1)
    no_pace = db["pacemaker"].isna() if "pacemaker" in db else True
    base = db.strat_fold.isin(folds) & (db.validated_by_human == True) & no_pace  # noqa: E712
    if group == "NORM":
        ok = c.apply(lambda d: d.get("NORM", 0) == 100 and set(d) <= {"NORM", "SR"})
        ok &= db[[k for k in QUALITY_COLS if k in db]].isna().all(1)
    elif group == "AFIB":
        ok = c.apply(lambda d: bool(AF_CODES & set(d)) and "AFLT" not in d) & clean_noise
    elif group == "WIDE":
        # exclude co-existing AF/flutter so that AF-rule positives on this group are false positives
        ok = c.apply(lambda d: any(d.get(k, 0) >= 80 for k in WIDE_CODES) and not ({"AFIB", "AFLT"} & set(d))) & clean_noise
    else:
        raise ValueError(group)
    sel = db[base & ok]
    return sel.sample(n=min(n, len(sel)), random_state=seed)


def fwave_amp(x, fs, F, lead="V1"):
    """95th percentile of |4-9 Hz activity| in lead V1 during the T-Q intervals.
    Same definition as the f_amp parameter of the AF edit (which scales the 95th
    percentile of the injected f-wave in V1), measured identically on real and edited ECGs."""
    j = E.LEADS.index(lead)
    f = E._filt(x[:, j], fs, 4, 9, order=3)
    n = len(f)
    mask = np.zeros(n, bool)
    R = F["R"]
    for i in range(len(R) - 1):
        a = F["T_off"][i] if not np.isnan(F["T_off"][i]) else R[i] + E._ms(fs, 400)
        b = F["QRS_on"][i + 1] if not np.isnan(F["QRS_on"][i + 1]) else R[i + 1] - E._ms(fs, 60)
        a, b = int(a) + E._ms(fs, 40), int(b) - E._ms(fs, 20)
        if b > a:
            mask[a:b] = True
    if mask.sum() < E._ms(fs, 200):
        return np.nan
    return float(np.percentile(np.abs(f[mask]), 95))


def profile(x, fs):
    F = E.delineate(x, fs)
    m = E.measure(x, fs, F)
    f = E.findings(m)
    return dict(hr_bpm=m["hr_bpm"], rr_cv=m["rr_cv"], p_present_frac=m["p_present_frac"],
                pli=E.phase_locking_index(E._filt(x, fs, hi=40), fs, F), fwave_v1=fwave_amp(x, fs, F),
                qrs_ms=m["qrs_ms"], qt_ms=m["qt_ms"], qtc_ms=m["qtc_ms"], n_beats=m["n_beats"],
                AF_pattern=f["AF_pattern"], wide_QRS=f["wide_QRS"])


def real_profiles(ptbxl, db, folds, n_per_group, seed=0):
    rows = []
    for g in ["NORM", "AFIB", "WIDE"]:
        sel = pick(db, g, folds, n_per_group, seed)
        t0 = time.time()
        for ecg_id, r in sel.iterrows():
            try:
                x, fs = load(ptbxl, r.filename_hr)
                rows.append(dict(ecg_id=ecg_id, group=g, fold=r.strat_fold, **profile(x, fs)))
            except Exception as e:  # unreadable / undelineable record
                rows.append(dict(ecg_id=ecg_id, group=g, fold=r.strat_fold, error=str(e)[:80]))
        print(f"{g}: {len(sel)} records ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def af_edit_grid(ptbxl, db, folds, n_base, cv_grid, f_grid, f_fixed=0.05, cv_fixed=0.15, seed=1):
    """AF edits on real NORM recordings over (f_amp, cv) and the current levels 1-3"""
    sel = pick(db, "NORM", folds, n_base, seed)
    saved = dict(E.AF_SEVERITY)
    configs = [("current", lv, *saved[lv]) for lv in sorted(saved)]
    configs += [("cv_grid", None, f_fixed, cv) for cv in cv_grid]
    configs += [("f_grid", None, fa, cv_fixed) for fa in f_grid]
    rows = []
    try:
        for ecg_id, r in sel.iterrows():
            x, fs = load(ptbxl, r.filename_hr)
            F = E.delineate(x, fs)
            m0 = E.measure(x, fs, F)
            if baseline_reason(m0, None, None):
                continue
            for k, (fam, lv, fa, cv) in enumerate(configs):
                E.AF_SEVERITY[99] = (fa, cv)
                y, _ = E.atrial_fibrillation(x, fs, F, 99, np.random.default_rng(int(ecg_id) * 100 + k))
                rows.append(dict(ecg_id=ecg_id, family=fam, level=lv, f_amp=fa, cv_target=cv, **profile(y, fs)))
    finally:
        E.AF_SEVERITY.clear()
        E.AF_SEVERITY.update(saved)
    return pd.DataFrame(rows)


def roc(neg, pos):
    """AUC (pos scores higher) and the full curve, no sklearn dependency"""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    thr = np.unique(np.r_[neg, pos, np.inf, -np.inf])
    tpr = np.array([(pos >= t).mean() for t in thr])
    fpr = np.array([(neg >= t).mean() for t in thr])
    o = np.argsort(fpr + 1e-12 * tpr)
    _tz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_tz(tpr[o], fpr[o]))
    j = np.argmax(tpr - fpr)
    return dict(auc=auc, youden_thr=float(thr[j]), sens=float(tpr[j]), spec=float(1 - fpr[j]),
                n_neg=len(neg), n_pos=len(pos))


def at_threshold(neg, pos, thr, higher_is_pos=True):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    if higher_is_pos:
        return dict(sens=float((pos >= thr).mean()), spec=float((neg < thr).mean()))
    return dict(sens=float((pos < thr).mean()), spec=float((neg >= thr).mean()))


def selection_table(ptbxl, db, folds=(10,), hr=(50, 90)):
    """all clean validated NORM candidates of the test fold: included vs excluded"""
    to_sex = infer_sex_coding(db)
    c = scp(db)
    normal = c.apply(lambda d: d.get("NORM", 0) == 100 and set(d) <= {"NORM", "SR"})
    clean = db[[k for k in QUALITY_COLS if k in db]].isna().all(1)
    cand = db[normal & clean & (db.validated_by_human == True) & db.strat_fold.isin(folds)]  # noqa: E712
    rows = []
    for ecg_id, r in cand.iterrows():
        sex, age = to_sex(r.sex), clean_age(r.age)
        try:
            x, fs = load(ptbxl, r.filename_hr)
            F = E.delineate(x, fs)
            m0 = E.measure(x, fs, F)
        except Exception as e:
            rows.append(dict(ecg_id=ecg_id, sex=sex, age=age, excluded=f"unreadable ({str(e)[:40]})"))
            continue
        rows.append(dict(ecg_id=ecg_id, sex=sex, age=age, excluded=baseline_reason(m0, sex, age, hr),
                         **{k: m0[k] for k in ["hr_bpm", "qrs_ms", "qtc_ms", "p_present_frac", "stj_V2", "stj_V3",
                                               "t_V2", "t_V5"]}))
    return pd.DataFrame(rows)


def smd(a, b):
    a, b = pd.Series(a, dtype=float).dropna(), pd.Series(b, dtype=float).dropna()
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def describe_selection(sel):
    inc = sel[sel.excluded == ""]
    exc = sel[sel.excluded != ""]
    rows = []
    for col, lab in [("age", "age (y)"), ("hr_bpm", "heart rate (bpm)"), ("qrs_ms", "QRS (ms)"),
                     ("qtc_ms", "QTc Bazett (ms)"), ("stj_V2", "J-point V2 (mV)"), ("stj_V3", "J-point V3 (mV)"),
                     ("t_V2", "T amplitude V2 (mV)"), ("t_V5", "T amplitude V5 (mV)")]:
        if col not in sel:
            continue
        q = lambda d: f"{d[col].median():.3g} [{d[col].quantile(.25):.3g}-{d[col].quantile(.75):.3g}]"
        rows.append(dict(variable=lab, included=q(inc), excluded=q(exc), SMD=round(smd(inc[col], exc[col]), 2)))
    male = lambda d: (d.sex == "M").mean()
    rows.append(dict(variable="male (%)", included=f"{100 * male(inc):.0f}", excluded=f"{100 * male(exc):.0f}",
                     SMD=round(smd((inc.sex == "M").astype(float), (exc.sex == "M").astype(float)), 2)))
    rows.append(dict(variable="n", included=str(len(inc)), excluded=str(len(exc)), SMD=np.nan))
    reasons = exc.excluded.str.split(":").str[0].str.replace(r"HR \d+", "HR outside 50-90", regex=True)
    return pd.DataFrame(rows), reasons.value_counts()
