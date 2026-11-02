"""Explicit denial of the edited component in the answer (whole text, reasoning included)."""
import re
from faith_parse import _sentences, _asserted, DECISION
ST_TERMS = [r"st[- ]?(segment )?elevations?", r"st[- ]?segments? (is |are )?(\w+ )?elevated", r"elevat\w* (of )?(the )?st\b"]
def denies_st(text):
    for s in _sentences(text):
        if not re.search(r"\bst\b|st[- ]segment", s): continue
        if re.search(r"isoelectric|st[- ]?segments?[^.]{0,40}(within normal|normal|unremarkable|no (significant )?(deviation|elevation|change))", s):
            return True
        if any(re.search(p, s) for p in ST_TERMS) and not any(_asserted(p, s) for p in ST_TERMS):
            return True
    return False
def denies_qt(text):
    for s in _sentences(text):
        if not re.search(r"\bqtc?\b", s): continue
        if re.search(r"qtc?( interval)?s?[^.]{0,60}(within normal|normal limits|(is|are|remains?) normal|not prolonged|unremarkable|no (evidence of )?prolong)", s):
            return True
        if any(re.search(p, s) for p in DECISION["qt_long"]) and not any(_asserted(p, s) for p in DECISION["qt_long"]):
            return True
    return False


ANT = r"\bv[1-4]\b|v1\s*[-–to]+\s*v?[2-6]|v2\s*[-–to]+\s*v?[3-6]|anterior|anteroseptal|septal|precordial"


def denies_st_anterior(text):
    """denial of ST elevation in a sentence that names the edited territory (V1-V4, anterior)"""
    for s in _sentences(text):
        if not re.search(ANT, s) or not re.search(r"\bst\b|st[- ]segment", s):
            continue
        if re.search(r"isoelectric|st[- ]?segments?[^.]{0,40}(within normal|normal|unremarkable|no (significant )?(deviation|elevation|change))", s):
            return True
        if any(re.search(p, s) for p in ST_TERMS) and not any(_asserted(p, s) for p in ST_TERMS):
            return True
    return False


# ---------------------------------------------------------------------------------------------
# v6 (14 Oct 2026, after the third audit; frozen before the second human annotation round):
# also catches anterior sentences in which the negation or "normal" PRECEDES "ST segment"
# ("Leads V2 to V4 show normal QRS complexes and ST segments", "V2-V4 ... do not show significant
# Q waves or ST segment deviations"). Sentences that name V1 alone (right-ventricular remarks)
# are not anterior for this purpose; sentences asserting ST elevation are excluded.
ANT6 = r"\bv[2-4]\b|v1\s*[-–to]+\s*v?[2-6]|v2\s*[-–to]+\s*v?[3-6]|anterior|anteroseptal|septal|precordial"
_PRE6 = re.compile(r"\b(normal|no|not|without|absence of|absent|unremarkable|neither|nor|free of|lack of|isoelectric)\b"
                   r"[^.;]{0,90}?\bst[- ]?(segments?|-t|t)?\b")
_POST6 = re.compile(r"\bst[- ]?segments?\b[^.;]{0,80}(within normal|\bnormal\b|unremarkable|isoelectric|"
                    r"no (significant |acute |obvious )?(deviation|elevation|depression|change|abnormalit)|"
                    r"not (show|demonstrate|reveal|appear)\w*|without (significant )?(deviation|elevation|change))")


def _st_denial_sentence_v6(s):
    if not re.search(ANT6, s) or not re.search(r"\bst\b|st[- ]segment", s):
        return False
    if any(_asserted(p, s) for p in ST_TERMS + [r"\bstemi\b", r"st[- ]elevated"]):
        return False
    return bool(_PRE6.search(s) or _POST6.search(s))


def denies_st_anterior_v6(text):
    return denies_st_anterior(text) or any(_st_denial_sentence_v6(s) for s in _sentences(text))


# ---------------------------------------------------------------------------------------------
# v7 (20 Oct 2026, after the final audit). Fixes two scope errors of v6 found in the validation and
# the audit: (1) the sentence is split into clauses at contrastive conjunctions (but, however,
# although, whereas, while, yet) and semicolons, as stated in the paper, and the negation must be in
# the same clause as the anterior leads and the ST segment; (2) clauses that name inferior leads
# (II, III, aVF) without naming V2-V4 are not anterior even if they mention an "anterior" diagnosis.
# QT: only explicit statements that the QT/QTc is normal count (the v5/v6 branch "negated mention of
# prolonged QT" extended a preceding negation, e.g. "absence of ischaemic changes ... prolonged QT").
_CONTRAST = re.compile(r"\b(?:but|however|although|though|whereas|while|yet)\b|;")
_V24 = r"\bv[2-4]\b|v1\s*[-–to]+\s*v?[2-6]|v2\s*[-–to]+\s*v?[3-6]"
_INF = r"\b(?:ii|iii|avf)\b|inferior leads"
_QT_NORMAL = re.compile(r"qtc?( interval)?s?[^.]{0,60}(within normal|normal limits|(is|are|remains?) normal|not prolonged|unremarkable|no (evidence of )?prolong)")


def _st_denial_clause_v7(c):
    if not re.search(r"\bst\b|st[- ]segment", c) or not re.search(ANT6, c):
        return False
    if re.search(_INF, c) and not re.search(_V24, c):
        return False
    if any(_asserted(p, c) for p in ST_TERMS + [r"\bstemi\b", r"st[- ]elevated"]):
        return False
    if re.search(r"isoelectric|st[- ]?segments?[^.]{0,40}(within normal|normal|unremarkable|no (significant )?(deviation|elevation|change))", c):
        return True
    return bool(_PRE6.search(c) or _POST6.search(c))


def denies_st_anterior_v7(text):
    return any(_st_denial_clause_v7(c.strip()) for s in _sentences(text) for c in _CONTRAST.split(s))


def denies_qt_v7(text):
    return any(_QT_NORMAL.search(c) for s in _sentences(text) for c in _CONTRAST.split(s) if re.search(r"\bqtc?\b", c))
