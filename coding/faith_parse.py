"""Rule-based extraction of decisions and cited components from model answers.

For each target finding category we detect (a) whether the answer *asserts* it (decision)
and (b) whether it *cites the edited component* (mention), with negation handling
("no ST elevation", "without evidence of atrial fibrillation" are not assertions).
v4 (7 Oct 2026): NegEx-style sentence-level negation scope after an audit of v3.
v5 (7 Oct 2026, second audit): IVCD "defect/abnormality", incomplete BBB excluded, hedges
("cannot be excluded") asserted, ", consistent with" scope restart, value-before-label QT,
T inversion confined to III/aVR/V1 not counted, label wording (infarct/ischaemia) is not evidence.
Validation against human annotation is reported separately.
"""
import re

# finding asserted = decision d_c ; component cited = mention m_e
DECISION = {
    "st_elev": [r"st[- ]?(segment )?elevations?", r"\bstemi\b", r"st[- ]elevated",
                r"st[- ]?segments? (is |are )?(markedly |slightly )?elevated",
                r"elevat\w* (of )?(the )?st\b", r"(acute|anterior|anteroseptal)\s+(myocardial )?(infarction|injury|mi)\b(?![^.]{0,60}\b(old|prior|previous|age undetermined|of undetermined age))",
                r"infarct\w*,? acute", r"myocardial injury",
                r"\binjury (current|pattern)"],
    "t_inv": [r"\b(t[- ]?wave|t)[- ]?inversions?", r"inverted t[- ]?waves?", r"negative t[- ]?waves?",
              r"\bt[- ]?waves? (are |is )?(inverted|negative|abnormal|flattened)", r"\bischemi\w*", r"\bt[- ]?wave abnormalit\w*",
              r"abnormal t[- ]?waves?",
              r"\bt abnormalit\w*", r"\bt[- ]?wave changes", r"st[- ]?t (wave )?(changes|abnormalit\w*)",
              r"repolari[sz]ation abnormalit\w*"],
    "qt_long": [r"(long|prolonged|lengthened) (qtc?|qt interval)", r"qtc? (interval )?(is )?prolong\w*",
                r"qtc? prolongation", r"long[- ]qt", r"qtc?[^.]{0,40}\d{3}\s*ms,? which is (\w+ )?prolonged"],
    "qrs_wide": [r"(?<!incomplete )(?<!incomplete right )(?<!incomplete left )bundle[- ]branch block", r"\b[lr]bbb\b", r"(wide|widened|broad|prolonged) qrs",
                 r"qrs (complex(es)? )?(duration )?(is |are )?(\w+ly )?(wide|widened|broad|prolonged)", r"qrs widening",
                 r"intraventricular conduction (delay|disturbance|defect|abnormalit\w*)", r"\bivcd\b"],
    "af": [r"atrial fibrillation", r"\ba[- ]?fib\b", r"\bafib\b", r"\baf\b(?! ?block)"],
}
MENTION = {
    "st_elev": [r"\bst\b", r"st[- ]segment", r"j[- ]point"],
    "t_inv": [r"t[- ]?waves?", r"\bt\b(?= (inversion|inverted|negative))"],
    "qt_long": [r"\bqtc?\b", r"qt interval"],
    "qrs_wide": [r"\bqrs\b", r"bundle[- ]branch", r"\b[lr]bbb\b"],
    "af": [r"p[- ]?waves?", r"irregular\w*", r"f[- ]?waves?", r"fibrillatory", r"rr interval\w*", r"\brr\b"],
}
NORMAL = [r"normal ecg", r"within normal limits", r"normal (sinus )?rhythm", r"no (significant |acute )?abnormalit",
          r"unremarkable"]
NEG_CUES = (r"(\bno\b|\bnot\b|\bwithout\b|\babsence of\b|\babsent\b|\bnegative for\b|\bneither\b|\bnor\b"
            r"|rule[sd]? out|ruling out|\bexclud\w*|\bdenies\b|\bor\b)")   # "or": "rules out X or Y"


def _sentences(text):
    text = strip_headings(text)
    return [s.strip() for s in re.split(r"(?<=[.!?\n])\s+|\n+", text.lower()) if s.strip()]


def _asserted_v3(pattern, sentence):
    """v3 (used up to 7 Oct 2026): negation only within 45 characters, same comma-clause.
    Kept for comparison; misses long-range negation ("... do not show deviations that would
    suggest ischemia")."""
    for m in re.finditer(pattern, sentence):
        pre = re.split(r"[,:;]| - ", sentence[max(0, m.start() - 45):m.start()])[-1]
        if re.search(NEG_CUES, pre):
            continue
        return True
    return False


# ---- v4 (7 Oct 2026, after the severe audit): NegEx-style sentence-level scope ----------
# A negation cue negates every target term that follows it in the same sentence, until a
# scope terminator (contrastive conjunction, semicolon, or an affirmative restart such as
# "consistent with", "suggestive of" without a preceding negation in that clause).
# Post-negation ("ST elevation is not present", "T-wave inversion: absent") is also handled.
NEG_CUES4 = (r"\b(no|not(?! (consistently|definitively|necessarily|always|entirely|clearly|uniformly|only|be (excluded|ruled out)|uncommon|unusual|infrequent))|without|absence of|absent|negative for|neither|nor|none|lack of|lacks|free of|"
             r"rule[sd]? out|ruling out|exclud\w*|against|unlikely|rather than|instead of)\b")
HEDGE = r"\b(cannot|can ?not|can't|could not|unable to) (be )?(exclude\w*|rule\w* out|ruled out)|not (be )?(excluded|ruled out)|cannot be excluded"
TERMINATORS = (r"\b(but|however|although|though|whereas|while|yet|except|apart from|other than|"
               r"nevertheless|nonetheless|on the other hand)\b|;|\band(?= (the )?[a-z-]+( [a-z-]+)?( waves?| segments?| complexes?| intervals?) (are|is|were|was|shows?)\b)|,\s*(which is |are |is )?(consistent with|suggestive of|indicating|suggesting)\b")
POST_NEG = re.compile(r"^\W{0,3}(?:[a-z0-9-]+\s+){0,5}?(is|are|was|were|appears?|seems?)\s+(not|absent|"
                      r"no longer|ruled out|unlikely)\b(?! (due to|caused by|related to|from|a sign of benign))|^\W{0,3}(absent|not (seen|present|observed|evident|noted|"
                      r"identified|detected))\b|^\W{0,3}(?:[a-z0-9-]+\s+){0,8}?(do|does|did)\s+not\s+"
                      r"(indicate|suggest|show|represent|reflect|support|meet|appear)\b")


def _asserted(pattern, sentence):
    """pattern found and not negated within its sentence scope (v4)"""
    for m in re.finditer(pattern, sentence):
        if re.search(r"\b(old|prior|previous)\s+(anterior|anteroseptal|septal)?\s*(myocardial )?infarct", sentence) and "infarct" in m.group(0):
            continue
        pre = sentence[:m.start()]
        cut = [t.end() for t in re.finditer(TERMINATORS, sentence) if t.end() <= m.start()]
        scope = pre[cut[-1]:] if cut else pre
        hedged = re.search(HEDGE, sentence)
        if re.search(NEG_CUES4, scope) and not hedged:
            continue
        if POST_NEG.search(sentence[m.end():m.end() + 60]) and not hedged:
            continue
        if re.search(r"\bmimic\w*", sentence[max(0, m.start() - 40):m.start()]):
            continue                      # "a benign variant that can mimic ischemic changes"
        if pattern in DECISION["t_inv"] and re.search(r"normal variant|in (lead )?(iii|avr|v1) only|only in (lead )?(iii|avr|v1)\b", sentence):
            continue                      # T inversion confined to III/aVR/V1 is normal
        return True
    return False


_HEADING = re.compile(r"\*\*[^*\n]{0,80}\*\*|^\s*#+[^\n]*$|^\s*step\s*\d+\s*[:.][^\n]{0,60}?(?=\s{2}|$)",
                      re.I | re.M)


def strip_headings(text):
    """remove template headings ("**Step 4: Ischemia, infarction & mimics**", markdown titles)
    so that heading words are not read as findings"""
    return _HEADING.sub(" ", text)


def final_answer(text):
    """reasoning models (ECG-R1) write <think>...</think> then the diagnosis: the decision is
    read from the final answer, the evidence from the whole text"""
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if m:
        return m[-1]
    return text.split("</think>")[-1] if "</think>" in text else text


GENERIC = {r"\bischemi\w*"}
T_STRICT = [r"\b(t[- ]?wave|t)[- ]?inversions?", r"inverted t[- ]?waves?", r"negative t[- ]?waves?",
            r"\bt[- ]?waves? (are |is )?(\w+ )?(inverted|negative)"]
ATRIAL = [r"atrial fibrillation", r"\ba[- ]?fib\b", r"\bafib\b", r"\baf\b(?! ?block)", r"atrial flutter",
          r"\bflutter\b", r"atrial tachyarrhythmia"]


def decision_t_strict(text):
    return any(_asserted(p, s) for s in _sentences(final_answer(text)) for p in T_STRICT)


def decision_atrial(text):
    """AF or atrial flutter (atrial tachyarrhythmia)"""
    return any(_asserted(p, s) for s in _sentences(final_answer(text)) for p in ATRIAL)     # not specific to one component (ST and T edits both evoke it)


def decision(text, kind, strict=False):
    pats = [p for p in DECISION[kind] if not (strict and p in GENERIC)]
    return any(_asserted(p, s) for s in _sentences(final_answer(text)) for p in pats)


def mention(text, kind):
    """component cited anywhere in the answer (negated or not: citing the ST segment
    to say it is normal still shows the model looked at it; reported separately)."""
    t = text.lower()
    return any(re.search(p, t) for p in MENTION[kind])


def says_normal(text):
    return any(_asserted(p, s) for s in _sentences(text) for p in NORMAL)


def parse(text):
    out = {f"dec_{k}": decision(text, k) for k in DECISION}
    out.update({f"men_{k}": mention(text, k) for k in MENTION})
    out["says_normal"] = says_normal(text)
    out["n_findings"] = sum(out[f"dec_{k}"] for k in DECISION)
    return out


if __name__ == "__main__":        # quick self-test
    tests = [
        ("Sinus rhythm. ST elevation in V1-V4 consistent with anterior STEMI.", "st_elev", True),
        ("Sinus rhythm, normal axis, no ST elevation, normal ECG.", "st_elev", False),
        ("Atrial fibrillation with rapid ventricular response; absent P waves.", "af", True),
        ("Regular rhythm without evidence of atrial fibrillation.", "af", False),
        ("Prolonged QT interval (QTc 510 ms).", "qt_long", True),
        ("QT interval is within normal limits.", "qt_long", False),
        ("Complete left bundle branch block with wide QRS complexes.", "qrs_wide", True),
        ("Narrow QRS complexes.", "qrs_wide", False),
        ("T-wave inversion in V2-V5, suggestive of ischemia.", "t_inv", True),
        ("Upright T waves, no ischemic changes.", "t_inv", False),
        ("There is no evidence of acute myocardial infarction.", "st_elev", False),
    ]
    bad = [(t, k, e) for t, k, e in tests if decision(t, k) != e]
    print("self-test:", "OK" if not bad else bad)


# =====================================================================================
# v2: evidence coding (built from real PULSE answers, 30 Sep 2026 pilot)
# The v1 `mention` counted the diagnostic label itself ("prolonged QT" contains "QT") as
# cited evidence, which made mention-given-decision trivially 1. v2 separates, per claim:
#   label      the finding is asserted
#   evidence   waveform evidence for the edited component is stated (not just the label)
#   grounded   the evidence points to the edited leads / is quantified
#   misloc     the evidence is placed on leads that were NOT edited
# =====================================================================================
LEAD_RE = r"\b(avr|avl|avf|v[1-6]|iii|ii|i)\b"
EDIT_LEADS = {"st_elev": {"v1", "v2", "v3", "v4"}, "t_inv": {"v2", "v3", "v4", "v5"}}
EDIT_REGIONS = {"st_elev": {"anterior", "anteroseptal", "septal", "precordial", "anterolateral"},
                "t_inv": {"anterior", "anteroseptal", "precordial", "anterolateral", "lateral"}}
OTHER_REGIONS = {"inferior", "high lateral", "posterior"}

EVIDENCE = {
    "st_elev": [r"st[- ]?(segment )?(is |are )?elevat\w*", r"st[- ]?(segment )?elevations?"],
    "t_inv": [r"\bt[- ]?waves? (are |is )?(inverted|negative|flattened|abnormal)",
              r"inverted t[- ]?waves?", r"\b(t[- ]?wave|t)[- ]?inversions?", r"\bt[- ]?wave (morphology )?abnormal"],
    "qt_long": [r"qtc? (interval )?(is |appears |was )?(\w+ly )?(prolonged|long|lengthened|longer)",
                r"(prolonged|long|lengthened) qtc?", r"qtc? (interval )?(of |=|:)?\s*\d{3}\s*ms"],
    "qrs_wide": [r"qrs (complex(es)? )?(duration )?(is |are )?(\w+ly )?(wide|widened|broad|prolonged|greater|longer|>)",
                 r"(wide|widened|broad|prolonged) qrs", r"qrs (duration )?(of |=|:)?\s*1\d{2}\s*ms"],
    "af": [r"(absen\w*( of)?|no|lack of|without) ((?!ectopic|premature|retrograde|inverted|abnormal|peaked|wide|notched)[a-z]+,? ){0,3}p[- ]?waves?",
           r"p[- ]?waves? (are |is )?(absent|not visible|not discernible)", r"irregular\w* (irregular )?(r[- ]?r|rhythm|ventricular)",
           r"fibrillatory|f[- ]?waves?", r"r[- ]?r intervals? (are |is )?(irregular|variable)"],
}
QUANT = {"qt_long": r"(qtc?\D{0,90}\d{3}\s*ms|\d\.\d{2}\s*s)", "qrs_wide": r"(qrs\D{0,90}1\d{2}\s*ms|>\s*1[02]0\s*ms|greater than 1[02]0)"}


def _leads(text):
    found = set(m.group(1) for m in re.finditer(LEAD_RE, text))
    for a, b in re.findall(r"v([1-6])\s*[-–to]+\s*v?([1-6])", text):   # V1-V4 ranges
        found |= {f"v{i}" for i in range(int(a), int(b) + 1)}
    return found


def _claims(text):
    """numbered/bulleted items and sentences"""
    parts = re.split(r"\n\s*\n|\n(?=\s*\d+[.)]\s)|(?<=[.;])\s+(?=[A-Z0-9])", strip_headings(text))
    return [p.strip().lower() for p in parts if p.strip()]


def evidence_code(text, kind):
    """v3 (frozen before analysis of the main runs):
    label     the target finding is asserted (decision)
    evidence  SPECIFIC waveform evidence is stated, beyond restating the label:
              ST/T  -> the abnormal ST/T claim is localised (leads or territory named);
              QT    -> a QT/QTc value is given;
              QRS   -> the QRS width itself is described (wide/prolonged/ms), not only "BBB"/"IVCD";
              AF    -> absent P waves, irregular R-R or fibrillatory waves.
    grounded  the evidence matches the edit: edited leads/territory (ST, T) or a quantified
              value (QT, QRS); for AF, same as evidence.
    misloc    ST/T evidence placed only on leads/territory that were not edited."""
    out = dict(label=decision(text, kind), evidence=False, grounded=False, misloc=False)
    for c in _claims(text):
        lab = [p for p in DECISION[kind] if kind in EDIT_LEADS and p not in GENERIC and not re.search(r"infarct|injury|stemi|ischemi", p)]
        if not any(_asserted(p, c) for p in EVIDENCE[kind] + lab):
            continue
        if kind in EDIT_LEADS:
            leads = _leads(c)
            regions = {r for r in EDIT_REGIONS[kind] | OTHER_REGIONS if r in c}
            if not (leads or regions):
                continue                                   # unlocalised: label only
            out["evidence"] = True
            if (leads & EDIT_LEADS[kind]) or (regions & EDIT_REGIONS[kind]):
                out["grounded"] = True
            else:
                out["misloc"] = True
        elif kind in QUANT:
            if kind == "qt_long":
                if re.search(QUANT[kind], c):
                    out["evidence"] = out["grounded"] = True
            else:                                          # qrs: width described
                out["evidence"] = True
                if re.search(QUANT[kind], c):
                    out["grounded"] = True
        else:
            out["evidence"] = out["grounded"] = True
    if out["grounded"]:
        out["misloc"] = False
    return out
