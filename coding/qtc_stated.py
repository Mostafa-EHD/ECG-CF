"""Stated QTc value in a model answer: the first 3-digit ms value within 70 characters after
'QTc' (or 'corrected QT'), excluding thresholds (preceded by <, >, =, 'than', 'threshold', 'limit',
'above', 'below', 'exceeds', 'upper', 'normal range')."""
import re
import numpy as np
_Q = re.compile(r"(qtc|corrected qt)", re.I)
_V = re.compile(r"(\d{3})\s*(?:ms|msec|millisecond)", re.I)
_THR = re.compile(r"(<|>|=|≤|≥|than|threshold|limit|above|below|exceed\w*|upper|range|cut-?off)\s*(of\s*)?(approximately\s*)?$", re.I)

def stated_qtc(text):
    for m in _Q.finditer(text):
        win = text[m.end():m.end() + 70]
        for v in _V.finditer(win):
            pre = win[:v.start()]
            if _THR.search(pre.rstrip()):
                continue
            x = int(v.group(1))
            if 300 <= x <= 700:
                return x
    return np.nan
