"""Decode LayoutLMv3 word-level tag probabilities into field values: contiguous B-/I- spans (argmax) plus a best-B fallback span; span score = mean tag probability.
Tuned thresholds tau (val, real docs when enough) and isotonic calibration (val) are applied in final_eval.py / predict.py."""
import numpy as np, re
from schema import *
from baseline import infer_locale, infer_dayfirst
from data import bbox_of
TAGS = ["O"] + [f"{p}-{f}" for f in FIELDS for p in ("B", "I")]; T2I = {t: i for i, t in enumerate(TAGS)}

def spans_for(P, f):
    b, i_ = T2I["B-" + f], T2I["I-" + f]; am = P.argmax(1); sp = []; cur = None
    for k in range(len(am)):
        if am[k] == b: 
            if cur: sp.append(cur)
            cur = [k]
        elif am[k] == i_:
            if cur is not None and cur[-1] == k - 1: cur.append(k)
            else:
                if cur: sp.append(cur)
                cur = [k]
        else:
            if cur: sp.append(cur); cur = None
    if cur: sp.append(cur)
    out = [(float(np.mean([P[k, b] + P[k, i_] if k != s[0] else P[k, b] + P[k, i_] for k in s])), s) for s in sp]
    # fallback: strongest B word extended by confident I words
    kb = int(P[:, b].argmax())
    if P[kb, b] >= 0.05 and not any(kb in s for _, s in out):
        s = [kb]
        while s[-1] + 1 < len(P) and P[s[-1] + 1, i_] >= 0.3: s.append(s[-1] + 1)
        out.append((float(np.mean([P[k, b] + P[k, i_] for k in s])), s))
    return sorted(out, key=lambda t: -t[0])

def decode(doc, P):
    P = np.asarray(P, np.float32); words = doc["words"]; text = " ".join(words); loc = infer_locale(text); dfirst = infer_dayfirst(text); out = {}
    if not words or P.shape[0] != len(words): return out
    for f in FIELDS:
        if FIELDS[f] == "enum" and False: continue
        for sc, s in spans_for(P, f):
            val = " ".join(words[k] for k in s); n = normalize(f, val, loc, dfirst)
            if f == "currency" or n is None: 
                if f == "currency": continue
                continue
            out[f] = dict(value=val, norm=n, word_idx=list(s), bbox=bbox_of(doc, list(s)), score=min(sc, 1.0), source="layoutlmv3"); break
    return out
