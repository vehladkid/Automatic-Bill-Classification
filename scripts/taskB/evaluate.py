"""Field-level evaluation. For field f only documents whose source SUPPORTS f are scored (GT may be 'absent').
TP = predicted & equals GT ; FP = predicted & (GT absent or different) ; FN = GT present & (not predicted or different). Exact = normalised equality;
soft = 0.9 string similarity (text/place fields only, reported separately). Slices: all / real / synthetic / real_indian / per expense category."""
import collections
from schema import *

def doc_flags(d): return {"all": True, "real": not d["synthetic"], "synthetic": d["synthetic"], "real_indian": (not d["synthetic"]) and d["country"] == "IN"}

def _rawn(x): return " ".join(str(x).casefold().split())
def score_field(docs, preds, f, soft=False):
    tp = fp = fn = n_gt = n_pred = n_docs = n_absent = fp_absent = raw_ok = 0
    for d in docs:
        if f not in d["supported_fields"]: continue
        n_docs += 1
        g0 = d["fields"].get(f); g = g0["norm"] if g0 else None
        p0 = (preds.get(d["document_id"]) or {}).get(f); p = p0["norm"] if p0 else None
        n_gt += g is not None; n_pred += p is not None
        if g is None: n_absent += 1; fp_absent += p is not None
        if g is not None and p is not None and FIELDS[f] in ("text", "id", "gstin", "place") and _rawn(p0["value"]) == _rawn(g0["value"]): raw_ok += 1
        if p is not None and g is not None and values_match(f, p, g, soft): tp += 1
        else:
            if p is not None: fp += 1
            if g is not None: fn += 1
    pr = tp / (tp + fp) if tp + fp else None; rc = tp / (tp + fn) if tp + fn else None
    f1 = 2 * pr * rc / (pr + rc) if pr and rc else (0.0 if (pr is not None or rc is not None) and (tp + fp + fn) else None)
    typed = FIELDS[f] in ("text", "id", "gstin", "place")
    return dict(n_docs=n_docs, null_rate=None if not n_docs else round(1 - n_pred / n_docs, 4), halluc_rate=None if not n_absent else round(fp_absent / n_absent, 4), n_absent=n_absent,
                exact_raw_rate=(round(raw_ok / n_gt, 4) if (typed and n_gt) else None), n_gt=n_gt, n_pred=n_pred, tp=tp, fp=fp, fn=fn, precision=None if pr is None else round(pr, 4), recall=None if rc is None else round(rc, 4), f1=None if f1 is None else round(f1, 4))

def evaluate(docs, preds, fields=None):
    fields = fields or list(FIELDS); out = {}
    for sl in ("all", "real", "synthetic", "real_indian"):
        sub = [d for d in docs if doc_flags(d)[sl]]; out[sl] = {}
        for f in fields:
            r = score_field(sub, preds, f)
            if FIELDS[f] in ("text", "place"): r["soft_f1"] = score_field(sub, preds, f, soft=True)["f1"]
            if r["n_gt"] or r["fp"]: out[sl][f] = r
    return out

def by_category(docs, preds, fields):
    out = {}
    for c in sorted({d["expense_category"] for d in docs}):
        sub = [d for d in docs if d["expense_category"] == c]; out[c] = {"n_docs": len(sub), "n_real": sum(not d["synthetic"] for d in sub)}
        for f in fields:
            r = score_field(sub, preds, f)
            if r["n_gt"]: out[c][f] = r
    return out

def macro(res, fields):
    v = [res[f]["f1"] for f in fields if f in res and res[f]["f1"] is not None and res[f]["n_gt"] >= 5]; return round(sum(v) / len(v), 4) if v else None
def micro(res, fields):
    tp = sum(res[f]["tp"] for f in fields if f in res); fp = sum(res[f]["fp"] for f in fields if f in res); fn = sum(res[f]["fn"] for f in fields if f in res)
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / (p + r), 4) if p + r else 0.0, tp=tp, fp=fp, fn=fn)
