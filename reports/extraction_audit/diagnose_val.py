"""Phase-1 diagnosis on the REAL VALIDATION split (295 docs). Read-only; uses the cached LLM answers (no model run). Writes reports/extraction_audit/diagnose_val.json.
Question per field: when V1 is wrong/missing, (a) is the GT value even present in the OCR words (ceiling), (b) did the LLM have it right, (c) did the default fusion policy let it through?"""
import sys, json, pickle, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/"app", ROOT/"scripts", ROOT/"scripts"/"taskB"): sys.path.insert(0, str(p))
from schema import *
from data import load
from evaluate import evaluate, micro
import llm_eval as LE
from llm import validate as V, fuse as FU
F = LE.FIELDS_EVAL
docs = [d for d in load("val") if not d["synthetic"]]
P = pickle.load(open(ROOT/"reports/taskB/all_predictions.pkl","rb"))["preds"]["val"]
cache = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT/"reports/taskB/llm/bench_cache_val_qwen2.5vl-3b_text.jsonl")}
out = collections.defaultdict(lambda: collections.Counter()); tot = collections.Counter()
fused_p, v1_p, add_p = {}, {}, {}
for d in docs:
    did = d["document_id"]; ocr = V.OCR(LE.segs_of(d)); v1 = LE.v1_fields(P.get(did, {})); js = (cache.get(did) or {}).get("js")
    v1_p[did] = {f: P[did][f] for f in F if f in P.get(did, {})}
    if js is None: tot["no_llm_answer"] += 1; fused_p[did] = add_p[did] = v1_p[did]; continue
    props = (js.get("fields") or {}); chk = {}; llm_norm = {}
    for f in F:
        p = props.get(f); p = p if isinstance(p, dict) else ({"value": p} if p is not None else None); v = (p or {}).get("value")
        if v in (None, "", []): continue
        g = V.ground(f, v, ocr)
        if g["status"] in ("grounded", "corrected"): chk[f] = dict(value=v, ground=g)
        llm_norm[f] = (v, g)
    fused, _ = FU.fuse(dict(v1), chk, ocr)
    fused_p[did] = {f: LE.as_pred(f, v) for f, v in fused.items() if f in F and v.get("origin") != "llm"}
    add_p[did] = {f: LE.as_pred(f, v) for f, v in fused.items() if f in F}
    for f in F:
        if f not in d["supported_fields"]: continue
        g0 = d["fields"].get(f); gt = g0["norm"] if g0 else None
        if gt is None: continue
        c = out[f]; c["n_gt"] += 1
        # ceiling: GT value visible in the OCR words?
        gv = g0["value"]; gr = V.ground(f, gt if FIELDS[f] in ("amount","date") else gv, ocr)
        in_ocr = gr["status"] in ("grounded","corrected"); c["gt_in_ocr"] += in_ocr
        pv = v1_p[did].get(f); v1ok = bool(pv and pv["norm"] is not None and values_match(f, pv["norm"], gt))
        c["v1_ok"] += v1ok; c["v1_missing"] += pv is None or pv["norm"] is None
        if f in llm_norm:
            lv, lg = llm_norm[f]; ln = normalize(f, lv) if FIELDS[f] in ("amount","date","gstin","id","int","code","enum") else norm_text(lv)
            lok = ln is not None and values_match(f, ln, gt, soft=False); c["llm_proposed"] += 1; c["llm_ok_raw"] += lok
            c["llm_grounded"] += lg["status"] in ("grounded","corrected"); c["llm_ok_grounded"] += lok and lg["status"] in ("grounded","corrected")
            if not v1ok and lok: c["llm_fixes_v1_raw"] += 1
            if v1ok and not lok: c["llm_contradicts_correct_v1_raw"] += 1
        if not v1ok and not in_ocr: c["v1_wrong_and_gt_not_in_ocr"] += 1
        if not v1ok and in_ocr: c["v1_wrong_but_gt_in_ocr"] += 1
res = {"n_docs": len(docs), "no_llm_answer": tot["no_llm_answer"], "per_field": {f: dict(c) for f, c in out.items()}}
for name, pr in (("v1", v1_p), ("fused_default_suggest", fused_p), ("fused_add", add_p)):
    ev = evaluate(docs, pr, F)["real"]; res[name] = micro(ev, F)
(ROOT/"reports/extraction_audit/diagnose_val.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: res[k] for k in ("n_docs","no_llm_answer","v1","fused_default_suggest","fused_add")}, indent=1))
for f, c in res["per_field"].items(): print(f"{f:14s}", dict(c))

# ---- slice by OCR provenance: SROIE/CORD words are dataset GROUND-TRUTH OCR, receiptbench/invoices_pt words come from RapidOCR (the app's engine)
def src(d): return "rapidocr" if d["source_dataset"] in ("receiptbench", "invoices_pt") else "ground_truth_ocr(" + d["source_dataset"] + ")"
sl = {}
for key in sorted({src(d) for d in docs}) + ["ALL_ground_truth_ocr", "ALL_rapidocr"]:
    sub = [d for d in docs if (src(d) == key or (key == "ALL_ground_truth_ocr" and src(d).startswith("ground")) or (key == "ALL_rapidocr" and src(d) == "rapidocr"))]
    sl[key] = dict(n_docs=len(sub))
    for name, pr in (("v1", v1_p), ("fused_default_suggest", fused_p), ("fused_add", add_p)):
        sl[key][name] = micro(evaluate(sub, pr, F)["real"], F)
res["by_ocr_provenance"] = sl
(ROOT/"reports/extraction_audit/diagnose_val.json").write_text(json.dumps(res, indent=1, default=float))
print("\nBY OCR PROVENANCE")
for k, v in sl.items(): print(f"{k:34s} n={v['n_docs']:3d}", {n: (v[n]['f1'], v[n]['tp'], v[n]['fp'], v[n]['fn']) for n in ("v1","fused_default_suggest","fused_add")})
