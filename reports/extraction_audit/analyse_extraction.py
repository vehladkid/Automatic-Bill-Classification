"""Phase 3/5: controlled comparison on the REAL VALIDATION split with the app's real OCR path (RapidOCR 2400 + date repair), frozen Task B V1 and the cached local-LLM answers.
All systems score the same documents with the same scorer (scripts/taskB/evaluate.py). hsn_sac excluded (as in docs/TAX_AND_LLM.md).
Usage: .venv/bin/python reports/extraction_audit/analyse_extraction.py [--split val] [--v1 side2400+repair] [--llm llm_live_val_app_side2400_repair.jsonl]"""
import sys, json, argparse, collections, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/"app", ROOT/"scripts", ROOT/"scripts"/"taskB"): sys.path.insert(0, str(p))
from schema import *
from data import load
from evaluate import evaluate, micro
import llm_eval as LE
from llm import validate as V, fuse as FU
HERE = ROOT/"reports/extraction_audit"; F = LE.FIELDS_EVAL

def parse_llm(js):
    """-> {field: raw value}; keeps the model's own 'null' as absence"""
    props = (js.get("fields") if isinstance(js, dict) else None) or {}; out = {}
    for f in F:
        p = props.get(f); p = p if isinstance(p, dict) else ({"value": p} if p is not None else None); v = (p or {}).get("value")
        if v not in (None, "", []): out[f] = v
    return out

def build(docs, v1p, llm, ocrs, policy=None):
    """returns systems {name: {did: {field: pred}}} + per-proposal records for the error taxonomy"""
    S = collections.defaultdict(dict); rec = []
    for d in docs:
        did = d["document_id"]; ocr = ocrs[did]; v1 = LE.v1_fields(v1p.get(did, {})); S["v1"][did] = {f: v1p[did][f] for f in F if f in v1p.get(did, {})}
        a = llm.get(did)
        if a is None or a.get("js") is None:
            for k in ("llm_raw", "llm_validated", "fused_default", "fused_add"): S[k][did] = S["v1"][did] if k.startswith("fused") else {}
            continue
        props = parse_llm(a["js"]); raw_p, val_p, chk = {}, {}, {}
        for f, v in props.items():
            rp = LE.raw_pred(f, v)
            if rp: raw_p[f] = rp
            g = V.ground(f, v, ocr)
            if g["status"] in ("grounded", "corrected"): chk[f] = dict(value=v, ground=g); val_p[f] = dict(value=str(v), norm=g["norm"], score=None, confidence=None)
            gt = d["fields"].get(f) if f in d["supported_fields"] else "unsupported"
            rec.append(dict(did=did, f=f, v=v, status=g["status"], llm_norm=(rp or {}).get("norm"), gt=(gt or {}).get("norm") if isinstance(gt, dict) else gt, v1=(S["v1"][did].get(f) or {}).get("norm")))
        fused, _ = FU.fuse(dict(v1), chk, ocr)
        S["llm_raw"][did], S["llm_validated"][did] = raw_p, val_p
        S["fused_add"][did] = {f: LE.as_pred(f, v) for f, v in fused.items() if f in F}
        S["fused_default"][did] = {f: LE.as_pred(f, v) for f, v in fused.items() if f in F and v.get("origin") != "llm"}
        if policy: S[policy.__name__][did] = policy(d, v1, fused, chk, ocr, S["fused_default"][did])
    return S, rec

def boot(docs, A, B, n=1000, seed=0):
    rnd = random.Random(seed); idx = list(range(len(docs))); diffs = []
    per = {}
    for i, d in enumerate(docs):
        ea = evaluate([d], A, F)["real"]; eb = evaluate([d], B, F)["real"]; per[i] = (sum(r["tp"] for r in ea.values()), sum(r["fp"] for r in ea.values()), sum(r["fn"] for r in ea.values()), sum(r["tp"] for r in eb.values()), sum(r["fp"] for r in eb.values()), sum(r["fn"] for r in eb.values()))
    f1 = lambda tp, fp, fn: (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    for _ in range(n):
        s = [per[rnd.choice(idx)] for _ in idx]; ta, fa, na, tb, fb, nb = (sum(x[k] for x in s) for k in range(6)); diffs.append(f1(tb, fb, nb) - f1(ta, fa, na))
    diffs.sort(); return round(diffs[int(.025 * n)], 4), round(diffs[int(.975 * n)], 4)

SLICES = {"all": lambda d: True, "gt_ocr_source(sroie+cord)": lambda d: d["source_dataset"] in ("sroie", "cord"), "rapidocr_source(receiptbench+invoices_pt)": lambda d: d["source_dataset"] in ("receiptbench", "invoices_pt"), "real_indian": lambda d: d["country"] == "IN"}
def score(docs, S):
    out = {}
    for k, pr in S.items():
        ev = evaluate(docs, pr, F)["real"]; out[k] = dict(micro=micro(ev, F), per_field={f: dict(f1=r["f1"], tp=r["tp"], fp=r["fp"], fn=r["fn"], n_gt=r["n_gt"], halluc=r["halluc_rate"]) for f, r in ev.items()}, slices={})
        for sn, fn in SLICES.items():
            sub = [d for d in docs if fn(d)]; e2 = evaluate(sub, pr, F)["real"]; out[k]["slices"][sn] = dict(n_docs=len(sub), **micro(e2, F))
    return out

def load_all(split, v1name, llmname):
    docs = [d for d in load(split) if not d["synthetic"]]
    v1p = json.loads((HERE/f"v1_preds_{split}__{v1name}.json").read_text()); llm = {}
    for l in open(HERE/llmname):
        j = json.loads(l); llm[j["id"]] = j
    import ocr_text
    live = json.loads((HERE/f"live_ocr_{split}__{v1name.split('+')[0]}.json").read_text()); ocrs = {}
    for d in docs:
        segs = live[d["document_id"]]["segs"]; segs = ocr_text.repair_segments(segs) if v1name.endswith("+repair") else segs; ocrs[d["document_id"]] = V.OCR(segs)
    return docs, v1p, llm, ocrs

def taxonomy(rec, llm):
    """what the LLM did, by field, against ground truth (only fields the source dataset annotates)"""
    t = collections.defaultdict(collections.Counter)
    for r in rec:
        if r["gt"] == "unsupported": t[r["f"]]["proposed_on_unannotated_field"] += 1; continue
        c = t[r["f"]]; c["proposed"] += 1; c["grounded"] += r["status"] in ("grounded", "corrected"); c["ungrounded_or_invalid"] += r["status"] not in ("grounded", "corrected")
        ln = r["llm_norm"]
        if r["gt"] is None: c["value_where_gt_absent"] += 1; continue
        ok = ln is not None and values_match(r["f"], ln, r["gt"])
        c["correct"] += ok; c["wrong"] += not ok
        if r["v1"] is not None:
            v1ok = values_match(r["f"], r["v1"], r["gt"])
            c["llm_ok_v1_ok"] += ok and v1ok; c["llm_ok_v1_wrong"] += ok and not v1ok; c["llm_wrong_v1_ok"] += (not ok) and v1ok; c["llm_wrong_v1_wrong"] += (not ok) and (not v1ok)
        else: c["v1_missing_llm_ok"] += ok; c["v1_missing_llm_wrong"] += not ok
    n_bad = sum(1 for a in llm.values() if a.get("js") is None)
    return {f: dict(c) for f, c in t.items()}, n_bad

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); ap.add_argument("--v1", default="side2400+repair"); ap.add_argument("--llm", default="llm_live_val_app_side2400_repair.jsonl"); a = ap.parse_args()
    docs, v1p, llm, ocrs = load_all(a.split, a.v1, a.llm); docs = [d for d in docs if d["document_id"] in llm] if len(llm) < len(docs) else docs
    S, rec = build(docs, v1p, llm, ocrs); base_v1 = json.loads((HERE/f"v1_preds_{a.split}__base.json").read_text()); S["v1_current_app_ocr"] = {did: {f: base_v1[did][f] for f in F if f in base_v1.get(did, {})} for did in (d["document_id"] for d in docs)}
    res = score(docs, S); tx, n_bad = taxonomy(rec, llm)
    print(f"n docs {len(docs)}; LLM answers {len(llm)}; no/invalid answer {n_bad}")
    for k, v in res.items(): print(f"{k:22s} F1 {v['micro']['f1']}  P {v['micro']['precision']} R {v['micro']['recall']}  TP/FP/FN {v['micro']['tp']}/{v['micro']['fp']}/{v['micro']['fn']}")
    print("fused_default - v1(new ocr) 95% CI:", boot(docs, S["v1"], S["fused_default"]))
    for f, c in tx.items(): print(f"  {f:14s}", dict(c))
    (HERE/f"analysis_{a.split}.json").write_text(json.dumps(dict(systems=res, llm_taxonomy=tx, n_docs=len(docs), llm_no_answer=n_bad), indent=1, default=float))
