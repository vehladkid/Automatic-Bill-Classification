"""Measured before/after evaluation of the LLM-assisted extraction on the frozen Task B split. READ-ONLY on datasets/ and models/; writes only reports/taskB/llm/.
Systems compared (same documents, same OCR words, same scorer as final_eval.py):
  v1            frozen hybrid (gated) from reports/taskB/all_predictions.pkl  (the numbers in docs/TASK_B_REPORT.md)
  llm_raw       LLM output used as-is (NO validation)  -> shows what validation prevents
  llm_validated LLM output after OCR-grounding checks only
  fused         v1 + validated LLM, evidence-based fusion  (what the app uses when the LLM runs)
  auto          fused only when needs_llm() fires, else v1  (what HRBC_LLM_MODE=auto does)
Fusion weights are fixed in app/llm/fuse.py and were NOT tuned on this split. hsn_sac is excluded (V1 scalar vs V2 list schema).
Usage: .venv/bin/python scripts/taskB/llm_eval.py --split test [--limit N] [--real-only]   (needs a running local LLM: see app/README.md)"""
import sys, json, argparse, time, pickle, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "app", ROOT / "scripts", ROOT / "scripts" / "taskB"):
    sys.path.insert(0, str(p))
from PIL import Image
from schema import *
from data import load
from evaluate import evaluate, micro, doc_flags
import baseline
from llm import validate as V, enhance as EN, fuse as FU
from llm.client import OllamaClient, LLMUnavailable, LLMBadOutput
from llm.config import settings
OUT = ROOT / "reports/taskB/llm"; FIELDS_EVAL = [f for f in FU.LLM_FIELDS if f != "hsn_sac"]

def segs_of(d):
    """annotated OCR words (the same input V1 saw) grouped into text lines; per-line confidence is unknown in the annotations -> 1.0"""
    out = []
    for l in baseline.lines_of(d["words"], d["boxes"]):
        bs = [d["boxes"][i] for i in l["idx"]]; out.append(dict(text=l["text"], bbox=[int(min(b[0] for b in bs)), int(min(b[1] for b in bs)), int(max(b[2] for b in bs)), int(max(b[3] for b in bs))], conf=1.0))
    return out

def v1_fields(preds):
    t = lambda f, p: p["norm"] if FIELDS[f] in ("amount", "date", "gstin", "id", "int", "code", "enum") else p["value"]
    return {f: dict(value=t(f, p), raw_text=p["value"], confidence=p.get("confidence"), source=p.get("source"), bbox=p.get("bbox")) for f, p in preds.items() if f in FIELDS_EVAL and p.get("norm") is not None}

def as_pred(f, v):
    val = v["value"]; norm = S_norm(f, val) if FIELDS[f] in ("text", "place") else val
    return dict(value=val if isinstance(val, str) else (v.get("raw_text") or str(val)), norm=norm, score=None, confidence=v.get("evidence_score"))
def S_norm(f, val): return norm_text(val)

def raw_pred(f, p):
    """LLM value with no validation at all (but normalised the same way)."""
    n = normalize(f, p)
    return None if n is None else dict(value=str(p), norm=n, score=None, confidence=None)

def cache_get(cache, did): return cache.get(did)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int); ap.add_argument("--real-only", action="store_true"); ap.add_argument("--no-image", action="store_true"); a = ap.parse_args()
    cfg = settings() | {"mode": "always"}; 
    if a.no_image: cfg["use_image"] = False
    cl = OllamaClient(cfg); okm, names = cl.available()
    if not okm: raise SystemExit(f"LLM model {cfg['model']!r} not available at {cfg['url']} (have: {names}). See app/README.md.")
    docs = load("test" if a.split == "test" else "val"); docs = [d for d in docs if not (a.real_only and d["synthetic"])]
    docs = [d for d in docs if not d["synthetic"]]                  # real documents only: synthetic scores are a pipeline check, never a result
    if a.limit: docs = docs[: a.limit]
    sp = "test" if a.split == "test" else "val"; P = pickle.load(open(ROOT / "reports/taskB/all_predictions.pkl", "rb"))["preds"][sp]
    OUT.mkdir(parents=True, exist_ok=True); tag = f"{sp}_{cfg['model'].replace(':', '-').replace('/', '_')}{'_noimg' if a.no_image else ''}"
    cache_f = OUT / f"cache_{tag}.jsonl"; cache = {}
    if cache_f.exists(): cache = {json.loads(l)["id"]: json.loads(l) for l in open(cache_f)}
    sysp = {k: {} for k in ("v1", "llm_raw", "llm_validated", "fused", "auto")}; meta = []; fails = collections.Counter(); t0 = time.time(); rej_total = 0; hall = 0
    for i, d in enumerate(docs):
        did = d["document_id"]; segs = segs_of(d); v1 = v1_fields(P.get(did, {})); ocr = V.OCR(segs)
        if did in cache: js = cache[did]["js"]; secs = cache[did]["secs"]
        else:
            try: js, raw, secs = EN.run_llm(cl, segs, (d["width"], d["height"]), Image.open(ROOT / "datasets" / d["image"]).convert("RGB") if cfg["use_image"] else None)
            except (LLMUnavailable, LLMBadOutput) as e: fails[type(e).__name__] += 1; js, secs = None, None
            with open(cache_f, "a") as f: f.write(json.dumps(dict(id=did, js=js, secs=secs)) + "\n")
        if js is None: sysp["fused"][did] = sysp["auto"][did] = {f: dict(p, norm=P[did][f]["norm"]) for f, p in ((f, P.get(did, {}).get(f)) for f in FIELDS_EVAL) if p}; sysp["v1"][did] = sysp["fused"][did]; continue
        props = (js.get("fields") if isinstance(js, dict) else None) or {}
        raw_p, val_p, checked = {}, {}, {}
        for f in FIELDS_EVAL:
            p = props.get(f); p = p if isinstance(p, dict) else ({"value": p} if p is not None else None)
            v = (p or {}).get("value")
            if v in (None, "", []): continue
            rp = raw_pred(f, v)
            if rp: raw_p[f] = rp
            g = V.ground(f, v, ocr)
            if g["status"] in ("grounded", "corrected"): checked[f] = dict(value=v, ground=g); val_p[f] = dict(value=str(v), norm=g["norm"], score=None, confidence=None)
            else: rej_total += 1
        fused, _ = FU.fuse({f: p for f, p in v1.items()}, checked, ocr)
        reasons = EN.needs_llm(v1, ocr)
        sysp["v1"][did] = {f: P[did][f] for f in FIELDS_EVAL if f in P.get(did, {})}; sysp["llm_raw"][did] = raw_p; sysp["llm_validated"][did] = val_p
        sysp["fused"][did] = {f: as_pred(f, v) for f, v in fused.items() if f in FIELDS_EVAL}
        sysp["auto"][did] = sysp["fused"][did] if reasons else sysp["v1"][did]
        meta.append(dict(id=did, secs=secs, triggered=bool(reasons), flags={f: (v["needs_review"], v["level"]) for f, v in fused.items() if f in FIELDS_EVAL}))
        if (i + 1) % 20 == 0: print(f"{i + 1}/{len(docs)}  {time.time() - t0:.0f}s", flush=True)
    res = {"split": sp, "model": cfg["model"], "image_used": cfg["use_image"], "n_docs": len(docs), "n_scored": len(meta), "llm_failures": dict(fails), "fields_scored": FIELDS_EVAL, "systems": {}, "note": "real documents only; fusion weights fixed in app/llm/fuse.py, not tuned on this split"}
    scored = [d for d in docs if d["document_id"] in {m["id"] for m in meta}]
    for s, preds in sysp.items():
        ev = evaluate(scored, preds, FIELDS_EVAL); res["systems"][s] = {sl: dict(micro=micro(ev[sl], FIELDS_EVAL), per_field={f: dict(f1=r["f1"], precision=r["precision"], recall=r["recall"], n_gt=r["n_gt"], halluc_rate=r["halluc_rate"], fp=r["fp"], fn=r["fn"]) for f, r in ev[sl].items()}) for sl in ("real", "real_indian")}
    # review-flag usefulness for the fused system (real): are flagged fields the wrong ones?
    byl = collections.defaultdict(lambda: [0, 0]); fl = [0, 0, 0, 0]       # [flagged&wrong, flagged&right, unflagged&wrong, unflagged&right]
    mm = {m["id"]: m for m in meta}
    for d in scored:
        for f, p in sysp["fused"][d["document_id"]].items():
            if f not in d["supported_fields"]: continue
            g = d["fields"].get(f); ok = g is not None and values_match(f, p["norm"], g["norm"]); lv = mm[d["document_id"]]["flags"].get(f)
            if lv: byl[lv[1]][0] += 1; byl[lv[1]][1] += ok; fl[(0 if ok is False else 1) + (0 if lv[0] else 2)] += 0; fl[(0 if lv[0] else 2) + (1 if ok else 0)] += 1
    res["fused_accuracy_by_evidence_level"] = {k: dict(n=v[0], precision=round(v[1] / v[0], 4)) for k, v in byl.items() if v[0]}
    res["fused_review_flags"] = dict(flagged_wrong=fl[0], flagged_right=fl[1], unflagged_wrong=fl[2], unflagged_right=fl[3])
    res["llm_validation"] = dict(proposals_rejected_as_ungrounded=rej_total)
    secs = [m["secs"] for m in meta if m["secs"]]; res["latency_seconds"] = dict(mean=round(sum(secs) / len(secs), 2) if secs else None, n=len(secs))
    res["auto_mode"] = dict(llm_triggered_docs=sum(m["triggered"] for m in meta), of=len(meta))
    (OUT / f"llm_eval_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({s: {sl: v[sl]["micro"] for sl in v} for s, v in res["systems"].items()}, indent=1)); print("by level:", res["fused_accuracy_by_evidence_level"], "flags:", res["fused_review_flags"], "auto:", res["auto_mode"], "latency:", res["latency_seconds"])
if __name__ == "__main__": main()
