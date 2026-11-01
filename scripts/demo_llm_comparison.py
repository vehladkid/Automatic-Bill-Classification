"""Faculty demo: evaluation metrics WITHOUT vs WITH the local LLM (qwen2.5vl:3b via Ollama). CPU only, ~1 minute: it re-scores CACHED LLM answers, so no GPU, no Ollama and no model is needed.
Same documents, same ground truth, same scorer (scripts/taskB/evaluate.py) for every row. Real validation documents only; the test split is not used.
   A) 295 real validation docs, dataset OCR text      (the original benchmark, 'docs/TAX_AND_LLM.md' section G)
   B) validation docs with the app's LIVE OCR         (RapidOCR 2400 + date repair; partial: only documents the LLM answered before the run was stopped)
   C) expense classification (Task A): the LLM is not part of it
Usage:  .venv/bin/python scripts/demo_llm_comparison.py"""
import sys, json, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT/"app", ROOT/"scripts", ROOT/"scripts"/"taskB", ROOT/"reports"/"extraction_audit"): sys.path.insert(0, str(p))
import analyse_extraction as AE
import llm_eval as LE
from data import load
from llm import validate as V
F = AE.F
NAMES = {"v1": "WITHOUT LLM  (V1 extractor only)", "fused_default": "WITH LLM, app default (LLM-only values kept as suggestions)", "fused_add": "WITH LLM, write every validated LLM value into the fields", "llm_raw": "LLM alone, unvalidated", "llm_validated": "LLM alone, validated against OCR"}

def table(title, docs, S):
    res = AE.score(docs, S)
    print(f"\n{title}\n{'=' * len(title)}\n{len(docs)} real documents; fields scored: {', '.join(F)}")
    print(f"{'system':62s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s}  {'TP':>4s} {'FP':>4s} {'FN':>4s}")
    for k in NAMES:
        if k in res: m = res[k]["micro"]; print(f"{NAMES[k]:62s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}  {m['tp']:4d} {m['fp']:4d} {m['fn']:4d}")
    lo, hi = AE.boot(docs, S["v1"], S["fused_default"], n=300)
    print(f"Paired bootstrap, F1 (with LLM default) minus F1 (without): 95% CI [{lo:+.3f}, {hi:+.3f}]  ->  " + ("interval includes 0: no measurable gain" if lo <= 0 <= hi else "interval excludes 0"))
    print(f"\n{'field':16s} {'n_gt':>5s} | {'F1 without':>10s} {'F1 with':>8s} {'delta':>7s} | {'hallucination w/o':>17s} {'with':>6s}")
    for f in F:
        a, b = res["v1"]["per_field"].get(f), res["fused_default"]["per_field"].get(f)
        if a and b and a["n_gt"]: print(f"{f:16s} {a['n_gt']:5d} | {a['f1'] or 0:10.3f} {b['f1'] or 0:8.3f} {((b['f1'] or 0) - (a['f1'] or 0)):+7.3f} | {a['halluc'] if a['halluc'] is not None else float('nan'):17.3f} {b['halluc'] if b['halluc'] is not None else float('nan'):6.3f}")
    return res

def main():
    docs = [d for d in load("val") if not d["synthetic"]]
    P = pickle.load(open(ROOT/"reports/taskB/all_predictions.pkl", "rb"))["preds"]["val"]
    llm = {}
    for l in open(ROOT/"reports/taskB/llm/bench_cache_val_qwen2.5vl-3b_text.jsonl"): j = json.loads(l); llm[j["id"]] = j
    ocrs = {d["document_id"]: V.OCR(LE.segs_of(d)) for d in docs}
    S, rec = AE.build(docs, P, llm, ocrs); table("A) Field extraction, dataset OCR text", docs, S)
    tx, n_bad = AE.taxonomy(rec, llm); wrong_over_ok = sum(c.get("llm_wrong_v1_ok", 0) for c in tx.values()); ok_over_wrong = sum(c.get("llm_ok_v1_wrong", 0) for c in tx.values())
    print(f"\nLLM raw answers vs a CORRECT V1 value: LLM wrong {wrong_over_ok} times; LLM right where V1 was wrong {ok_over_wrong} times; unusable answers {n_bad}")
    try:
        docsB, v1B, llmB, ocrB = AE.load_all("val", "side2400+repair", "llm_live_val_app_side2400_repair.jsonl"); docsB = [d for d in docsB if d["document_id"] in llmB]
        SB, _ = AE.build(docsB, v1B, llmB, ocrB); table("B) Field extraction, the app's LIVE OCR (partial run)", docsB, SB)
    except Exception as e: print("\nB) live-OCR section skipped:", type(e).__name__, e)
    ta = json.loads((ROOT/"reports/extraction_audit/taskA_live_val_results.json").read_text())
    print("\nC) Expense classification (Task A, 16 classes, real validation docs)\n====================================================")
    print("The LLM never changes the expense category (by design; an LLM category experiment was written but not run), so WITHOUT and WITH are identical.")
    for k, lab in (("stored", "dataset OCR text"), ("live2400r", "app live OCR")): print(f"  {lab:18s} accuracy {ta[k]['accuracy']:.4f}   macro-F1 {ta[k]['macro_f1']:.4f}   auto-route precision {ta[k]['routing']['auto_precision']:.4f}   coverage {ta[k]['routing']['auto_coverage']:.4f}   (n={ta[k]['n']})")
    print("\nCost of the LLM: about 12 s per document on this GPU (4.6 s in the first benchmark), about 4.3 GB of the 6 GB GPU while loaded, so it is off by default.")
if __name__ == "__main__": main()
