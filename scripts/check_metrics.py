"""Read-only accuracy check. Recomputes Task A test accuracy / macro-F1 from the SAVED model outputs (no model is run, nothing is written, no retraining)
and prints the Task B V1 numbers from the saved final evaluation.
  .venv/bin/python scripts/check_metrics.py            # both
  .venv/bin/python scripts/check_metrics.py taskA      # or taskB"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "taskA")); sys.path.insert(0, str(ROOT / "scripts"))

def taskA():
    from common_a import load, LABELS, softmax, macro_f1
    te = load("test"); y = np.array([r["y"] for r in te]); real = np.array([not r["synthetic"] for r in te]); ind = real & np.array([r["country"] == "IN" for r in te])
    cfg = json.load(open(ROOT / "models/taskA/fusion.json")); M = ROOT / "models/taskA"
    raw = {"text": np.log(np.clip(np.load(M / "probs_text.npz")["test"], 1e-6, 1)), "image": np.load(M / "logits_image.npz")["test"], "layoutlm": np.load(M / "logits_layoutlm.npz")["test"]}
    P = {k: softmax(v / cfg["temperatures"][k]) for k, v in raw.items()}
    z = sum(cfg["weights"][k] * np.log(softmax(raw[k] / cfg["temperatures"][k])) for k in raw); P["FUSION (final)"] = softmax(z / cfg["fused_temperature"])
    print("TASK A - expense category, TEST split (629 docs: real %d, real-Indian %d, synthetic %d)" % (real.sum(), ind.sum(), (~real).sum()))
    print(f"{'model':16s} {'real acc':>9s} {'real macroF1':>13s} {'Indian acc':>11s} {'all acc':>8s} {'synth acc':>10s}")
    for k, p in P.items():
        pr = p.argmax(1); acc = lambda m: (pr[m] == y[m]).mean()
        print(f"{k:16s} {acc(real):9.3f} {macro_f1(list(y[real]), list(pr[real])):13.3f} {acc(ind):11.3f} {acc(np.ones_like(real)):8.3f} {acc(~real):10.3f}")
    f = P["FUSION (final)"]; conf = f.max(1); auto = conf >= cfg["tau"]; ok = f.argmax(1) == y
    print(f"\nHR routing at tau={cfg['tau']} (real docs): auto-processed {(auto & real).sum() / real.sum():.1%}, precision of auto-processed {ok[auto & real].mean():.1%}, sent to HR {(~auto & real).sum()}")
    print("\nPer-class (fusion, real test): class  n  precision  recall  F1")
    pr = f.argmax(1)
    for i, l in enumerate(LABELS):
        n = ((y == i) & real).sum(); tp = ((pr == i) & (y == i) & real).sum(); fp = ((pr == i) & (y != i) & real).sum(); fn = ((pr != i) & (y == i) & real).sum()
        if n == 0: print(f"  {l:14s} {n:3d}      -        -       -   (no real test documents)"); continue
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1); print(f"  {l:14s} {n:3d}  {p_:9.3f} {r_:7.3f} {2 * p_ * r_ / max(p_ + r_, 1e-9):6.3f}")
    rep = json.load(open(ROOT / "reports/taskA/fusion.json"))["fused"]["test"]["real"]
    print(f"\nCross-check with saved report reports/taskA/fusion.json: real acc {rep['accuracy']}, macro-F1 {rep['macro_f1']}")

def taskB():
    r = json.load(open(ROOT / "reports/taskB/final_results.json"))
    def micro(d):
        tp = sum(v["tp"] for v in d.values() if isinstance(v, dict) and "tp" in v); fp = sum(v["fp"] for v in d.values() if isinstance(v, dict) and "fp" in v); fn = sum(v["fn"] for v in d.values() if isinstance(v, dict) and "fn" in v)
        p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); return 2 * p * rc / max(p + rc, 1e-9)
    print("TASK B V1 - field extraction, frozen TEST split (636 docs), micro-F1 from saved results (reports/taskB/final_results.json)")
    print(f"{'system':16s} {'real':>7s} {'real-Indian':>12s} {'synthetic':>10s}")
    for s in ("baseline", "ranker", "layoutlm", "hybrid"):
        m = r["by_system"][s]["test"]; print(f"{s + (' (FINAL)' if s == 'hybrid' else ''):16s} {micro(m['real']):7.3f} {micro(m['real_indian']):12.3f} {micro(m['synthetic']):10.3f}")
    print("\nHybrid per-field F1, REAL test (n = documents with a ground-truth value):")
    h = r["by_system"]["hybrid"]["test"]["real"]
    for f in ("merchant_name", "merchant_address", "invoice_number", "invoice_date", "subtotal", "tax_total", "total_amount", "gstin", "cgst", "sgst"):
        v = h.get(f)
        if v: print(f"  {f:17s} n={v['n_gt']:4d}  P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}" + ("   (tiny support - indicative only)" if v["n_gt"] < 20 else ""))
    print("  igst / hsn_sac: no real test data")
    print("\nSynthetic scores are NOT evidence of real-world accuracy; the report quotes real micro-F1 0.663 for the hybrid.")

if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "both"
    if w in ("both", "taskA"): taskA()
    if w == "both": print("\n" + "=" * 100 + "\n")
    if w in ("both", "taskB"): taskB()
