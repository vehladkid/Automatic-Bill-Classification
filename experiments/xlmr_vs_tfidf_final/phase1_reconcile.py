"""Phase 1 - reconcile the provisional numbers. For every figure in the earlier comparison (TF-IDF vs the single original XLM-R run) this recomputes it from canonical artefacts
(stored logits + the project's split files + labels) and records the source file/key. Investigates the 0.057 vs 0.053 macro-F1 difference."""
import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xlmr_lib import *
from sklearn.metrics import f1_score
tr, va, te = splits(); yt = np.array([r["y"] for r in te]); real = np.array([not r["synthetic"] for r in te]); T = ROOT / "experiments/text_vs_neural/outputs"
src = {"tfidf": ("tfidf_lr_logits.npz", "m1_tfidf.json"), "xlmr_original": ("xlmr_logits.npz", "m5_xlmr.json")}; out = dict(artifacts={}, recomputed={}, discrepancy={})
for k, (lf, jf) in src.items():
    J = json.loads((T / jf).read_text()); z = np.load(T / lf); lt = z["test"].astype(np.float64); lv = z["val"].astype(np.float64); yv = np.array([r["y"] for r in va]); Tm = fit_T(lv, yv); pc = softmax(lt / Tm)[real]; m = cls_metrics(pc, yt[real])
    out["recomputed"][k] = {q: m[q] for q in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "ece", "nll", "brier")}; out["recomputed"][k]["temperature"] = Tm
    out["artifacts"][k] = dict(logits=f"experiments/text_vs_neural/outputs/{lf}", metrics_json=f"experiments/text_vs_neural/outputs/{jf}", stored_calibrated_real={q: J["calibrated"]["real"][q] for q in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "ece", "nll", "brier")},
                               stored_raw_ece=J["raw"]["real"]["ece"], model_bytes=J.get("model_bytes"), train_seconds=J.get("train_seconds"), latency_cpu_median_ms=(J.get("latency_cpu") or {}).get("median_ms"), latency_gpu_median_ms=(J.get("latency_gpu") or {}).get("median_ms"))
    out["recomputed"][k]["max_abs_diff_vs_stored"] = max(abs(J["calibrated"]["real"][q] - m[q]) for q in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "ece", "nll", "brier"))
a, b = out["recomputed"]["xlmr_original"], out["recomputed"]["tfidf"]
d = a["macro_f1"] - b["macro_f1"]; out["discrepancy"]["point_estimate_macro_f1_diff"] = d; out["discrepancy"]["displayed_values_diff"] = round(a["macro_f1"], 3) - round(b["macro_f1"], 3)
# where does 0.057 come from?  = MEAN OF THE BOOTSTRAP DIFFERENCES stored in paired_bootstrap (results.json), not the difference of the two point estimates
R = json.loads((T / "results.json").read_text())["paired_bootstrap"]["xlmr"]; out["discrepancy"]["stored_bootstrap_mean_delta_macro_f1"] = R["delta_macro_f1"][0]; out["discrepancy"]["stored_bootstrap_ci"] = R["delta_macro_f1"][1:]
pred = {k: np.load(T / lf)["test"].argmax(1)[real] for k, (lf, _) in src.items()}; y = yt[real]; g = np.array([r["group"] for r in te])[real]; ug = np.unique(g); mem = [np.where(g == u)[0] for u in ug]; rng = np.random.default_rng(31); B = 10000; fx = np.empty(B); ft = np.empty(B)
for i in range(B):
    s = np.concatenate([mem[j] for j in rng.integers(0, len(ug), len(ug))]); ys = y[s]; labs = sorted(set(ys.tolist())); fx[i] = f1_score(ys, pred["xlmr_original"][s], labels=labs, average="macro", zero_division=0); ft[i] = f1_score(ys, pred["tfidf"][s], labels=labs, average="macro", zero_division=0)
out["discrepancy"]["bootstrap_10000"] = dict(mean_f1_xlmr=float(fx.mean()), mean_f1_tfidf=float(ft.mean()), mean_diff=float((fx - ft).mean()), point_xlmr=a["macro_f1"], point_tfidf=b["macro_f1"], bias_xlmr=float(fx.mean() - a["macro_f1"]), bias_tfidf=float(ft.mean() - b["macro_f1"]), ci95_diff=[float(x) for x in np.percentile(fx - ft, [2.5, 97.5])])
bb = out["discrepancy"]["bootstrap_10000"]
out["discrepancy"]["explanation"] = (f"The two figures measure different things. {d:.4f} (= {a['macro_f1']:.4f} - {b['macro_f1']:.4f}; displayed as 0.884 - 0.831 = 0.053 after rounding each score to three decimals) is the difference of the two macro-F1 values on the actual test set. "
    f"{R['delta_macro_f1'][0]:.3f} (stored) / {bb['mean_diff']:.3f} (10,000 resamples) is the MEAN of the macro-F1 differences over bootstrap resamples. Macro-F1 is a non-linear statistic (a class absent from a resample drops out of the average, and small classes change the denominator), so the mean over resamples is below each model's point estimate "
    f"(XLM-R {bb['bias_xlmr']:+.4f}, TF-IDF {bb['bias_tfidf']:+.4f}); the downward bias is larger for TF-IDF, which makes the difference of means exceed the point-estimate difference by about {bb['mean_diff'] - d:.3f}. "
    "It is NOT caused by different prediction files, test subsets, class-inclusion rules or averaging methods: both models use the same 386 real test documents and the same rule (macro average over classes present in the test set). The point estimate is the correct effect size; bootstrap percentiles are used only for the interval.")
out["discrepancy"]["effect_on_conclusion"] = "None: the 95% interval includes zero under either figure."
# sizes / time / latency provenance
out["provenance"] = dict(xlmr_original_fine_tuned_checkpoint_available=False, xlmr_model_bytes_refers_to="the pretrained checkpoint (same shape as the fine-tuned model)", tfidf_model_bytes_file="models/taskA/text_lr.joblib (production artefact)", xlmr_latency_note="measured on the original run's in-memory fine-tuned model (GPU bf16 batch 1, CPU fp32 4 threads); re-measured in the earlier audit with the pretrained weights: 110 ms CPU, 6.4 ms GPU")
print(json.dumps({k: (v if k != "artifacts" else "...") for k, v in out.items()}, indent=1, default=float)[:3500]); (FIN / "outputs" / "phase1_reconciliation.json").write_text(json.dumps(out, indent=1, default=float))
