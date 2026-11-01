"""Reproduce the production TF-IDF + Logistic Regression baseline (unchanged configuration) and save everything needed to audit it:
fitted vectoriser + classifier (joblib), per-document predictions/probabilities/labels for validation and test (CSV), metrics JSON, configuration, environment. Verifies the re-fit equals models/taskA/text_lr.joblib.
Run with low priority / 1 thread while the GPU queue is running:  OMP_NUM_THREADS=1 nice -n 19 python tfidf_repro.py"""
import sys, json, time, csv, platform
import joblib, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "taskA"))
from xlmr_lib import *
import t1_text_baseline as T1
tr, va, te = splits(); C = json.loads((ROOT / "reports/taskA/text_baseline.json").read_text())["C"]; D = FIN / "outputs" / "tfidf"; D.mkdir(exist_ok=True)
t0 = time.perf_counter(); vm = T1.fit(tr, C); train_s = time.perf_counter() - t0; v, m = vm
prod = joblib.load(PROD_MODELS / "text_lr.joblib"); pv, pt = T1.proba(vm, va), T1.proba(vm, te); qv, qt = T1.proba(prod, va), T1.proba(prod, te)
same = dict(max_abs_prob_diff_val=float(np.abs(pv - qv).max()), max_abs_prob_diff_test=float(np.abs(pt - qt).max()), identical_predictions=bool((pt.argmax(1) == qt.argmax(1)).all()))
joblib.dump(vm, D / "tfidf_lr_refit.joblib"); lg = lambda p: np.log(np.clip(p, 1e-6, 1)); np.savez(D / "tfidf_probs.npz", val=pv, test=pt)
for name, rows, P in (("val", va, pv), ("test", te, pt)):
    with open(D / f"predictions_{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["document_id", "split", "synthetic", "group", "label", "predicted"] + [f"p_{l}" for l in LABELS])
        for r, p in zip(rows, P): w.writerow([r["id"], name, int(r["synthetic"]), r["group"], LABELS[r["y"]], LABELS[int(p.argmax())]] + [f"{x:.8f}" for x in p])
res = evaluate_model("TF-IDF + LR (re-fit)", lg(pv), lg(pt), va, te)
cfg = dict(C=C, class_weight="balanced", max_iter=3000, char_ngram_range=(2, 5), word_ngram_range=(1, 2), analyzer_char="char_wb", min_df=2, max_features_char=250000, max_features_word=150000, sublinear_tf=True, lowercase=True, n_train=len(tr), n_val=len(va), n_test=len(te),
           n_features=int(sum(len(t[1].vocabulary_) for t in v.transformer_list)), train_seconds=train_s, device="CPU (1 thread, nice 19, GPU job running concurrently)", reproduces_production=same, versions=env_info(), command="OMP_NUM_THREADS=1 python experiments/xlmr_vs_tfidf_final/tfidf_repro.py",
           production_model_file="models/taskA/text_lr.joblib", production_bytes=(PROD_MODELS / "text_lr.joblib").stat().st_size, refit_bytes=(D / "tfidf_lr_refit.joblib").stat().st_size)
(D / "config.json").write_text(json.dumps(cfg, indent=1, default=float)); (D / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
print("reproduces production:", same, "| refit train %.0fs" % train_s); print("real test calibrated:", {k: round(res["calibrated"]["real"][k], 4) for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1", "ece", "nll", "brier")})
