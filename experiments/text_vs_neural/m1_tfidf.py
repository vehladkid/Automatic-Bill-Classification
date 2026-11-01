"""Model 1 - TF-IDF + Logistic Regression (the project's existing text model, UNCHANGED configuration).
Re-fits scripts/taskA/t1_text_baseline.vec()/fit() with the C that the project selected on validation (reports/taskA/text_baseline.json), checks that the re-fit reproduces
the production model's probabilities, and records training time, CPU inference latency and model size. Production files are only read."""
import sys, time, json, joblib
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "taskA"))
from exp_common import *
import t1_text_baseline as T1

tr, va, te = splits(); C = json.loads((ROOT / "reports/taskA/text_baseline.json").read_text())["C"]
t0 = time.perf_counter(); vm = T1.fit(tr, C); train_s = time.perf_counter() - t0
pv, pt = T1.proba(vm, va), T1.proba(vm, te)
prod = joblib.load(PROD_MODELS / "text_lr.joblib"); ppv, ppt = T1.proba(prod, va), T1.proba(prod, te)
print("C =", C, "| refit train time %.1fs" % train_s, "| max |p_refit - p_production| val/test:", float(np.abs(pv - ppv).max()), float(np.abs(pt - ppt).max()))
v, m = vm; cfg = dict(C=C, class_weight="balanced", max_iter=3000, char_ngram=(2, 5), word_ngram=(1, 2), analyzer_char="char_wb", min_df=2, max_features_char=250000, max_features_word=150000,
                      sublinear_tf=True, lowercase=True, n_features=int(sum(len(t[1].vocabulary_) for t in v.transformer_list)), n_train=len(tr), n_val=len(va), n_test=len(te))
sample = [r["text"] for r in te[:40]]; i = [0]
def one():
    t = sample[i[0] % len(sample)]; i[0] += 1; m.predict_proba(v.transform([t]))
lat = time_fn(one, 5, 120)
size = (PROD_MODELS / "text_lr.joblib").stat().st_size
lg = lambda p: np.log(np.clip(p, 1e-6, 1))
save_logits("tfidf_lr", lg(pv), lg(pt))
res = evaluate_model("TF-IDF + Logistic Regression", lg(pv), lg(pt), va, te)
res.update(config=cfg, train_seconds=train_s, device="CPU", latency_cpu=lat, model_bytes=size, peak_mem_mb=None, refit_matches_production=bool(np.abs(pt - ppt).max() < 1e-6))
save_json("m1_tfidf.json", res); print({k: res[k] for k in ("temperature", "val_real_macro_f1")}); print("real test raw:", {k: round(res["raw"]["real"][k], 4) for k in ("accuracy", "macro_f1", "ece", "nll")})
print("real test calibrated:", {k: round(res["calibrated"]["real"][k], 4) for k in ("accuracy", "macro_f1", "ece", "nll")}, "| latency", lat)
