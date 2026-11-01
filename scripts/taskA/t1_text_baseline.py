"""Model 1: TF-IDF (char 2-5 + word 1-2) + LogisticRegression on OCR text.
Outputs reports/taskA/text_*.json, models/taskA/text_lr.joblib, probs (val/test) + group-OOF probs (train+val) for label audit and fusion.
C is chosen on VAL real-only macro-F1; test is evaluated once at the end. Ablation: train on real only vs real+synthetic."""
import joblib, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_union
from common_a import *

def vec(): return make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=250000, sublinear_tf=True, lowercase=True),
                             TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000, sublinear_tf=True, lowercase=True))
def fit(rows, C):
    v = vec(); X = v.fit_transform([r["text"] for r in rows]); y = [r["y"] for r in rows]
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y); return v, m
def proba(vm, rows):
    v, m = vm; p = np.zeros((len(rows), len(LABELS))); p[:, m.classes_] = m.predict_proba(v.transform([r["text"] for r in rows])); return p

if __name__ == "__main__":
    tr, va, te = load("train"), load("val"), load("test"); t0 = time.time()
    realva = np.array([not r["synthetic"] for r in va]); res = {}
    for C in (3, 10, 30):
        vm = fit(tr, C); p = proba(vm, va); yv = [r["y"] for r in va]; pr = p.argmax(1)
        f = macro_f1([a for a, k in zip(yv, realva) if k], [b for b, k in zip(pr, realva) if k]); res[C] = f; print(f"C={C} val real macro-F1 {f:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    C = max(res, key=res.get); vm = fit(tr, C); joblib.dump(vm, MODELS / "text_lr.joblib")
    pv, pt = proba(vm, va), proba(vm, te); np.savez(MODELS / "probs_text.npz", val=pv, test=pt)
    rep = {"C": C, "val": evaluate(va, pv), "test": evaluate(te, pt), "val_C_search": res}
    # ablation: real-only training data
    tr_real = [r for r in tr if not r["synthetic"]]; vr = fit(tr_real, C); rep["ablation_real_only_train"] = {"val": evaluate(va, proba(vr, va)), "test": evaluate(te, proba(vr, te)), "n_train": len(tr_real)}
    # group-OOF probabilities over train+val (for label audit / diagnostics)
    allr = tr + va; oof = np.zeros((len(allr), len(LABELS))); g = [r["group"] for r in allr]
    for k, (a, b) in enumerate(GroupKFold(5).split(allr, groups=g)):
        m = fit([allr[i] for i in a], C); oof[b] = proba(m, [allr[i] for i in b]); print("fold", k, f"{time.time()-t0:.0f}s", flush=True)
    np.savez(MODELS / "oof_text.npz", ids=np.array([r["id"] for r in allr]), probs=oof)
    rep["group_cv5_trainval"] = evaluate(allr, oof); save_json(REPORTS / "text_baseline.json", rep)
    for k in ("val", "test"): print(k, {s: (rep[k][s]["accuracy"], rep[k][s]["macro_f1"]) for s in rep[k]})
    print("ablation real-only train, test:", {s: (v["accuracy"], v["macro_f1"]) for s, v in rep["ablation_real_only_train"]["test"].items()})
    print("group-CV(train+val):", {s: (v["accuracy"], v["macro_f1"]) for s, v in rep["group_cv5_trainval"].items()})
