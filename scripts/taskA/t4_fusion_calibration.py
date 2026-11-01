"""Fuse text-LR + ConvNeXt + LayoutLMv3 (whichever exist): per-model temperature scaling on VAL, weighted log-prob average (weights by VAL NLL),
fused temperature, then the HR-routing rule: auto-process only if confidence >= tau, tau = smallest threshold whose accepted VAL-real precision >= TARGET.
Test is touched once, here. Writes reports/taskA/fusion.json + models/taskA/fusion.json + predictions CSV + confusion matrices."""
import itertools, csv
import numpy as np
from scipy.optimize import minimize_scalar
from common_a import *
TARGET = 0.95

def load_models():
    m = {}
    if (MODELS / "probs_text.npz").exists(): z = np.load(MODELS / "probs_text.npz"); m["text"] = (np.log(np.clip(z["val"], 1e-6, 1)), np.log(np.clip(z["test"], 1e-6, 1)))
    for k, f in (("image", "logits_image.npz"), ("layoutlm", "logits_layoutlm.npz")):
        if (MODELS / f).exists(): z = np.load(MODELS / f); m[k] = (z["val"], z["test"])
    return m

def fit_T(logits, y): return float(minimize_scalar(lambda t: nll(softmax(logits / t), y), bounds=(0.3, 6), method="bounded").x)

if __name__ == "__main__":
    va, te = load("val"), load("test"); yv = np.array([r["y"] for r in va]); yt = np.array([r["y"] for r in te]); M = load_models(); names = list(M)
    T = {k: fit_T(M[k][0], yv) for k in names}; cal = {k: (np.log(softmax(M[k][0] / T[k])), np.log(softmax(M[k][1] / T[k]))) for k in names}
    best = None
    for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(names)):
        if abs(sum(w) - 1) > 1e-6: continue
        lv = sum(wi * cal[k][0] for wi, k in zip(w, names)); s = nll(softmax(lv), yv)
        if best is None or s < best[0]: best = (s, w)
    w = dict(zip(names, best[1])); fv = sum(w[k] * cal[k][0] for k in names); ft = sum(w[k] * cal[k][1] for k in names); Tf = fit_T(fv, yv)
    pv, pt = softmax(fv / Tf), softmax(ft / Tf)
    rep = {"models": names, "temperatures": T, "weights": w, "fused_temperature": Tf, "single_models": {}, "fused": {"val": evaluate(va, pv), "test": evaluate(te, pt)}}
    for k in names: rep["single_models"][k] = {"val": evaluate(va, softmax(M[k][0] / T[k])), "test": evaluate(te, softmax(M[k][1] / T[k]))}
    # HR routing threshold from VAL real docs
    real = np.array([not r["synthetic"] for r in va]); conf = pv.max(1); ok = (pv.argmax(1) == yv); tau = None
    for t in np.round(np.arange(0.30, 1.0, 0.01), 2):
        m = real & (conf >= t)
        if m.sum() >= 30 and ok[m].mean() >= TARGET: tau = float(t); break
    def route(p, y, rows, t):
        c = p.max(1); a = c >= t; r_ = np.array([not x["synthetic"] for x in rows]); ok_ = p.argmax(1) == y
        return {k: dict(auto_coverage=round(float(a[m].mean()), 3), auto_precision=round(float(ok_[a & m].mean()), 3) if (a & m).any() else None, n=int(m.sum()), sent_to_hr=int((~a & m).sum()))
                for k, m in (("real", r_), ("all", np.ones(len(rows), bool)))}
    rep["routing"] = {"target_precision": TARGET, "tau": tau, "val": route(pv, yv, va, tau or 1.01), "test": route(pt, yt, te, tau or 1.01)}
    curve = [(float(t), *[(lambda a, m: (round(float(a[m].mean()), 3), round(float((pt.argmax(1) == yt)[a & m].mean()), 3) if (a & m).any() else None))(pt.max(1) >= t, np.array([not x["synthetic"] for x in te]))]) for t in np.arange(0.3, 1.0, 0.05)]
    rep["risk_coverage_test_real"] = [dict(threshold=round(t, 2), coverage=c[0], precision=c[1]) for t, c in curve]
    cm = confusion(yt[[not r["synthetic"] for r in te]], pt.argmax(1)[[not r["synthetic"] for r in te]]); np.savetxt(REPORTS / "confusion_test_real.csv", cm, fmt="%d", delimiter=",", header=",".join(LABELS), comments="")
    with open(REPORTS / "test_predictions.csv", "w", newline="") as f:
        w_ = csv.writer(f); w_.writerow(["document_id", "source", "synthetic", "country", "true", "pred", "confidence", "auto", "correct"])
        for r, p in zip(te, pt): w_.writerow([r["id"], r["source"], r["synthetic"], r["country"], r["label"], LABELS[p.argmax()], round(float(p.max()), 3), int(p.max() >= (tau or 1.01)), int(p.argmax() == r["y"])])
    save_json(REPORTS / "fusion.json", rep); save_json(MODELS / "fusion.json", {"models": names, "temperatures": T, "weights": w, "fused_temperature": Tf, "tau": tau, "labels": LABELS})
    for k in names: print(k, "test real:", rep["single_models"][k]["test"]["real"]["accuracy"], rep["single_models"][k]["test"]["real"]["macro_f1"])
    print("FUSED test real:", rep["fused"]["test"]["real"]["accuracy"], rep["fused"]["test"]["real"]["macro_f1"], "weights", {k: round(v, 2) for k, v in w.items()}, "tau", tau); print(rep["routing"]["test"])
