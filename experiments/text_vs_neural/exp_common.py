"""Shared code for the text-classification comparison (TF-IDF vs BiLSTM vs BiLSTM+attention vs LexRank pipeline vs XLM-R vs LayoutLMv3).
Reuses the project's existing splits (scripts/taskA/common_a.load) and ECE (common_a.ece). Nothing in scripts/ or models/ is modified.
Every text model receives row["text"], the same OCR text of the same documents; labels, splits and metric definitions are shared.
Calibration: ONE temperature per model, fitted on VALIDATION logits only (same recipe as scripts/taskA/t4_fusion_calibration.fit_T); test data never touches it."""
import sys, json, time, platform
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "taskA")); sys.path.insert(0, str(ROOT / "scripts"))
from common_a import load, LABELS, L2I, softmax, macro_f1, ece as ece10, nll as nll_fn, MODELS as PROD_MODELS, SEED   # noqa: E402
from scipy.optimize import minimize_scalar   # noqa: E402
EXP = Path(__file__).resolve().parent; OUT = EXP / "outputs"; OUT.mkdir(exist_ok=True)
NC = len(LABELS); ECE_BINS = 10

def splits():
    tr, va, te = load("train"), load("val"), load("test")
    return tr, va, te

def fit_T(logits, y):
    return float(minimize_scalar(lambda t: nll_fn(softmax(logits / t), y), bounds=(0.3, 6), method="bounded").x)

def prf(y, p, labels):
    """per-class precision / recall / F1 / support for the given label set"""
    y = np.asarray(y); p = np.asarray(p); rows = {}
    for c in labels:
        tp = int(((y == c) & (p == c)).sum()); fp = int(((y != c) & (p == c)).sum()); fn = int(((y == c) & (p != c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0; f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        rows[c] = dict(precision=pr, recall=rc, f1=f, support=tp + fn)
    return rows

def cls_metrics(probs, y, with_calibration=True):
    """probs: (n, NC); y: int labels. Macro averages are over classes that occur in y (same convention as scripts/taskA/common_a.evaluate)."""
    y = np.asarray(y); p = probs.argmax(1); labels = sorted(set(y.tolist())); pc = prf(y, p, labels)
    sup = np.array([pc[c]["support"] for c in labels], float)
    out = dict(n=int(len(y)), accuracy=float((p == y).mean()),
               macro_precision=float(np.mean([pc[c]["precision"] for c in labels])), macro_recall=float(np.mean([pc[c]["recall"] for c in labels])),
               macro_f1=float(np.mean([pc[c]["f1"] for c in labels])), weighted_f1=float(np.sum([pc[c]["f1"] * s for c, s in zip(labels, sup)]) / sup.sum()),
               per_class={LABELS[c]: pc[c] for c in labels})
    if with_calibration:
        onehot = np.eye(NC)[y]
        out.update(ece=float(ece10(probs, y, ECE_BINS)), nll=float(nll_fn(probs, y)), brier=float(np.mean(np.sum((probs - onehot) ** 2, axis=1))))
    return out

def confusion(y, p):
    m = np.zeros((NC, NC), int)
    for a, b in zip(y, p): m[a, b] += 1
    return m

def evaluate_model(name, val_logits, test_logits, rows_val, rows_test):
    """-> dict with raw + temperature-scaled metrics on test (real subset = headline, all = secondary). T is fitted on val only."""
    yv = np.array([r["y"] for r in rows_val]); yt = np.array([r["y"] for r in rows_test]); real = np.array([not r["synthetic"] for r in rows_test])
    T = fit_T(val_logits, yv); res = dict(name=name, temperature=T)
    for tag, lg in (("raw", test_logits), ("calibrated", test_logits / T)):
        pr = softmax(lg)
        res[tag] = {"real": cls_metrics(pr[real], yt[real]), "all": cls_metrics(pr, yt)}
    res["val_real_macro_f1"] = cls_metrics(softmax(val_logits)[[not r["synthetic"] for r in rows_val]], yv[[not r["synthetic"] for r in rows_val]], False)["macro_f1"]
    res["confusion_real"] = confusion(yt[real], test_logits.argmax(1)[real]).tolist()
    return res

def paired_bootstrap(rows_test, logits_by_model, ref, n_boot=2000, seed=SEED):
    """Group-bootstrap over REAL test documents (documents of one template/merchant are not independent). The SAME resampled documents are used for every model.
    -> {model: {metric: (delta_mean, lo, hi)}} for accuracy and macro-F1 minus the reference model, plus per-model 95% CIs."""
    real_idx = [i for i, r in enumerate(rows_test) if not r["synthetic"]]; yt = np.array([rows_test[i]["y"] for i in real_idx])
    preds = {m: lg.argmax(1)[real_idx] for m, lg in logits_by_model.items()}
    groups = {}
    for k, i in enumerate(real_idx): groups.setdefault(rows_test[i]["group"], []).append(k)
    keys = list(groups); rng = np.random.default_rng(seed); acc = {m: [] for m in preds}; f1 = {m: [] for m in preds}
    for _ in range(n_boot):
        idx = np.array([k for g in rng.choice(len(keys), len(keys)) for k in groups[keys[g]]]); ys = yt[idx]; labs = sorted(set(ys.tolist()))
        for m, p in preds.items():
            ps = p[idx]; acc[m].append(float((ps == ys).mean())); f1[m].append(macro_f1(list(ys), list(ps), labs))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = {}
    for m in preds:
        d_acc = np.array(acc[m]) - np.array(acc[ref]); d_f1 = np.array(f1[m]) - np.array(f1[ref])
        out[m] = dict(accuracy_ci95=ci(acc[m]), macro_f1_ci95=ci(f1[m]), delta_accuracy=[float(d_acc.mean())] + ci(d_acc), delta_macro_f1=[float(d_f1.mean())] + ci(d_f1),
                      p_better_macro_f1=float((d_f1 > 0).mean()))
    return out

def env_info():
    import sklearn, scipy
    d = dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, seed=SEED, cpu=platform.processor() or platform.machine())
    try:
        import torch; d.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        import transformers; d["transformers"] = transformers.__version__
    except Exception: pass
    return d

def time_fn(fn, n_warm=5, n_run=50, sync=None):
    """per-call latency in ms (median, p95) after warm-up; sync() is called before reading the clock (use torch.cuda.synchronize for GPU)"""
    for _ in range(n_warm): fn()
    ts = []
    for _ in range(n_run):
        if sync: sync()
        t = time.perf_counter(); fn()
        if sync: sync()
        ts.append((time.perf_counter() - t) * 1000)
    return dict(median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)), mean_ms=float(np.mean(ts)))

def save_logits(name, val, test):
    np.savez(OUT / f"{name}_logits.npz", val=val, test=test)

def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
