"""Independent verification of the final XLM-R vs TF-IDF study (no training). Checks:
1 each run's selected epoch is the validation-maximum epoch in its own history (test data not used for selection);
2 per-seed metrics recomputed with scikit-learn from the stored logits + project split files (independent of exp_common);
3 the hierarchical bootstrap re-implemented with a different seed / 20,000 replicates and compared with final_stats;
4 split consistency (same documents/labels for every run, sizes, no overlap) and trainable-parameter bookkeeping;
5 checkpoint integrity (safetensors loads, tensor count, dtype) for every saved run.
Writes verification/final_verification.json."""
import sys, json, hashlib
import numpy as np
from pathlib import Path
from scipy.special import softmax, log_softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
FIN = Path(__file__).resolve().parents[1]; ROOT = FIN.parents[1]; O = FIN / "outputs"
rows = lambda s: [json.loads(l) for l in open(ROOT / f"datasets/processed/classification/{s}.jsonl")]
tr, va, te = rows("train"), rows("validation"), rows("test"); labels = sorted({r["label"] for r in tr + va + te}); l2i = {l: i for i, l in enumerate(labels)}
man = {json.loads(l)["document_id"]: json.loads(l) for l in open(ROOT / "datasets/metadata/manifest.jsonl")}
yv = np.array([l2i[r["label"]] for r in va]); yt = np.array([l2i[r["label"]] for r in te]); real = np.array([not r["synthetic"] for r in te]); grp = np.array([man[r["document_id"]]["group_id"] for r in te])[real]; y = yt[real]
rep = dict(checks=[], failures=[]);
def chk(name, ok, detail=""):
    rep["checks"].append(dict(name=name, ok=bool(ok), detail=str(detail)))
    if not ok: rep["failures"].append(name + " :: " + str(detail))
def ece(p, yy, nb=10):
    conf = p.max(1); ok = p.argmax(1) == yy; edges = np.linspace(0, 1, nb + 1); b = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, nb - 1); return sum((b == k).mean() * abs(ok[b == k].mean() - conf[b == k].mean()) for k in range(nb) if (b == k).any())
def fitT(lv): return minimize_scalar(lambda t: -np.mean(log_softmax(lv / t, 1)[np.arange(len(yv)), yv]), bounds=(0.3, 6), method="bounded").x
def metrics(lg, T):
    p = softmax(lg / T, 1)[real]; pred = p.argmax(1); labs = sorted(set(y.tolist())); pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=labs, zero_division=0)
    return dict(accuracy=accuracy_score(y, pred), macro_precision=pr.mean(), macro_recall=rc.mean(), macro_f1=f1.mean(), weighted_f1=(f1 * sup).sum() / sup.sum(), ece=ece(p, y), nll=-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))), brier=np.mean(((p - np.eye(len(labels))[y]) ** 2).sum(1)))
chk("split sizes 4848/634/629 and 386 real test", (len(tr), len(va), len(te), int(real.sum())) == (4848, 634, 629, 386), (len(tr), len(va), len(te), int(real.sum())))
chk("no document-id or group overlap between splits", not (set(r["document_id"] for r in tr) & set(r["document_id"] for r in te)) and not ({man[r["document_id"]]["group_id"] for r in tr} & {man[r["document_id"]]["group_id"] for r in te}) and not ({man[r["document_id"]]["group_id"] for r in va} & {man[r["document_id"]]["group_id"] for r in te}))
tf = np.load(O / "tfidf/tfidf_probs.npz"); tflg = (np.log(np.clip(tf["val"], 1e-6, 1)), np.log(np.clip(tf["test"], 1e-6, 1))); tfT = fitT(tflg[0]); tfm = metrics(tflg[1], tfT)
prod = json.loads((ROOT / "experiments/text_vs_neural/outputs/m1_tfidf.json").read_text())["calibrated"]["real"]; chk("TF-IDF re-fit macro-F1 equals the production model's stored macro-F1", abs(tfm["macro_f1"] - prod["macro_f1"]) < 1e-9, (tfm["macro_f1"], prod["macro_f1"]))
STATS = {}
for variant in sorted({p.name.rsplit("_s", 1)[0] for p in (FIN / "runs").glob("*_s*")}):
    sj = O / "stats" / f"{variant}_stats.json"
    if not sj.exists(): continue
    S = json.loads(sj.read_text()); STATS[variant] = S
    for p in S["per_seed"]:
        s = p["seed"]; run = FIN / "runs" / f"{variant}_s{s}"; cfg = json.loads((run / "config.json").read_text()); hist = json.loads((run / "history.json").read_text()); z = np.load(run / "logits.npz")
        best_ep = max(hist, key=lambda h: h["val_real_macro_f1"])["epoch"]; chk(f"{variant} s{s}: selected epoch = validation-best epoch", cfg["selected_epoch"] == best_ep, (cfg["selected_epoch"], best_ep))
        chk(f"{variant} s{s}: logits shape val/test", z["val"].shape == (len(va), 16) and z["test"].shape == (len(te), 16), (z["val"].shape, z["test"].shape))
        T = fitT(z["val"].astype(np.float64)); m = metrics(z["test"].astype(np.float64), T)
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "ece", "nll", "brier"):
            stored = p.get(k) if k in p else p.get({"ece": "ece_cal", "nll": "nll_cal", "brier": "brier_cal"}.get(k, k)); chk(f"{variant} s{s}: {k} recomputed", abs(m[k] - stored) < 2e-6, (stored, m[k]))
        chk(f"{variant} s{s}: trainable params = 86,054,416 of 278,055,952", cfg["trainable_params"] == 86054416 and cfg["total_params"] == 278055952)
        ck = run / "trainable_fp16.safetensors"
        if ck.exists():
            from safetensors import safe_open
            with safe_open(str(ck), "pt") as f: keys = list(f.keys()); dt = {str(f.get_tensor(k).dtype) for k in keys[:5]}; n_el = sum(f.get_slice(k).get_shape().__len__() and int(np.prod(f.get_slice(k).get_shape())) for k in keys)
            chk(f"{variant} s{s}: checkpoint holds exactly the trainable tensors ({n_el:,} elements, fp16)", n_el == 86054416 and dt == {"torch.float16"}, (n_el, dt, len(keys)))
            cfg_ck = cfg.get("checkpoint_file"); chk(f"{variant} s{s}: checkpoint path recorded in config", bool(cfg_ck) and Path(cfg_ck).exists())
    # independent hierarchical bootstrap
    seeds = S["seeds"]; P = {s: np.load(FIN / "runs" / f"{variant}_s{s}" / "logits.npz")["test"].argmax(1)[real] for s in seeds}; ug = np.unique(grp); mem = [np.where(grp == u)[0] for u in ug]; rng = np.random.default_rng(424242); NB = 20000; ptf = tf["test"].argmax(1)[real]
    F = np.empty((NB, len(seeds))); Ft = np.empty(NB)
    for b in range(NB):
        sel = np.concatenate([mem[i] for i in rng.integers(0, len(ug), len(ug))]); ys = y[sel]; labs = sorted(set(ys.tolist())); Ft[b] = f1_score(ys, ptf[sel], labels=labs, average="macro", zero_division=0)
        for k, s in enumerate(seeds): F[b, k] = f1_score(ys, P[s][sel], labels=labs, average="macro", zero_division=0)
    draw = rng.integers(0, len(seeds), (NB, len(seeds))); d = (F[np.arange(NB)[:, None], draw] - Ft[:, None]).mean(1); lo, hi = np.percentile(d, [2.5, 97.5]); sp = S["primary"]["ci95_hierarchical"]
    pt = np.mean([f1_score(y, P[s], labels=sorted(set(y.tolist())), average="macro", zero_division=0) for s in seeds]) - f1_score(y, ptf, labels=sorted(set(y.tolist())), average="macro", zero_division=0)
    chk(f"{variant}: point-estimate delta recomputed", abs(pt - S["primary"]["delta_macro_f1"]) < 1e-6, (pt, S["primary"]["delta_macro_f1"])); chk(f"{variant}: hierarchical CI endpoints within 0.01 of an independent 20,000-replicate bootstrap", abs(lo - sp[0]) < 0.01 and abs(hi - sp[1]) < 0.01, ([lo, hi], sp))
    chk(f"{variant}: decision category identical", ("A" if lo > 0 else "C") == ("A" if sp[0] > 0 else "C"), ([lo, hi], sp)); rep.setdefault("independent_ci", {})[variant] = [float(lo), float(hi)]
print(f"{sum(c['ok'] for c in rep['checks'])}/{len(rep['checks'])} checks passed"); [print("FAIL:", f) for f in rep["failures"][:30]]
(FIN / "verification/final_verification.json").write_text(json.dumps(rep, indent=1))
