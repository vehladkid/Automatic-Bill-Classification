"""Statistical comparison XLM-R vs TF-IDF following PREREGISTRATION.md.  usage: python final_stats.py [variant ...]   (default: primary256)
Reads runs/<variant>_s<seed>/logits.npz (validation + test document-level logits) and outputs/tfidf/tfidf_probs.npz; labels/groups/real-mask from the project's split files.
PRIMARY: macro-F1 on the 386 real test documents; Delta = MEAN over seeds of (F1_seed - F1_tfidf) (point estimate). Hierarchical percentile bootstrap (B=10,000): resample test GROUPS with replacement and draw S seeds with replacement.
All classifiers are scored on the same resampled documents. Decision categories A-D are assigned mechanically from the interval. Writes outputs/stats/<variant>_stats.json (+ per-seed CSV)."""
import sys, json, csv
import numpy as np
from pathlib import Path
from scipy.stats import binomtest, t as tdist
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xlmr_lib import *
VARIANTS = sys.argv[1:] or ["primary256"]; B = 10000; ST = FIN / "outputs" / "stats"; ST.mkdir(exist_ok=True); THRESH = 0.02
tr, va, te = splits(); yv = np.array([r["y"] for r in va]); yt = np.array([r["y"] for r in te]); real = np.array([not r["synthetic"] for r in te]); y = yt[real]; g = np.array([r["group"] for r in te])[real]
ug = np.unique(g); members = [np.where(g == u)[0] for u in ug]
def mf1(y_, p_):
    sup = np.bincount(y_, minlength=NC); pc = np.bincount(p_, minlength=NC); tp = np.bincount(y_[y_ == p_], minlength=NC); m = sup > 0
    return float((2 * tp[m] / (sup[m] + pc[m])).mean())
tf = np.load(FIN / "outputs/tfidf/tfidf_probs.npz"); p_tf = tf["test"].argmax(1)[real]; tf_cal = evaluate_model("tfidf", np.log(np.clip(tf["val"], 1e-6, 1)), np.log(np.clip(tf["test"], 1e-6, 1)), va, te)["calibrated"]["real"]
def seeds_of(variant, rng=None):
    ss = sorted(int(p.name.split("_s")[-1]) for p in (FIN / "runs").glob(f"{variant}_s*") if (p / "logits.npz").exists() and json.loads((p / "config.json").read_text())["status"] == "completed")
    if rng: lo, hi = map(int, rng.split("-")); ss = [x for x in ss if lo <= x <= hi]      # e.g. primary256:0-4 (pre-registered) or primary256:0-9 (sensitivity analysis)
    return ss
def analyse(variant, rng=None, tag=None, alpha=0.05, rng_seed=20261028):
    tag = tag or variant; seeds = seeds_of(variant, rng); S = len(seeds)
    if not S: return None
    runs = {}
    for s in seeds:
        z = np.load(FIN / "runs" / f"{variant}_s{s}" / "logits.npz"); cfg = json.loads((FIN / "runs" / f"{variant}_s{s}" / "config.json").read_text()); ev = evaluate_model(f"{variant} s{s}", z["val"].astype(np.float64), z["test"].astype(np.float64), va, te)
        runs[s] = dict(pred=z["test"].argmax(1)[real], eval=ev, cfg=cfg)
    per_seed = []
    for s in seeds:
        r = runs[s]; c = r["eval"]["calibrated"]["real"]; raw = r["eval"]["raw"]["real"]; p = r["pred"]; b01 = int(((p == y) & (p_tf != y)).sum()); b10 = int(((p != y) & (p_tf == y)).sum())
        per_seed.append(dict(seed=s, accuracy=c["accuracy"], macro_f1=c["macro_f1"], macro_precision=c["macro_precision"], macro_recall=c["macro_recall"], weighted_f1=c["weighted_f1"], ece_cal=c["ece"], ece_raw=raw["ece"], nll_cal=c["nll"], brier_cal=c["brier"],
                             delta_macro_f1=c["macro_f1"] - tf_cal["macro_f1"], delta_accuracy=c["accuracy"] - tf_cal["accuracy"], relative_macro_f1_gain_pct=100 * (c["macro_f1"] - tf_cal["macro_f1"]) / tf_cal["macro_f1"], mcnemar_only_xlmr_correct=b01, mcnemar_only_tfidf_correct=b10,
                             mcnemar_exact_p=float(binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0, selected_epoch=r["cfg"]["selected_epoch"], val_real_macro_f1=r["cfg"]["best_val_real_macro_f1"], train_seconds=r["cfg"]["train_seconds"], peak_gpu_mb=r["cfg"]["peak_gpu_allocated_mb"]))
    # ---- hierarchical bootstrap ----
    rng = np.random.default_rng(rng_seed); F = np.empty((B, S)); A = np.empty((B, S)); Ft = np.empty(B); At = np.empty(B)
    for b in range(B):
        sel = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))]); ys = y[sel]; Ft[b] = mf1(ys, p_tf[sel]); At[b] = (p_tf[sel] == ys).mean()
        for k, s in enumerate(seeds): ps = runs[s]["pred"][sel]; F[b, k] = mf1(ys, ps); A[b, k] = (ps == ys).mean()
    draw = rng.integers(0, S, (B, S)); rows = np.arange(B)[:, None]
    dF_h = (F[rows, draw] - Ft[:, None]).mean(1); dA_h = (A[rows, draw] - At[:, None]).mean(1)       # seeds AND documents resampled
    dF_d = (F - Ft[:, None]).mean(1); dA_d = (A - At[:, None]).mean(1)                              # documents only (mean over the observed seeds)
    pct = lambda d, lo=alpha / 2: [float(np.percentile(d, 100 * lo)), float(np.percentile(d, 100 * (1 - lo)))]
    d_seed = np.array([p["delta_macro_f1"] for p in per_seed]); mean_d = float(d_seed.mean()); sd_d = float(d_seed.std(ddof=1)) if S > 1 else None
    t_ci = [float(mean_d - tdist.ppf(0.975, S - 1) * sd_d / np.sqrt(S)), float(mean_d + tdist.ppf(0.975, S - 1) * sd_d / np.sqrt(S))] if S > 1 else None
    ci_h = pct(dF_h); decision = ("A+B: practically meaningful superiority established" if ci_h[0] > THRESH else "A: statistically supported superiority established (not shown to exceed the 0.02 practical threshold)" if ci_h[0] > 0 and mean_d > 0 else
                                  "D: inferior performance established" if ci_h[1] < 0 else "C: inconclusive")
    ens = np.mean([softmax(np.load(FIN / "runs" / f"{variant}_s{s}" / "logits.npz")["test"] / runs[s]["eval"]["temperature"]) for s in seeds], axis=0)[real].argmax(1)
    out = dict(variant=variant, seeds=seeds, n_real_test=int(real.sum()), n_groups=len(ug), bootstrap_replicates=B, alpha=alpha, tfidf=dict(accuracy=tf_cal["accuracy"], macro_f1=tf_cal["macro_f1"], macro_precision=tf_cal["macro_precision"], macro_recall=tf_cal["macro_recall"], weighted_f1=tf_cal["weighted_f1"], ece=tf_cal["ece"], nll=tf_cal["nll"], brier=tf_cal["brier"]),
               per_seed=per_seed, stability=dict(macro_f1_mean=float(np.mean([p["macro_f1"] for p in per_seed])), macro_f1_sd=float(np.std([p["macro_f1"] for p in per_seed], ddof=1)) if S > 1 else None, macro_f1_min=float(min(p["macro_f1"] for p in per_seed)), macro_f1_max=float(max(p["macro_f1"] for p in per_seed)),
                                                  accuracy_mean=float(np.mean([p["accuracy"] for p in per_seed])), accuracy_sd=float(np.std([p["accuracy"] for p in per_seed], ddof=1)) if S > 1 else None, accuracy_min=float(min(p["accuracy"] for p in per_seed)), accuracy_max=float(max(p["accuracy"] for p in per_seed)),
                                                  delta_macro_f1_sd_over_seeds=sd_d, seeds_with_positive_delta=int((d_seed > 0).sum()), seeds_with_delta_above_0_02=int((d_seed > THRESH).sum())),
               primary=dict(delta_macro_f1=mean_d, relative_macro_f1_gain_pct=100 * mean_d / tf_cal["macro_f1"], ci95_hierarchical=ci_h, ci95_documents_only=pct(dF_d), ci95_seed_t_interval=t_ci, bootstrap_mean_delta_hierarchical=float(dF_h.mean()), p_delta_positive_hierarchical=float((dF_h > 0).mean()),
                            delta_accuracy=float(np.mean([p["delta_accuracy"] for p in per_seed])), ci95_accuracy_hierarchical=pct(dA_h), decision=decision, practical_threshold=THRESH,
                            bonferroni_ci_hierarchical={"k3": pct(dF_h, 0.05 / 3 / 2), "k4": pct(dF_h, 0.05 / 4 / 2)}),
               seed_ensemble_secondary=dict(accuracy=float((ens == y).mean()), macro_f1=mf1(y, ens), note="mean of the temperature-scaled probabilities of all seeds; a different (ensemble) classifier, shown for information only"),
               mcnemar_note="McNemar's exact test compares paired CORRECTNESS (accuracy), not macro-F1")
    pc = {}
    for c in range(NC):
        if (y == c).sum() == 0: continue
        f_seed = []; pr_tf = None
        for s in seeds:
            pcl = runs[s]["eval"]["calibrated"]["real"]["per_class"].get(LABELS[c]); f_seed.append(pcl["f1"])
        pc[LABELS[c]] = dict(support=int((y == c).sum()), tfidf_f1=tf_cal["per_class"][LABELS[c]]["f1"], xlmr_f1_mean=float(np.mean(f_seed)), xlmr_f1_min=float(min(f_seed)), xlmr_f1_max=float(max(f_seed)), delta_mean=float(np.mean(f_seed) - tf_cal["per_class"][LABELS[c]]["f1"]), seeds_better=int(sum(f > tf_cal["per_class"][LABELS[c]]["f1"] for f in f_seed)))
    out["per_class"] = pc
    with open(ST / f"{tag}_per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0].keys())); w.writeheader(); w.writerows(per_seed)
    np.save(ST / f"{tag}_hier_delta_macro_f1.npy", dF_h.astype(np.float32)); out["tag"] = tag; return out
for arg in VARIANTS:
    v, rg = (arg.split(":") + [None])[:2]; tg = v if not rg else (f"{v}_seeds{rg.replace('-', 'to')}" if rg != "0-4" or True else v)
    if rg == "0-4": tg = v                                      # the pre-registered five-seed analysis keeps the plain name
    r = analyse(v, rg, tg)
    if r is None: print(v, "no completed runs"); continue
    (ST / f"{tg}_stats.json").write_text(json.dumps(r, indent=1, default=float)); P = r["primary"]; stb = r["stability"]
    print(f"{v}: seeds {r['seeds']} | XLM-R macro-F1 mean {stb['macro_f1_mean']:.4f} sd {stb['macro_f1_sd']} [min {stb['macro_f1_min']:.4f}, max {stb['macro_f1_max']:.4f}] vs TF-IDF {r['tfidf']['macro_f1']:.4f}")
    print(f"   Delta macro-F1 = {P['delta_macro_f1']:+.4f}  (relative {P['relative_macro_f1_gain_pct']:+.1f}%)  95% hierarchical CI [{P['ci95_hierarchical'][0]:+.4f}, {P['ci95_hierarchical'][1]:+.4f}] | docs-only [{P['ci95_documents_only'][0]:+.4f}, {P['ci95_documents_only'][1]:+.4f}] | seed t-int {P['ci95_seed_t_interval']}")
    print(f"   Delta accuracy = {P['delta_accuracy']:+.4f} CI [{P['ci95_accuracy_hierarchical'][0]:+.4f}, {P['ci95_accuracy_hierarchical'][1]:+.4f}] | seeds with Delta>0: {stb['seeds_with_positive_delta']}/{len(r['seeds'])} | decision: {P['decision']}")
    for p in r["per_seed"]: print(f"   seed {p['seed']}: acc {p['accuracy']:.4f} macro-F1 {p['macro_f1']:.4f} d {p['delta_macro_f1']:+.4f} McNemar b01={p['mcnemar_only_xlmr_correct']} b10={p['mcnemar_only_tfidf_correct']} p={p['mcnemar_exact_p']:.3f} val {p['val_real_macro_f1']:.4f} epoch {p['selected_epoch']}")
