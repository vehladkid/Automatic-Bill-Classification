"""Hybrid cascade evaluation (pre-registered in PREREGISTRATION.md, Addendum 4). TF-IDF first; if top probability < tau the document goes to XLM-R. tau per seed chosen on VALIDATION real-only macro-F1 only.
Evaluated on the main real test set and on the unseen sets with frozen models. -> outputs/hybrid/hybrid_stats.json"""
import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xlmr_lib import *
from scipy.stats import t as tdist
B = 10000; THRESH = 0.02; OUT = FIN / "outputs" / "hybrid"; OUT.mkdir(exist_ok=True); O = FIN / "outputs"
GRID = [0.0] + [round(0.30 + 0.05 * i, 2) for i in range(14)] + [1.01]     # 0 = never route, 1.01 = always XLM-R
def mf1(y_, p_, classes=None):
    sup = np.bincount(y_, minlength=NC); pc = np.bincount(p_, minlength=NC); tp = np.bincount(y_[y_ == p_], minlength=NC); m = (sup > 0) if classes is None else np.isin(np.arange(NC), classes) & (sup > 0)
    return float((2 * tp[m] / (sup[m] + pc[m])).mean())
def hyb(ptf, xl, tau): conf = ptf.max(1); route = conf < tau; return np.where(route, xl, ptf.argmax(1)), route
tr, va, te = splits(); yv = np.array([r["y"] for r in va]); real_v = np.array([not r["synthetic"] for r in va]); yt = np.array([r["y"] for r in te]); real = np.array([not r["synthetic"] for r in te]); gt = np.array([r["group"] for r in te])
tf = np.load(O / "tfidf/tfidf_probs.npz"); seeds = sorted(int(p.name.split("_s")[-1]) for p in (FIN / "runs").glob("primary256_s*") if (p / "logits.npz").exists() and json.loads((p / "config.json").read_text())["status"] == "completed")
Z = {s: np.load(FIN / "runs" / f"primary256_s{s}" / "logits.npz") for s in seeds}
# ---- tau per seed from validation ----
tau = {}; valcurve = {}
for s in seeds:
    xv = Z[s]["val"].argmax(1); sc = []
    for t_ in GRID: p, _ = hyb(tf["val"], xv, t_); sc.append(mf1(yv[real_v], p[real_v]))
    best = max(sc); tau[s] = max(t_ for t_, v in zip(GRID, sc) if v >= best - 1e-12); valcurve[s] = dict(zip(map(str, GRID), sc))
print("tau per seed:", tau)
def run_set(y, ptf, X, grp, mask, classes, seed_list, tag, curve=False):
    idx = np.where(mask)[0]; yy = y[idx]; gg = grp[idx]; ug = np.unique(gg); members = [np.where(gg == u)[0] for u in ug]; S = len(seed_list)
    PH, RT, PX = {}, {}, {}
    for s in seed_list: p, r_ = hyb(ptf[idx], X[s][idx], tau[s]); PH[s], RT[s] = p, r_.mean(); PX[s] = X[s][idx]
    ptfa = ptf[idx].argmax(1); f_tf = mf1(yy, ptfa, classes); f_h = [mf1(yy, PH[s], classes) for s in seed_list]; f_x = [mf1(yy, PX[s], classes) for s in seed_list]; d = np.array(f_h) - f_tf; dx = np.array(f_h) - np.array(f_x)
    rng = np.random.default_rng(20261103); Fh = np.empty((B, S)); Fx = np.empty((B, S)); Ft = np.empty(B)
    for b in range(B):
        sel = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))]); ys = yy[sel]; Ft[b] = mf1(ys, ptfa[sel], classes)
        for k, s in enumerate(seed_list): Fh[b, k] = mf1(ys, PH[s][sel], classes); Fx[b, k] = mf1(ys, PX[s][sel], classes)
    dr = rng.integers(0, S, (B, S)); R = np.arange(B)[:, None]; dh = (Fh[R, dr] - Ft[:, None]).mean(1); dhx = (Fh[R, dr] - Fx[R, dr]).mean(1); ci = [float(np.percentile(dh, 2.5)), float(np.percentile(dh, 97.5))]; cix = [float(np.percentile(dhx, 2.5)), float(np.percentile(dhx, 97.5))]; md = float(d.mean())
    dec = "A+B: practically meaningful" if ci[0] > THRESH else "A: superior to TF-IDF (not above 0.02)" if ci[0] > 0 and md > 0 else "D: inferior" if ci[1] < 0 else "C: inconclusive"
    o = dict(tag=tag, n_docs=int(len(idx)), n_groups=int(len(ug)), seeds=seed_list, tfidf_macro_f1=f_tf, xlmr_macro_f1_mean=float(np.mean(f_x)), hybrid_macro_f1_mean=float(np.mean(f_h)), hybrid_macro_f1_per_seed=f_h, routed_share_mean=float(np.mean(list(RT.values()))), routed_share_per_seed=[RT[s] for s in seed_list], tau_per_seed=[tau[s] for s in seed_list],
             delta_vs_tfidf=md, ci95_vs_tfidf=ci, delta_vs_xlmr=float(dx.mean()), ci95_vs_xlmr=cix, seeds_hybrid_above_tfidf=int((d > 0).sum()), decision_vs_tfidf=dec,
             hybrid_acc_mean=float(np.mean([(PH[s] == yy).mean() for s in seed_list])), tfidf_acc=float((ptfa == yy).mean()), xlmr_acc_mean=float(np.mean([(PX[s] == yy).mean() for s in seed_list])))
    if curve:   # descriptive routing curve on this set (NOT used for selection)
        o["routing_curve_descriptive"] = [dict(tau=t_, routed_share=float(np.mean([hyb(ptf[idx], X[s][idx], t_)[1].mean() for s in seed_list])), macro_f1=float(np.mean([mf1(yy, hyb(ptf[idx], X[s][idx], t_)[0], classes) for s in seed_list]))) for t_ in GRID]
    return o
res = dict(tau_per_seed=tau, grid=GRID, validation_curve_real_only=valcurve, analyses={})
XT = {s: Z[s]["test"].argmax(1) for s in seeds}
for tag, sl in (("seeds0to4", [s for s in seeds if s <= 4]), ("seeds0to9", seeds)):
    if tag == "seeds0to9" and len(seeds) <= 5: continue
    res["analyses"][f"main_real_test_{tag}"] = run_set(yt, tf["test"], XT, gt, real, None, sl, "main real test", curve=(tag == "seeds0to4"))
    for SET in ("synthetic", "real"):
        D = O / f"unseen_{SET}"; m = json.loads((D / "meta.json").read_text()); y = np.array(m["y"]); ptf = np.load(D / "tfidf_probs.npy"); X = {s: np.load(D / f"xlmr_primary256_s{s}_logits.npy").argmax(1) for s in sl}; grp = np.array(m["group"]); cond = np.array(m["condition"]); L = m["labels"]
        if SET == "synthetic":
            cl = list(range(NC)); res["analyses"][f"unseen_synthetic_pooled_{tag}"] = run_set(y, ptf, X, grp, np.ones(len(y), bool), cl, sl, "synthetic pooled", curve=(tag == "seeds0to4"))
            for c in ("clean", "noisy", "long"): res["analyses"][f"unseen_synthetic_{c}_{tag}"] = run_set(y, ptf, X, grp, cond == c, cl, sl, f"synthetic {c}")
        else:
            cl = [L.index("food"), L.index("miscellaneous")]; res["analyses"][f"unseen_real_pt_{tag}"] = run_set(y, ptf, X, grp, np.isin(y, cl), cl, sl, "real Portuguese (2 classes)")
# ---- cost model (estimate) ----
c = json.loads((O / "cost_analysis.json").read_text())["per_model"]; tcpu, xcpu, xgpu = c["tfidf"]["cpu_end_to_end_ms"], c["xlmr"]["cpu_end_to_end_ms"], c["xlmr"]["gpu_end_to_end_ms"]
f = res["analyses"]["main_real_test_seeds0to4"]["routed_share_mean"]
res["cost_estimate"] = dict(note="ESTIMATE = TF-IDF latency + routed share x XLM-R latency (measured medians, batch 1, OCR excluded); assumes XLM-R runs only on routed documents", routed_share_main_test=f, tfidf_cpu_ms=tcpu, xlmr_cpu_ms=xcpu, xlmr_gpu_ms=xgpu,
    hybrid_cpu_ms=tcpu + f * xcpu, hybrid_gpu_xlmr_ms=tcpu + f * xgpu, hybrid_over_tfidf_cpu=(tcpu + f * xcpu) / tcpu, hybrid_over_xlmr_cpu=(tcpu + f * xcpu) / xcpu, worst_case_all_routed_cpu_ms=tcpu + xcpu)
(OUT / "hybrid_stats.json").write_text(json.dumps(res, indent=1, default=float))
for k, a in res["analyses"].items(): print(f"{k:40s} n={a['n_docs']:4d} TF {a['tfidf_macro_f1']:.3f} XL {a['xlmr_macro_f1_mean']:.3f} HY {a['hybrid_macro_f1_mean']:.3f} routed {100*a['routed_share_mean']:.0f}% | HY-TF {a['delta_vs_tfidf']:+.3f} [{a['ci95_vs_tfidf'][0]:+.3f},{a['ci95_vs_tfidf'][1]:+.3f}] HY-XL {a['delta_vs_xlmr']:+.3f} -> {a['decision_vs_tfidf'].split(':')[0]}")
print(json.dumps(res["cost_estimate"], indent=1))
