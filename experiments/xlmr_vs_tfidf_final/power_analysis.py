"""Phase 7 - can the current test set decide the question, and how big would a decisive test set be?  Two clearly labelled estimates, both with stated assumptions (nothing here is a measured result):
(1) ACCURACY - analytic McNemar sample size from the observed discordance rates;
(2) MACRO-F1 (primary metric) - resampling simulation: groups of the real test set are resampled to N documents (class mix preserved) and the paired bootstrap interval is computed on each simulated test set; power = share of simulated sets whose lower bound is > 0
    (and > 0.02). The simulation ASSUMES the observed paired predictions are representative, i.e. the true effect equals the observed seed-mean effect; because the observed effect may be inflated by chance this is optimistic.
(3) the number of test documents needed per class. usage: python power_analysis.py [variant=primary256]"""
import sys, json
import numpy as np
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xlmr_lib import *
V = sys.argv[1] if len(sys.argv) > 1 else "primary256"; tr, va, te = splits(); real = np.array([not r["synthetic"] for r in te]); y = np.array([r["y"] for r in te])[real]; g = np.array([r["group"] for r in te])[real]
tf = np.load(FIN / "outputs/tfidf/tfidf_probs.npz")["test"].argmax(1)[real]; seeds = sorted(int(p.name.split("_s")[-1]) for p in (FIN / "runs").glob(f"{V}_s*") if (p / "logits.npz").exists()); P = [np.load(FIN / "runs" / f"{V}_s{s}" / "logits.npz")["test"].argmax(1)[real] for s in seeds]
def mf1(y_, p_):
    sup = np.bincount(y_, minlength=NC); pc = np.bincount(p_, minlength=NC); tp = np.bincount(y_[y_ == p_], minlength=NC); m = sup > 0; return float((2 * tp[m] / (sup[m] + pc[m])).mean())
out = dict(variant=V, seeds=seeds, assumptions=[])
# (1) analytic McNemar
b01 = np.mean([((p == y) & (tf != y)).mean() for p in P]); b10 = np.mean([((p != y) & (tf == y)).mean() for p in P]); pd_ = b01 + b10; obs = b01 - b10; za, zb = norm.ppf(0.975), norm.ppf(0.8)
def n_mcnemar(delta, pd_): return int(np.ceil((za * np.sqrt(pd_) + zb * np.sqrt(pd_ - delta ** 2)) ** 2 / delta ** 2))
out["accuracy_mcnemar"] = dict(observed_discordant_rate=float(pd_), observed_accuracy_gain=float(obs), alpha=0.05, power=0.8, assumed_discordant_rate="the observed rate (mean over seeds)", n_documents_needed={f"accuracy_gain_{d}": n_mcnemar(d, pd_) for d in (0.01, 0.02, 0.03, float(round(obs, 3)) if obs > 0.003 else 0.015)}, method="Miettinen/Connor normal approximation for the McNemar test, independent documents (templates not clustered)")
# (2) simulation for macro-F1
rng = np.random.default_rng(777); ug = np.unique(g); members = [np.where(g == u)[0] for u in ug]; sizes = np.array([len(m) for m in members]); out["macro_f1_simulation"] = dict(note="assumes the observed seed-mean effect AND the observed document-level error structure are the truth (optimistic if inflated by chance); simulated test sets are built by drawing observed groups with replacement, each draw counted as a new independent group; B=500 paired-bootstrap replicates per simulated test set, 200 simulated test sets per N", rows=[])
base_d = float(np.mean([mf1(y, p) for p in P]) - mf1(y, tf)); out["macro_f1_simulation"]["observed_delta"] = base_d
for N in (386, 600, 800, 1200, 1600, 2400, 3200):
    hit0 = hit2 = 0; sims = 200
    for _ in range(sims):
        pick = []; gid = []; n = 0; k = 0
        while n < N:
            i = rng.integers(0, len(ug)); pick.append(members[i]); gid.append(np.full(len(members[i]), k)); n += sizes[i]; k += 1      # every draw is a NEW group (a new independent template), even if the same observed group is drawn again
        sel = np.concatenate(pick)[:N]; gs = np.concatenate(gid)[:N]; ys = y[sel]; pt = tf[sel]; ps = P[rng.integers(0, len(P))][sel]; ug2 = np.unique(gs); mem2 = [np.where(gs == u)[0] for u in ug2]; d = np.empty(500)
        for b in range(500):
            ss = np.concatenate([mem2[q] for q in rng.integers(0, len(ug2), len(ug2))]); d[b] = mf1(ys[ss], ps[ss]) - mf1(ys[ss], pt[ss])
        lo = np.percentile(d, 2.5); hit0 += lo > 0; hit2 += lo > 0.02
    out["macro_f1_simulation"]["rows"].append(dict(n_documents=N, power_lower_ci_above_0=hit0 / sims, power_lower_ci_above_0_02=hit2 / sims)); print(N, hit0 / sims, hit2 / sims, flush=True)
# (3) per-class counts
cnt = np.bincount(y, minlength=NC); prev = cnt / cnt.sum(); rare = [(LABELS[c], int(cnt[c])) for c in range(NC) if 0 < cnt[c] < 15]
out["per_class"] = dict(current_real_test_counts={LABELS[c]: int(cnt[c]) for c in range(NC)}, classes_below_15=rare, documents_needed_for_30_per_class_at_current_mix=int(np.ceil(30 / prev[prev > 0].min())), note="if the test set were enlarged by sampling at the current class mix; a class-stratified (oversampled) collection of the rare classes needs far fewer")
(FIN / "outputs" / "power_analysis.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps({k: out[k] for k in ("accuracy_mcnemar", "per_class")}, indent=1, default=float)[:1800])
