"""Statistics for the unseen sets (synthetic / real). Frozen predictions only. Hierarchical bootstrap (groups AND seeds resampled, B = 10,000), same decision rule as the primary analysis.
usage: python unseen_stats.py synthetic|real      -> outputs/unseen_<set>/stats.json"""
import sys, json
import numpy as np
from pathlib import Path
from scipy.stats import t as tdist
FIN = Path(__file__).resolve().parent; SET = sys.argv[1]; O = FIN / "outputs" / f"unseen_{SET}"; B = 10000; THRESH = 0.02; NC = 16
meta = json.loads((O / "meta.json").read_text()); y = np.array(meta["y"]); cond = np.array(meta["condition"]); grp = np.array(meta["group"]); LAB = meta["labels"]
tf = np.load(O / "tfidf_probs.npy").argmax(1); tf256 = np.load(O / "tfidf_head256_probs.npy").argmax(1)
seed_logits = {int(p.stem.split("_s")[-1].split("_")[0]): np.load(p).argmax(1) for p in sorted(O.glob("xlmr_primary256_s*_logits.npy"))}
def mf1(yy, pp, classes):
    cm = np.bincount(yy * NC + pp, minlength=NC * NC).reshape(NC, NC); f = []
    for c in classes:
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp; f.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(f))
def analyse(mask, seeds, classes, rng_seed=20261102):
    idx = np.where(mask)[0]; yy, gg = y[idx], grp[idx]; ug = np.unique(gg); members = [np.where(gg == u)[0] for u in ug]; S = len(seeds); P = {s: seed_logits[s][idx] for s in seeds}; ptf = tf[idx]; p256 = tf256[idx]
    f_tf = mf1(yy, ptf, classes); f_x = [mf1(yy, P[s], classes) for s in seeds]; a_tf = float((ptf == yy).mean()); a_x = [float((P[s] == yy).mean()) for s in seeds]; d = np.array(f_x) - f_tf
    rng = np.random.default_rng(rng_seed); F = np.empty((B, S)); Ft = np.empty(B); F256 = np.empty(B)
    for b in range(B):
        sel = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))]); ys = yy[sel]; Ft[b] = mf1(ys, ptf[sel], classes); F256[b] = mf1(ys, p256[sel], classes)
        for k, s in enumerate(seeds): F[b, k] = mf1(ys, P[s][sel], classes)
    draw = rng.integers(0, S, (B, S)); dh = (F[np.arange(B)[:, None], draw] - Ft[:, None]).mean(1); ci = [float(np.percentile(dh, 2.5)), float(np.percentile(dh, 97.5))]
    md = float(d.mean()); sd = float(d.std(ddof=1)) if S > 1 else None; tci = [md - tdist.ppf(.975, S - 1) * sd / np.sqrt(S), md + tdist.ppf(.975, S - 1) * sd / np.sqrt(S)] if S > 1 else None
    dec = "A+B: practically meaningful superiority" if ci[0] > THRESH else "A: superiority established (not above 0.02)" if ci[0] > 0 and md > 0 else "D: XLM-R inferior established" if ci[1] < 0 else "C: inconclusive"
    pcl = {LAB[c]: dict(support=int((yy == c).sum()), tfidf_f1=mf1(yy, ptf, [c]), xlmr_f1_mean=float(np.mean([mf1(yy, P[s], [c]) for s in seeds]))) for c in classes}
    return dict(n_docs=int(len(idx)), n_groups=int(len(ug)), seeds=seeds, classes=[LAB[c] for c in classes], tfidf=dict(macro_f1=f_tf, accuracy=a_tf), tfidf_head256=dict(macro_f1=mf1(yy, p256, classes), accuracy=float((p256 == yy).mean())),
                xlmr=dict(macro_f1_mean=float(np.mean(f_x)), macro_f1_per_seed=f_x, accuracy_mean=float(np.mean(a_x)), accuracy_per_seed=a_x), delta_macro_f1=md, delta_accuracy=float(np.mean(a_x) - a_tf), ci95_hierarchical=ci, ci95_seed_t=tci, seeds_positive=int((d > 0).sum()),
                tfidf_head256_minus_tfidf_full=float(mf1(yy, p256, classes) - f_tf), decision=dec, per_class=pcl)
out = dict(set=SET, meta=dict(n=meta["n"], ocr_empty=meta["ocr_empty"], over_256_pct_by_condition=meta["over_256_pct_by_condition"]), analyses={})
S5 = [s for s in sorted(seed_logits) if s <= 4]; S10 = sorted(seed_logits)
if SET == "synthetic":
    cl = [c for c in range(NC) if (y == c).any()]
    for tag, seeds in (("seeds0to4", S5), ("seeds0to9", S10)):
        if len(seeds) < 2 or (tag == "seeds0to9" and len(S10) <= 5): continue
        out["analyses"][f"pooled_{tag}"] = analyse(np.ones(len(y), bool), seeds, cl)
        for c in ("clean", "noisy", "long"): out["analyses"][f"{c}_{tag}"] = analyse(cond == c, seeds, cl)
else:
    cnt = {c: int((y == c).sum()) for c in range(NC)}; cl = [c for c in range(NC) if cnt[c] >= 10]; keep = np.isin(y, cl)
    for tag, seeds in (("seeds0to4", S5), ("seeds0to9", S10)):
        if len(seeds) < 2 or (tag == "seeds0to9" and len(S10) <= 5): continue
        out["analyses"][f"primary_classes_ge10_{tag}"] = analyse(keep, seeds, cl)
    out["other_class_docs_descriptive"] = {LAB[c]: dict(n=cnt[c], tfidf_correct=int(((tf == y) & (y == c)).sum()), xlmr_correct_by_seed={s: int(((seed_logits[s] == y) & (y == c)).sum()) for s in S5}) for c in range(NC) if 0 < cnt[c] < 10}
(O / "stats.json").write_text(json.dumps(out, indent=1, default=float))
for k, a in out["analyses"].items(): print(k, f"n={a['n_docs']} groups={a['n_groups']} TFIDF {a['tfidf']['macro_f1']:.3f} XLMR {a['xlmr']['macro_f1_mean']:.3f} delta {a['delta_macro_f1']:+.3f} CI {a['ci95_hierarchical'][0]:+.3f}..{a['ci95_hierarchical'][1]:+.3f} -> {a['decision']}")
