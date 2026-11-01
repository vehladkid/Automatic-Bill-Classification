"""Multiplicity check: 7 paired comparisons were reported (6 vs TF-IDF + attention vs BiLSTM). Recompute their paired group-bootstrap intervals at the Bonferroni-adjusted level (alpha 0.05/7 -> 99.29% interval).
Independent re-implementation (seed 2024, 10,000 resamples); reads saved logits + split labels only."""
import json, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
ROOT = Path(__file__).resolve().parents[3]; OUT = ROOT / "experiments/text_vs_neural/outputs"
rows = [json.loads(l) for l in open(ROOT / "datasets/processed/classification/test.jsonl")]; labels = sorted({json.loads(l)["label"] for f in ("train", "validation", "test") for l in open(ROOT / f"datasets/processed/classification/{f}.jsonl")}); l2i = {l: i for i, l in enumerate(labels)}
man = {json.loads(l)["document_id"]: json.loads(l) for l in open(ROOT / "datasets/metadata/manifest.jsonl")}
y = np.array([l2i[r["label"]] for r in rows]); real = np.array([not r["synthetic"] for r in rows]); g = np.array([man[r["document_id"]]["group_id"] for r in rows])[real]; y = y[real]
P = {k: np.load(OUT / f"{f}_logits.npz")["test"].argmax(1)[real] for k, f in dict(tfidf="tfidf_lr", bilstm="bilstm", bilstm_attn="bilstm_attn", lexrank="lexrank_50", xlmr="xlmr", layoutlmv3="layoutlmv3", fusion="fusion").items()}
ug = np.unique(g); mem = [np.where(g == u)[0] for u in ug]; rng = np.random.default_rng(2024); NB = 10000; A = {k: np.empty(NB) for k in P}; F = {k: np.empty(NB) for k in P}
for b in range(NB):
    s = np.concatenate([mem[i] for i in rng.integers(0, len(ug), len(ug))]); ys = y[s]; labs = sorted(set(ys.tolist()))
    for k in P: A[k][b] = (P[k][s] == ys).mean(); F[k][b] = f1_score(ys, P[k][s], labels=labs, average="macro", zero_division=0)
alpha = 0.05 / 7; out = {}
comps = [(k, "tfidf") for k in ("bilstm", "bilstm_attn", "lexrank", "xlmr", "layoutlmv3", "fusion")] + [("bilstm_attn", "bilstm")]
for a, b_ in comps:
    r = {}
    for nm, arr in (("macro_f1", F), ("accuracy", A)):
        d = arr[a] - arr[b_]; lo95, hi95 = np.percentile(d, [2.5, 97.5]); lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)]); r[nm] = dict(mean=float(d.mean()), ci95=[float(lo95), float(hi95)], ci_bonferroni=[float(lo), float(hi)], excludes_zero_95=bool(lo95 > 0 or hi95 < 0), excludes_zero_bonferroni=bool(lo > 0 or hi < 0))
    out[f"{a} vs {b_}"] = r; print(f"{a:12s} vs {b_:7s} macro-F1 {r['macro_f1']['mean']:+.3f} 95% [{r['macro_f1']['ci95'][0]:+.3f},{r['macro_f1']['ci95'][1]:+.3f}] excl0={r['macro_f1']['excludes_zero_95']} | Bonferroni [{r['macro_f1']['ci_bonferroni'][0]:+.3f},{r['macro_f1']['ci_bonferroni'][1]:+.3f}] excl0={r['macro_f1']['excludes_zero_bonferroni']} || acc {r['accuracy']['mean']:+.3f} excl0 95={r['accuracy']['excludes_zero_95']} bonf={r['accuracy']['excludes_zero_bonferroni']}")
json.dump(out, open(ROOT / "experiments/text_vs_neural/verification/multiplicity_check.json", "w"), indent=1)
