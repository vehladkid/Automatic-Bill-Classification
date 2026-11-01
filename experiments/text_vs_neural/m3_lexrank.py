"""Model 3 (exploratory) - LexRank SUMMARIZATION + classifier.   LexRank is an extractive summarizer, NOT an OCR engine and NOT a classifier:
it ranks sentence-like units by graph centrality. Here the units are the OCR LINES of a bill (words grouped by their y-bucket in reading order, scripts/taskA/common_a.words),
because bills have no real sentences. The top-k fraction of lines (kept in original order) is the 'summary' and is fed to the SAME downstream classifier as Model 1
(TF-IDF char2-5 + word1-2 + LogisticRegression, C = project value, class-balanced). Comparison = same classifier on (a) the original flat OCR text, (b) all lines re-joined (LexRank fraction 1.0),
(c) LexRank summaries keeping 75 / 50 / 25 % of the lines.
LexRank: sentence TF-IDF (unigram, IDF fitted on TRAIN lines only) -> cosine similarity -> edges where sim > 0.10 -> row-normalised -> PageRank power iteration (damping 0.85, tol 1e-6)."""
import sys, time, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "taskA"))
from exp_common import *
import t1_text_baseline as T1
from common_a import words as row_words
from sklearn.feature_extraction.text import TfidfVectorizer

FRACS = (1.0, 0.75, 0.5, 0.25); THR, DAMP = 0.10, 0.85
tr, va, te = splits(); C = json.loads((ROOT / "reports/taskA/text_baseline.json").read_text())["C"]

def lines_of(row):
    """OCR lines in reading order. Falls back to 8-word chunks of the flat OCR text when the document has no word boxes."""
    ws = row_words(row)
    if ws:
        hh = max(row["h"] or 1, 1); bucket = max(hh // 120, 8); out = {}
        for t, b in ws: out.setdefault(b[1] // bucket, []).append(t)
        L = [" ".join(v) for _, v in sorted(out.items())]
        if len(L) >= 2: return L, True
    w = row["text"].split(); return [" ".join(w[i:i + 8]) for i in range(0, len(w), 8)] or [row["text"]], False

for rows in (tr, va, te):
    for r in rows: r["_lines"], r["_has_boxes"] = lines_of(r)
print("documents with real OCR line structure: train %.1f%%  val %.1f%%  test %.1f%%" % tuple(100 * np.mean([r["_has_boxes"] for r in rows]) for rows in (tr, va, te)))
print("median lines per doc: %d (p10 %d, p90 %d)" % (np.median([len(r["_lines"]) for r in tr]), np.percentile([len(r["_lines"]) for r in tr], 10), np.percentile([len(r["_lines"]) for r in tr], 90)))
sv = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=1).fit([l for r in tr for l in r["_lines"]])

def lexrank_scores(units):
    n = len(units)
    if n == 1: return np.ones(1)
    X = sv.transform(units); S = (X @ X.T).toarray(); A = (S > THR).astype(float); np.fill_diagonal(A, 0)
    deg = A.sum(1); P = np.where(deg[:, None] > 0, A / np.maximum(deg[:, None], 1), 1.0 / n)          # isolated unit -> uniform row
    s = np.full(n, 1.0 / n)
    for _ in range(200):
        s2 = (1 - DAMP) / n + DAMP * (P.T @ s)
        if np.abs(s2 - s).sum() < 1e-6: s = s2; break
        s = s2
    return s

def summarize(row, frac):
    L = row["_lines"]; k = max(1, int(round(frac * len(L))))
    if k >= len(L): return " ".join(L)
    sc = lexrank_scores(L); keep = sorted(np.argsort(-sc)[:k]); return " ".join(L[i] for i in keep)

def build(rows, frac): return [dict(r, text=summarize(r, frac)) for r in rows]
def words(rows): return float(np.mean([len(r["text"].split()) for r in rows]))
lg = lambda p: np.log(np.clip(p, 1e-6, 1)); results = []; full_words = words(te)
# (a) flat OCR text, identical to Model 1 (re-used so the table has one source of truth)
m1 = json.loads((OUT / "m1_tfidf.json").read_text()); results.append(dict(config="Full OCR text (Model 1, flat text)", frac=None, words_per_doc=full_words, reduction_pct=0.0, accuracy_real=m1["calibrated"]["real"]["accuracy"],
                                                                      macro_f1_real=m1["calibrated"]["real"]["macro_f1"], train_seconds=m1["train_seconds"], summarize_ms=0.0, infer_ms=m1["latency_cpu"]["median_ms"], note="same text as every other model"))
logits = {}
for f in FRACS:
    t0 = time.perf_counter(); trs = build(tr, f); t_sum = time.perf_counter() - t0; vas, tes = build(va, f), build(te, f)
    t0 = time.perf_counter(); vm = T1.fit(trs, C); t_fit = time.perf_counter() - t0
    pv, pt = T1.proba(vm, vas), T1.proba(vm, tes); name = f"LexRank {int(f * 100)}% of lines" if f < 1 else "All OCR lines re-joined (LexRank 100%)"
    res = evaluate_model(name, lg(pv), lg(pt), va, te); v_, m_ = vm; sample = [r["text"] for r in tes[:40]]; i = [0]
    def one(): t = sample[i[0] % 40]; i[0] += 1; m_.predict_proba(v_.transform([t]))
    lat = time_fn(one, 5, 80)
    rows40 = te[:40]; j = [0]
    def one_sum(): summarize(rows40[j[0] % 40], f); j[0] += 1
    lat_s = time_fn(one_sum, 5, 80) if f < 1 else dict(median_ms=0.0)
    logits[f] = (lg(pv), lg(pt)); save_logits(f"lexrank_{int(f * 100)}", lg(pv), lg(pt))
    results.append(dict(config=name, frac=f, words_per_doc=words(tes), reduction_pct=100 * (1 - words(tes) / full_words), accuracy_real=res["calibrated"]["real"]["accuracy"], macro_f1_real=res["calibrated"]["real"]["macro_f1"],
                        accuracy_all=res["calibrated"]["all"]["accuracy"], macro_f1_all=res["calibrated"]["all"]["macro_f1"], ece_real=res["calibrated"]["real"]["ece"], nll_real=res["calibrated"]["real"]["nll"],
                        train_seconds=t_fit + t_sum, summarize_ms=lat_s["median_ms"], infer_ms=lat["median_ms"], full=res))
    print(f"{name}: words/doc {results[-1]['words_per_doc']:.1f} (-{results[-1]['reduction_pct']:.1f}%)  real acc {results[-1]['accuracy_real']:.4f} macro-F1 {results[-1]['macro_f1_real']:.4f}  fit {t_fit:.0f}s  LexRank {lat_s['median_ms']:.2f} ms/doc", flush=True)
# the headline LexRank row for Table B = the 50% summary (pre-declared before seeing results: 'about half of the text')
best = [r for r in results if r["frac"] == 0.5][0]["full"]
best.update(config=dict(units="OCR lines (y-buckets)", lexrank=dict(threshold=THR, damping=DAMP, tol=1e-6, max_iter=200, idf="train lines"), keep_fraction=0.5, classifier="TF-IDF + LR (same as Model 1)", C=C),
            device="CPU", train_seconds=[r for r in results if r["frac"] == 0.5][0]["train_seconds"], latency_cpu=dict(median_ms=best_ms if (best_ms := [r for r in results if r["frac"] == 0.5][0]["infer_ms"] + [r for r in results if r["frac"] == 0.5][0]["summarize_ms"]) else 0),
            model_bytes=None, peak_mem_mb=None)
save_json("m3_lexrank.json", dict(best=best, sweep=[{k: v for k, v in r.items() if k != "full"} for r in results], lines_coverage=dict(zip(("train", "val", "test"), [float(np.mean([r["_has_boxes"] for r in rows])) for rows in (tr, va, te)]))))
