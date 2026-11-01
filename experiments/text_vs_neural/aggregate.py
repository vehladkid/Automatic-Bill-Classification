"""Builds every comparison table (CSV + one results.json) and every graph from the saved experiment outputs. Needs matplotlib (not in the project venv):
PYTHONPATH=<dir containing matplotlib> ../../.venv/bin/python aggregate.py          Nothing here trains or measures anything - it only reads outputs/*.json and outputs/*_logits.npz."""
import sys, json, csv
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = EXP / "figures"; FIG.mkdir(exist_ok=True)
J = lambda n: json.loads((OUT / n).read_text()) if (OUT / n).exists() else None
M = [("tfidf", "TF-IDF + Logistic Regression", J("m1_tfidf.json"), "tfidf_lr", "word 1-2 + char 2-5 TF-IDF", "text only"),
     ("bilstm", "BiLSTM", J("m_bilstm.json"), "bilstm", "word embeddings (128-d, trained from scratch)", "text only"),
     ("bilstm_attn", "BiLSTM + attention", J("m_bilstm_attn.json"), "bilstm_attn", "word embeddings (128-d, trained from scratch)", "text only"),
     ("lexrank", "LexRank 50% summary + TF-IDF/LR", (J("m3_lexrank.json") or {}).get("best"), "lexrank_50", "TF-IDF of the 50% most central OCR lines", "text only"),
     ("xlmr", "XLM-RoBERTa-base (fine-tuned)", J("m5_xlmr.json"), "xlmr", "SentencePiece subwords, pretrained contextual", "text only"),
     ("layoutlmv3", "LayoutLMv3-base (production)", J("m6_layoutlmv3.json"), "layoutlmv3", "text + 2-D boxes + page image", "YES: layout + image"),
     ("fusion", "Calibrated fusion (production, 3 models)", J("m7_fusion.json"), "fusion", "TF-IDF + ConvNeXt image + LayoutLMv3", "YES: layout + image")]
M = [m for m in M if m[2] is not None]; tr, va, te = splits(); real = np.array([not r["synthetic"] for r in te])
lg = {m[0]: np.load(OUT / f"{m[3]}_logits.npz")["test"] for m in M}
boot = paired_bootstrap(te, lg, "tfidf") if "tfidf" in lg else {}
boot_rnn = paired_bootstrap(te, {k: lg[k] for k in ("bilstm", "bilstm_attn")}, "bilstm") if "bilstm" in lg and "bilstm_attn" in lg else {}      # does attention help the plain BiLSTM?
f = lambda x, d=3: "N/A" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))
def lat(m, key):
    v = m[2].get(key); return None if not v else v["median_ms"]

# ---------------- Table B: classification ----------------
B = []
for k, name, r, _, rep, lay in M:
    c = r["calibrated"]["real"]; raw = r["raw"]["real"]; a = r["calibrated"]["all"]
    B.append(dict(key=k, model=name, text_representation=rep, uses_layout_or_image=lay, accuracy=c["accuracy"], macro_precision=c["macro_precision"], macro_recall=c["macro_recall"], macro_f1=c["macro_f1"], weighted_f1=c["weighted_f1"],
                  ece_raw=raw["ece"], ece_calibrated=c["ece"], nll_raw=raw["nll"], nll_calibrated=c["nll"], brier_calibrated=c["brier"], accuracy_ci95=boot.get(k, {}).get("accuracy_ci95"), macro_f1_ci95=boot.get(k, {}).get("macro_f1_ci95"),
                  delta_macro_f1_vs_tfidf=boot.get(k, {}).get("delta_macro_f1"), delta_accuracy_vs_tfidf=boot.get(k, {}).get("delta_accuracy"), accuracy_all_test=a["accuracy"], macro_f1_all_test=a["macro_f1"],
                  latency_cpu_ms=lat((k, name, r), "latency_cpu"), latency_gpu_ms=lat((k, name, r), "latency_gpu"), model_mb=(r.get("model_bytes") or 0) / 2**20 or None, temperature=r["temperature"]))
with open(EXP / "table_B_classification.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(B[0].keys())); w.writeheader(); [w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()}) for row in B]

# ---------------- Table C: resources ----------------
C = []
for k, name, r, _, _, _ in M:
    C.append(dict(model=name, training_seconds=(r.get("train_seconds") if not isinstance(r.get("train_seconds"), list) else None), device=r.get("device", "CPU"), peak_memory_mb=r.get("peak_mem_mb"), latency_cpu_ms=lat((k, name, r), "latency_cpu"),
                  latency_gpu_ms=lat((k, name, r), "latency_gpu"), model_mb=(r.get("model_bytes") or 0) / 2**20 or None, completed="yes"))
with open(EXP / "table_C_resources.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(C[0].keys())); w.writeheader(); w.writerows(C)

# ---------------- Table D: per class (real test) ----------------
classes = sorted({c for _, _, r, _, _, _ in M for c in r["calibrated"]["real"]["per_class"]})
D = []
for cl in classes:
    for k, name, r, _, _, _ in M:
        p = r["calibrated"]["real"]["per_class"].get(cl); D.append(dict(category=cl, model=name, precision=p["precision"] if p else None, recall=p["recall"] if p else None, f1=p["f1"] if p else None, support=p["support"] if p else None))
with open(EXP / "table_D_per_class.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(D[0].keys())); w.writeheader(); w.writerows(D)

# ---------------- Table E: LexRank ----------------
E = (J("m3_lexrank.json") or {}).get("sweep", [])
if E:
    with open(EXP / "table_E_lexrank.csv", "w", newline="") as fh:
        cols = list(dict.fromkeys(k for r in E for k in r)); w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(E)

# ---------------- Table A: OCR ----------------
A = []
for k, label in (("rapidocr", "RapidOCR PP-OCRv4 recogniser (production)"), ("crnn", "CRNN (CNN + BiLSTM + CTC), trained here"), ("trocr", "TrOCR-small-printed (Transformer)")):
    o = J(f"ocr_{k}.json")
    if o: A.append(dict(model=label, architecture=o["info"]["arch"], cer_strict_all=o["all"]["cer"], cer_normalised_all=o["all"]["cer_normalised"], wer_strict_all=o["all"]["wer"], wer_casefold_all=o["all"]["wer_casefold"],
                        cer_norm_sroie=o["sroie"]["cer_normalised"], cer_norm_funsd=o["funsd"]["cer_normalised"], latency_ms_per_page=o["latency_ms_per_page_median"], pages_per_second=o["pages_per_second"],
                        peak_memory_mb=o["info"].get("peak_mem_mb"), model_mb=o["info"]["model_bytes"] / 2**20, train_seconds=o["info"].get("train_seconds"), note=o["info"].get("contamination_note", "")))
if A:
    with open(EXP / "table_A_ocr.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(A[0].keys())); w.writeheader(); w.writerows(A)

save_json("results.json", dict(env=env_info(), table_A_ocr=A, table_B=B, table_C=C, table_D=D, table_E_lexrank=E, paired_bootstrap=boot, paired_bootstrap_attention_vs_bilstm=boot_rnn,
                              nn_seed_variation={k: r.get("seeds_test_real") for k, _, r, _, _, _ in M if r.get("seeds_test_real")}, test_set="project test split, REAL documents n=%d (groups bootstrapped); calibration temperature fitted on validation only" % real.sum()))

# ---------------- graphs ----------------
COL = {"tfidf": "#1f77b4", "bilstm": "#ff7f0e", "bilstm_attn": "#2ca02c", "lexrank": "#9467bd", "xlmr": "#d62728", "layoutlmv3": "#8c564b", "fusion": "#7f7f7f"}
names = [b["model"].replace(" (fine-tuned)", "").replace(" (production)", "").replace("TF-IDF + Logistic Regression", "TF-IDF + LR").replace("LexRank 50% summary + TF-IDF/LR", "LexRank 50% + TF-IDF/LR").replace("Calibrated fusion (production, 3 models)", "Fusion (production)") for b in B]
TITLE = "project test split, real documents (n=386)"
def bar(fn, vals, ylabel, title, errs=None, log=False, fmt="{:.3f}", note=None):
    fig, ax = plt.subplots(figsize=(8.2, 4.2)); x = np.arange(len(vals)); v = [0 if y is None else y for y in vals]
    ax.bar(x, v, color=[COL[b["key"]] for b in B], yerr=errs, capsize=3); ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8); ax.set_ylabel(ylabel); ax.set_title(title + "\n" + (note or TITLE), fontsize=9)
    if log: ax.set_yscale("log")
    for xi, y in zip(x, vals): ax.text(xi, (y or 0) * (1.08 if log else 1) + (0 if log else max(v) * 0.01), "N/A" if y is None else fmt.format(y), ha="center", va="bottom", fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / fn, dpi=170); plt.close()
ci_err = lambda key, vals: None if not all(boot.get(b["key"]) for b in B) else np.array([[vals[i] - boot[b["key"]][key][0], boot[b["key"]][key][1] - vals[i]] for i, b in enumerate(B)]).T
mf = [b["macro_f1"] for b in B]; ac = [b["accuracy"] for b in B]
bar("fig1_macro_f1.png", mf, "macro-F1", "Macro-F1 by model (bars: 95% group-bootstrap CI)", ci_err("macro_f1_ci95", mf))
bar("fig2_accuracy.png", ac, "accuracy", "Accuracy by model (bars: 95% group-bootstrap CI)", ci_err("accuracy_ci95", ac))
fig, ax = plt.subplots(figsize=(8.2, 4.2)); x = np.arange(len(B)); ax.bar(x - 0.2, [b["ece_raw"] for b in B], 0.4, label="raw", color="#bbbbbb"); ax.bar(x + 0.2, [b["ece_calibrated"] for b in B], 0.4, label="temperature-scaled (fit on validation)", color=[COL[b["key"]] for b in B])
for xi, b in zip(x, B): ax.text(xi - 0.2, b["ece_raw"], f"{b['ece_raw']:.3f}", ha="center", va="bottom", fontsize=7); ax.text(xi + 0.2, b["ece_calibrated"], f"{b['ece_calibrated']:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8); ax.set_ylabel("ECE (10 bins, lower is better)"); ax.set_title("Expected calibration error by model\n" + TITLE, fontsize=9); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / "fig3_ece.png", dpi=170); plt.close()
fig, ax = plt.subplots(figsize=(8.2, 4.2)); w_ = 0.4
ax.bar(x - 0.2, [b["latency_cpu_ms"] or 0 for b in B], w_, label="CPU (4 threads)", color="#4c78a8"); ax.bar(x + 0.2, [b["latency_gpu_ms"] or 0 for b in B], w_, label="GPU (RTX 4050)", color="#f58518")
for xi, b in zip(x, B):
    for off, key in ((-0.2, "latency_cpu_ms"), (0.2, "latency_gpu_ms")): ax.text(xi + off, (b[key] or 0.3) * 1.1, "N/A" if b[key] is None else f"{b[key]:.1f}", ha="center", va="bottom", fontsize=7)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8); ax.set_ylabel("median latency per document (ms, log scale)"); ax.set_title("Inference latency (batch 1, after warm-up; classifier only, OCR not included)", fontsize=9); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / "fig4_latency.png", dpi=170); plt.close()
bar("fig5_training_time.png", [c["training_seconds"] for c in C], "training time (s, log scale)", "Training duration", log=True, fmt="{:.0f}s", note="CPU for TF-IDF/LexRank, GPU for the neural models")
bar("fig6_model_size.png", [b["model_mb"] for b in B], "model size on disk (MB, log scale)", "Model size", log=True, fmt="{:.1f}", note="XLM-R: pretrained checkpoint; LayoutLMv3: production checkpoint")
show = [c for c in classes if max(r["calibrated"]["real"]["per_class"].get(c, {"support": 0})["support"] for _, _, r, _, _, _ in M) >= 8]
fig, ax = plt.subplots(figsize=(10, 4.4)); wd = 0.8 / len(B)
for j, (k, name, r, _, _, _) in enumerate(M): ax.bar(np.arange(len(show)) + j * wd, [r["calibrated"]["real"]["per_class"].get(c, {"f1": 0})["f1"] for c in show], wd, label=names[j], color=COL[k])
ax.set_xticks(np.arange(len(show)) + wd * (len(B) - 1) / 2); ax.set_xticklabels([f"{c}\n(n={max(r['calibrated']['real']['per_class'].get(c, {'support': 0})['support'] for _, _, r, _, _, _ in M)})" for c in show], fontsize=8)
ax.set_ylabel("F1"); ax.set_title("Per-class F1 (categories with >= 8 real test documents; full table: table_D_per_class.csv)\n" + TITLE, fontsize=9); ax.legend(fontsize=7, ncol=3); plt.tight_layout(); plt.savefig(FIG / "fig7_per_class_f1.png", dpi=170); plt.close()
if A:
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8)); lab = [a["model"].split(" (")[0].replace(" recogniser", "").replace("PP-OCRv4", "PP-OCRv4") for a in A]; xx = np.arange(len(A))
    ax[0].bar(xx - 0.2, [a["cer_strict_all"] for a in A], 0.4, label="CER strict", color="#bbbbbb"); ax[0].bar(xx + 0.2, [a["cer_normalised_all"] for a in A], 0.4, label="CER case/space-insensitive", color="#1f77b4")
    ax[1].bar(xx - 0.2, [a["wer_strict_all"] for a in A], 0.4, label="WER strict", color="#bbbbbb"); ax[1].bar(xx + 0.2, [a["wer_casefold_all"] for a in A], 0.4, label="WER case-insensitive", color="#d62728")
    for a_, ttl in ((ax[0], "Character error rate"), (ax[1], "Word error rate")):
        a_.set_xticks(xx); a_.set_xticklabels(lab, rotation=15, ha="right", fontsize=7); a_.set_title(ttl + "\nSROIE+FUNSD GT crops, test-split documents", fontsize=9); a_.legend(fontsize=7); a_.set_ylabel("error rate (lower is better)")
    plt.tight_layout(); plt.savefig(FIG / "fig8_ocr_cer_wer.png", dpi=170); plt.close()
if E:
    fig, ax = plt.subplots(figsize=(6.4, 4)); xs = [r["reduction_pct"] for r in E]; ax.plot(xs, [r["accuracy_real"] for r in E], "o-", label="accuracy"); ax.plot(xs, [r["macro_f1_real"] for r in E], "s-", label="macro-F1")
    for r in E: ax.annotate(f"{r['macro_f1_real']:.3f}", (r["reduction_pct"], r["macro_f1_real"]), textcoords="offset points", xytext=(0, -12), fontsize=7, ha="center")
    ax.set_xlabel("text removed by LexRank summarisation (% of words)"); ax.set_ylabel("score (real test documents)"); ax.set_title("LexRank summary + TF-IDF/LR: quality vs text reduction", fontsize=9); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / "fig9_lexrank_tradeoff.png", dpi=170); plt.close()
print("tables + figures written to", EXP); print("macro-F1 / accuracy (real test):", [(b["key"], round(b["macro_f1"], 3), round(b["accuracy"], 3)) for b in B])
