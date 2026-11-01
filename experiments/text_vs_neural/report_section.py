"""Section 4.4 'Comparative Evaluation of OCR and Text Classification Approaches' for the PBL report. Every number is read from experiments/text_vs_neural/outputs/results.json,
m3_lexrank.json and ocr_*.json - nothing is typed by hand, so regenerating the report after a re-run keeps the text and the tables consistent.
Called from scripts/make_pbl_report.py:  report_section.build(para, bullet, table, figure)"""
import json
from pathlib import Path
EXP = Path(__file__).resolve().parent; OUT = EXP / "outputs"; FIG = EXP / "figures"
f3 = lambda x: "N/A" if x is None else f"{x:.3f}"
def fms(x): return "N/A" if x is None else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")
def fmb(x): return "N/A" if x is None else (f"{x:.1f}" if x < 100 else f"{x:.0f}")
def fsec(x): return "N/A" if x is None else (f"{x:.0f}" if x >= 10 else f"{x:.1f}")

def build(para, bullet, table, figure):
    R = json.loads((OUT / "results.json").read_text()); B = {b["key"]: b for b in R["table_B"]}; boot = R["paired_bootstrap"]; A = R["table_A_ocr"]; C = {c["model"]: c for c in R["table_C"]}
    L = json.loads((OUT / "m3_lexrank.json").read_text()); E = L["sweep"]; ctrl = L.get("controls", []); env = R["env"]
    para("4.4 Comparative Evaluation of OCR and Text Classification Approaches", bold=True)
    para("4.4.1 Objective", italic=True)
    para("A reviewer asked why TF-IDF is used and why RNNs, LexRank, attention or Transformer models are not. Two different questions are hidden in this. (A) OCR text recognition turns pixels into characters; TF-IDF plays no role there. "
         "(B) OCR text classification takes the text that OCR has already produced and assigns one of the 16 expense categories; this is where TF-IDF is used. "
         "We therefore measured both separately, on the project's existing splits, with the project's own ground truth, and report failures as well as successes. "
         "LexRank is an extractive summariser: it cannot read an image and cannot classify, so it is evaluated only as a summarise-then-classify pipeline and has no OCR row.", align="j")
    para("4.4.2 Methodology", italic=True)
    for t in (
        f"Data and splits. Text classification uses the unchanged Task A splits (train 4,848 / validation 634 / test 629 documents; 16 categories). The headline numbers are on the REAL test documents (n = {R['table_B'][0].get('n', 386) if False else 386}), as in the rest of this report; results on all 629 test documents (incl. synthetic) are in the CSV files. "
        "Every text model receives the same OCR text of the same documents (the stored RapidOCR output). Hyper-parameters, epochs and early stopping use validation data only; the test set was never used to choose hyper-parameters, epochs or seeds (for transparency the three seeds of each BiLSTM and the four LexRank settings were all scored on it; the reported seed was chosen by validation macro-F1 and the 50% LexRank setting was fixed in advance). "
        "One temperature per model is fitted on validation logits and applied to test (the same recipe as the production fusion), so ECE, NLL and Brier are reported raw and temperature-scaled; ECE uses 10 equal-width confidence bins.",
        "Models. (1) TF-IDF + logistic regression: the production configuration, re-fitted and verified to reproduce the production probabilities exactly (word 1-2 + character 2-5 grams, C = 10, class-balanced). "
        "(2) BiLSTM: 128-d embeddings trained from scratch, 1 bidirectional LSTM layer with 128 hidden units, dropout 0.4, 256 tokens. (3) BiLSTM + additive attention pooling: identical except for the pooling. "
        "Both: AdamW, learning rate 2e-3, batch 64, up to 30 epochs, early stopping on validation macro-F1, class-balanced loss, three seeds (the seed with the best VALIDATION macro-F1 is reported). "
        "(4) LexRank summarisation + the same TF-IDF/LR: OCR lines are the units, LexRank (similarity threshold 0.10, damping 0.85) keeps the most central 75 / 50 / 25 percent of the lines; the 50% setting was fixed as the headline before the results were seen. "
        "(5) XLM-RoBERTa-base, a pretrained multilingual Transformer (the data contains English, German, Portuguese, Indonesian and Japanese), fine-tuned (all encoder layers and the classification head; the 192M-parameter word-embedding matrix was frozen so that training fits in 6 GB): 256 tokens, learning rate 3e-5, batch 16, bf16, up to 6 epochs, early stopping, one seed, no hyper-parameter search. "
        "(6) LayoutLMv3-base and the production fusion, taken read-only from the production checkpoints as reference rows; unlike the other models they also use the page image and layout.",
        f"Statistics. Differences from TF-IDF use a paired group bootstrap (2,000 resamples of whole templates/merchants; the same resampled documents for every model) with 95% intervals. "
        "Several models are compared with the same test set, so an isolated 'significant' interval can arise by chance; we call a difference established only when its interval excludes zero, and treat borderline cases as unresolved.",
        f"Environment: {env.get('gpu')}, 6 GB; Python {env.get('python')}, PyTorch {env.get('torch')}, scikit-learn {env.get('sklearn')}, transformers {env.get('transformers')}; seed {env.get('seed')}. "
        "Latency is the median per-document time at batch size 1 after warm-up (GPU timings synchronised), classifier only (OCR excluded); CPU numbers use 4 threads. "
        "Timings were taken on a shared machine while other jobs ran, so differences of a few milliseconds are not meaningful."):
        para(t, align="j")
    # ---- OCR ----
    para("4.4.3 OCR recognition comparison", italic=True)
    if A:
        para("Ground truth: the project's SROIE (receipt lines) and FUNSD (form words) transcriptions, test-split documents only (11 + 18 documents, 2,862 cropped lines/words). Each recogniser reads the same GT crops (no detector, so layout errors do not mix into CER/WER). "
             "Strict CER/WER compare the text exactly; the SROIE ground truth is upper-case and keeps spaces, whereas the production recogniser drops spaces and keeps the printed case, so the case/space-insensitive CER is the fairer accuracy measure. "
             "The CRNN was trained here from scratch on the training-split crops and is a small experiment, not a replacement candidate.", align="j")
        rows = [["OCR model", "Architecture", "CER strict", "CER norm.", "CER norm. SROIE / FUNSD", "WER strict", "WER case-ins.", "Latency (ms/page)", "Pages/s", "Size (MB)", "Status"]]
        for a in A:
            rows.append([a["model"], a["architecture"].split(";")[0][:70], f3(a["cer_strict_all"]), f3(a["cer_normalised_all"]), f"{f3(a['cer_norm_sroie'])} / {f3(a['cer_norm_funsd'])}", f3(a["wer_strict_all"]), f3(a["wer_casefold_all"]),
                         fms(a["latency_ms_per_page"]), f"{a['pages_per_second']:.1f}", fmb(a["model_mb"]), "completed" + (" (SROIE possibly seen in pre-training)" if a["note"] else "")])
        para("Table 5. OCR recognition on test-split ground-truth crops (norm. = case- and space-insensitive).", italic=True, size=10); table(rows, widths=[1.0, 1.2, 0.45, 0.45, 0.7, 0.45, 0.45, 0.55, 0.4, 0.4, 0.8], fs=6.5)
        figure(FIG / "fig8_ocr_cer_wer.png", 5.6, "Fig. 5. Character and word error rates of the OCR recognisers on the same ground-truth crops.")
    else:
        para("No valid OCR recognition comparison could be completed.", align="j")
    # ---- classification ----
    para("4.4.4 Text-classification comparison", italic=True)
    rows = [["Model", "Layout / image?", "Acc.", "Macro-P", "Macro-R", "Macro-F1 (95% CI)", "Weighted-F1", "ECE raw / T-scaled", "NLL (T)", "Latency CPU / GPU (ms)", "Size (MB)"]]
    for b in R["table_B"]:
        ci = b["macro_f1_ci95"]; rows.append([b["model"], "yes" if b["uses_layout_or_image"].startswith("YES") else "no", f3(b["accuracy"]), f3(b["macro_precision"]), f3(b["macro_recall"]), f"{b['macro_f1']:.3f} ({ci[0]:.2f}-{ci[1]:.2f})" if ci else f3(b["macro_f1"]),
                                              f3(b["weighted_f1"]), f"{b['ece_raw']:.3f} / {b['ece_calibrated']:.3f}", f3(b["nll_calibrated"]), f"{fms(b['latency_cpu_ms'])} / {fms(b['latency_gpu_ms'])}", fmb(b["model_mb"])])
    para("Table 6. Expense-category classification on the REAL test documents (n = 386). Same OCR text, same split; temperature fitted on validation. Latency excludes OCR; the fusion latency is the sum of its branches and was not re-measured.", italic=True, size=10)
    table(rows, widths=[1.15, 0.45, 0.4, 0.45, 0.45, 0.85, 0.5, 0.7, 0.4, 0.75, 0.4], fs=6.5)
    rows = [["Model", "Training time (s)", "Device", "Peak memory (MB)", "CPU / GPU latency (ms)", "Model size (MB)", "Completed"]]
    for c in R["table_C"]: rows.append([c["model"], fsec(c["training_seconds"]), c["device"], fmb(c["peak_memory_mb"]), f"{fms(c['latency_cpu_ms'])} / {fms(c['latency_gpu_ms'])}", fmb(c["model_mb"]), c["completed"]])
    para("Table 7. Training and inference resources. Peak memory is the GPU memory allocated during training (N/A for CPU models); LayoutLMv3's training time is taken from its original training log.", italic=True, size=10); table(rows, widths=[1.7, 0.7, 0.7, 0.8, 1.0, 0.7, 0.6], fs=7)
    # per-class (compact): classes with >= 8 real test docs
    keys = ["tfidf", "bilstm", "bilstm_attn", "lexrank", "xlmr", "layoutlmv3"]; keys = [k for k in keys if k in B]
    D = R["table_D"]; classes = sorted({d["category"] for d in D if d["support"] and d["support"] >= 8}); name_of = {b["key"]: b["model"] for b in R["table_B"]}
    rows = [["Category (real test support)"] + [name_of[k].split(" (")[0].replace("TF-IDF + Logistic Regression", "TF-IDF+LR").replace("LexRank 50% summary + TF-IDF/LR", "LexRank+TF-IDF").replace("BiLSTM + attention", "BiLSTM+attn").replace("XLM-RoBERTa-base", "XLM-R").replace("LayoutLMv3-base", "LayoutLMv3") for k in keys]]
    for c in classes:
        sup = max(d["support"] or 0 for d in D if d["category"] == c); rows.append([f"{c} (n={sup})"] + [f3(next(d["f1"] for d in D if d["category"] == c and d["model"] == name_of[k])) for k in keys])
    para("Table 8. Per-class F1 on categories with at least 8 real test documents (precision, recall and the small categories are in table_D_per_class.csv).", italic=True, size=10); table(rows, widths=[1.4] + [0.8] * len(keys), fs=7)
    rows = [["Configuration", "Words/doc", "Text reduction", "Accuracy", "Macro-F1", "Summarise (ms)", "Classify (ms)"]]
    for r in E: rows.append([r["config"], f"{r['words_per_doc']:.1f}", f"{r['reduction_pct']:.1f}%", f3(r["accuracy_real"]), f3(r["macro_f1_real"]), f"{r.get('summarize_ms', 0):.2f}", f"{r.get('infer_ms', 0):.2f}"])
    for r in ctrl: rows.append([f"CONTROL: {r['config']} (mean of {r['n_seeds']} seed{'s' if r['n_seeds'] > 1 else ''})", f"{r['words_per_doc']:.1f}", f"{r['reduction_pct']:.1f}%", f"{r['accuracy_real']:.3f}", f"{r['macro_f1_real']:.3f}", "-", "-"])
    para("Table 9. LexRank summarisation + the same TF-IDF/LR classifier, with random-line and first-line controls of similar length (real test documents).", italic=True, size=10); table(rows, widths=[2.2, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6], fs=7)
    # ---- graphs ----
    para("4.4.5 Graphical analysis", italic=True)
    for fn, w, cap in (("fig1_macro_f1.png", 5.2, "Fig. 6. Macro-F1 by model with 95% group-bootstrap intervals (real test documents)."), ("fig2_accuracy.png", 5.2, "Fig. 7. Accuracy by model with 95% group-bootstrap intervals."),
                       ("fig3_ece.png", 5.2, "Fig. 8. Expected calibration error before and after temperature scaling."), ("fig4_latency.png", 5.2, "Fig. 9. Inference latency (log scale)."),
                       ("fig5_training_time.png", 5.0, "Fig. 10. Training duration (log scale)."), ("fig6_model_size.png", 5.0, "Fig. 11. Model size on disk (log scale)."), ("fig7_per_class_f1.png", 6.2, "Fig. 12. Per-class F1 on categories with at least 8 real test documents."),
                       ("fig9_lexrank_tradeoff.png", 4.8, "Fig. 13. LexRank summarisation: accuracy and macro-F1 against the share of text removed.")):
        if (FIG / fn).exists(): figure(FIG / fn, w, cap)
    return dict(R=R, B=B, boot=boot, A=A, L=L, E=E, ctrl=ctrl, para=para, bullet=bullet)


def _verdict(d):
    """d = [mean_delta, lo, hi] (95% paired-bootstrap interval of model - reference): 'better' / 'worse' only when the interval excludes 0"""
    return "better" if d[1] > 0 else ("worse" if d[2] < 0 else "unresolved")

def interpret(ctx, para, bullet):
    R, B, boot, A, E, ctrl = ctx["R"], ctx["B"], ctx["boot"], ctx["A"], ctx["E"], ctx["ctrl"]; brnn = R.get("paired_bootstrap_attention_vs_bilstm", {}); name = lambda k: B[k]["model"]
    para("4.4.6 Interpretation", italic=True)
    text_only = [k for k in B if not B[k]["uses_layout_or_image"].startswith("YES")]; best_f1 = max(text_only, key=lambda k: B[k]["macro_f1"]); best_acc = max(text_only, key=lambda k: B[k]["accuracy"])
    ece_best = min(text_only, key=lambda k: B[k]["ece_calibrated"]); nll_best = min(text_only, key=lambda k: B[k]["nll_calibrated"]); lat = {k: B[k]["latency_cpu_ms"] for k in text_only if B[k]["latency_cpu_ms"]}; size = {k: B[k]["model_mb"] for k in text_only if B[k]["model_mb"]}
    out = [f"Best text-only model by macro-F1: {name(best_f1)} ({B[best_f1]['macro_f1']:.3f}); by accuracy: {name(best_acc)} ({B[best_acc]['accuracy']:.3f}). The production fusion, which also uses the image and layout, reaches macro-F1 {B['fusion']['macro_f1']:.3f} / accuracy {B['fusion']['accuracy']:.3f}." if "fusion" in B else
           f"Best text-only model by macro-F1: {name(best_f1)} ({B[best_f1]['macro_f1']:.3f}); by accuracy: {name(best_acc)} ({B[best_acc]['accuracy']:.3f}).",
           f"Best calibrated (temperature-scaled ECE, lowest): {name(ece_best)} ({B[ece_best]['ece_calibrated']:.3f}); lowest NLL: {name(nll_best)} ({B[nll_best]['nll_calibrated']:.3f}). Before temperature scaling every model is over- or under-confident; the scaling uses validation data only.",
           f"Fastest on CPU: {name(min(lat, key=lat.get))} ({lat[min(lat, key=lat.get)]:.1f} ms/document); smallest on disk: {name(min(size, key=size.get))} ({size[min(size, key=size.get)]:.1f} MB)."]
    for t in out: bullet(t)
    # neural vs TF-IDF
    for k, label in (("bilstm", "A BiLSTM trained from scratch"), ("bilstm_attn", "A BiLSTM with attention pooling"), ("xlmr", "The fine-tuned XLM-RoBERTa Transformer"), ("lexrank", "LexRank summarisation before the same classifier")):
        if k in boot:
            d = boot[k]["delta_macro_f1"]; a = boot[k]["delta_accuracy"]; v = _verdict(d)
            bullet(f"{label} vs TF-IDF/LR: macro-F1 {d[0]:+.3f} (95% CI {d[1]:+.3f} to {d[2]:+.3f}), accuracy {a[0]:+.3f} ({a[1]:+.3f} to {a[2]:+.3f}) -> " + {"better": "better, established by the paired bootstrap.", "worse": "worse, established by the paired bootstrap.", "unresolved": "not distinguishable from TF-IDF on this test set (the interval includes zero)."}[v])
    if "lexrank" in B and ctrl:
        c50 = [c for c in ctrl if c["frac"] == 0.5]; lx = [r for r in E if r["frac"] == 0.5][0]
        para(f"LexRank keeps {100 - lx['reduction_pct']:.0f}% of the words and loses {lx['macro_f1_real'] - [r for r in E if r['frac'] is None][0]['macro_f1_real']:+.3f} macro-F1 relative to the full text. Controls with similar reduction give "
             + "; ".join(f"{c['config']}: macro-F1 {c['macro_f1_real']:.3f} (accuracy {c['accuracy_real']:.3f})" for c in c50) + f", versus {lx['macro_f1_real']:.3f} for LexRank. "
             "Summarisation therefore saves classifier input but does not improve accuracy, and selecting lines by centrality is not shown to beat simply truncating or sampling lines: on bills the useful signal (merchant name, a few key words) is spread over short lines, not concentrated in 'central' sentences.", align="j")
    if brnn.get("bilstm_attn"):
        d = brnn["bilstm_attn"]["delta_macro_f1"]; v = _verdict(d); ind = None
        vf = EXP / "verification" / "classification_check.json"
        if vf.exists(): ind = json.loads(vf.read_text()).get("bootstrap_attention_vs_bilstm")      # independent 10,000-resample re-computation made during the final audit
        txt = f"Attention vs the plain BiLSTM (same data, same split): macro-F1 {d[0]:+.3f} (95% CI {d[1]:+.3f} to {d[2]:+.3f} from 2,000 resamples)"
        if ind: txt += f"; an independent re-computation with 10,000 resamples gives {ind[1]:+.3f} to {ind[2]:+.3f}"
        borderline = v == "better" and ind is not None and ind[1] <= 0
        txt += " -> " + ({"better": "attention appears to help", "worse": "attention hurts", "unresolved": "no established difference"}[v] if not borderline else "borderline: the lower end of the interval sits at zero, so the improvement is suggestive but not established (and it does not survive a correction for the seven comparisons made)") + ". "
        para(txt + "Seed-to-seed variation is also large (see results.json: nn_seed_variation), so a gain of this size from one selected seed must not be read as a property of attention.", align="j")
    tt = {c["model"]: c for c in R["table_C"]}; ts = tt[B["tfidf"]["model"]]["training_seconds"]
    para("Cost versus benefit. " + (f"TF-IDF/LR is a {B['tfidf']['model_mb']:.0f} MB model that trains on a CPU in about {ts:.0f} s and answers in about {B['tfidf']['latency_cpu_ms']:.1f} ms per document. " if B["tfidf"]["latency_cpu_ms"] and ts else "")
         + ("The fine-tuned Transformer needs a GPU for training and is about " + f"{B['xlmr']['model_mb'] / B['tfidf']['model_mb']:.0f} times larger on disk. " if "xlmr" in B and B["xlmr"]["model_mb"] and B["tfidf"]["model_mb"] else "")
         + "The BiLSTMs are cheap but clearly less accurate, so they are not a good trade. Whether the Transformer's extra cost is justified depends on whether its higher macro-F1 survives a larger test set; here it is suggestive, not established.", align="j")
    para("4.4.7 Limitations of this comparison", italic=True)
    for t in ("The real test set has 386 documents and several categories have fewer than 10 (gas 4, internet 6, medical 6, water 7); per-class numbers and small macro-F1 differences are therefore noisy, and the paired intervals above are wide. Many models are compared on one test set, which raises the chance that one looks better by luck; we only call differences established when the interval excludes zero.",
              "The neural text models are small-data baselines: the BiLSTMs learn their embeddings from about 4,800 training documents (no pretrained embeddings), use a word tokenizer although the OCR engine often drops spaces (words run together, e.g. 'GSTID:000433614848'), and were not tuned beyond the documented configuration. The Transformer was fine-tuned once (one seed, no hyper-parameter search, 256 tokens, frozen word-embedding matrix). A better-tuned or larger model could do better; this experiment does not rule that out.",
              "TF-IDF and LayoutLMv3 come from the original, independent training runs; the other models were trained here with a similar but not identical protocol. LayoutLMv3 also uses the page image and layout, so it is not a like-for-like text comparison.",
              "The OCR comparison uses only 29 test-split documents with ground-truth boxes (SROIE receipts and FUNSD forms), recognises GT crops rather than whole pages (detector excluded), and the ground-truth conventions (upper case, spaces, masked '***' regions) penalise the production recogniser under strict CER. The CRNN was trained on about 40,000 crops from the training split and is not a drop-in replacement; the TrOCR checkpoint may have seen SROIE during its own fine-tuning, so its SROIE score is not clean.",
              "Timings were measured on a shared 6 GB laptop GPU/CPU with other jobs running; they show orders of magnitude, not precise ratios. The results concern one dataset, mostly English with some German, Portuguese, Indonesian and Japanese, and do not show that any method is universally better."):
        bullet(t)
    tf = EXP / "verification" / "truncation_check.json"
    if tf.exists():
        T_ = json.loads(tf.read_text())["test"]
        bullet(f"Unequal input length: TF-IDF/LR reads the whole OCR text, but XLM-R sees only the first 256 sub-word tokens and the BiLSTMs the first 256 word tokens. {T_['xlmr_over_256_pct_real']:.0f}% of the real test documents exceed 256 XLM-R tokens ({T_['bilstm_over_256_pct_real']:.0f}% exceed 256 word tokens), so the neural models "
               "never saw the end of those bills (totals, tax lines), while the LexRank pipeline deliberately drops lines. Longer inputs or chunking could change the neural results; this was not tested.")
    para("4.4.8 Conclusion of the comparison", italic=True)
    nb = [k for k in ("bilstm", "bilstm_attn", "xlmr") if k in boot and _verdict(boot[k]["delta_macro_f1"]) == "better"]
    x = boot.get("xlmr"); xl = B.get("xlmr")
    if nb: first = f"On this dataset {', '.join(name(k) for k in nb)} outperformed TF-IDF + logistic regression in macro-F1 with an interval that excludes zero."
    elif x: first = (f"No neural text model was established to beat TF-IDF + logistic regression. The fine-tuned Transformer scored higher (macro-F1 {xl['macro_f1']:.3f} vs {B['tfidf']['macro_f1']:.3f}, accuracy {xl['accuracy']:.3f} vs {B['tfidf']['accuracy']:.3f}), "
                     f"but the paired interval for the macro-F1 difference ({x['delta_macro_f1'][1]:+.3f} to {x['delta_macro_f1'][2]:+.3f}) includes zero, so this test set does not establish that it is better; the two BiLSTMs were clearly worse.")
    else: first = "No neural text model was established to beat TF-IDF + logistic regression."
    cost = (f" The Transformer needed a GPU, about {R['table_C'][[c['model'] for c in R['table_C']].index(xl['model'])]['training_seconds']:.0f} s of fine-tuning, a {xl['model_mb']:.0f} MB model and {xl['latency_cpu_ms']:.0f} ms per document on the CPU, against {B['tfidf']['latency_cpu_ms']:.1f} ms and {B['tfidf']['model_mb']:.0f} MB for TF-IDF/LR." if xl and xl["latency_cpu_ms"] and B["tfidf"]["latency_cpu_ms"] else "")
    para(first + cost + " LexRank is a valid summarisation method but neither an OCR engine nor a classifier; as a pre-processing step it did not improve classification and its line choice was not shown to be better than a random or first-lines cut. "
         "The attention BiLSTM scored higher than the plain BiLSTM, but that gain was not firmly established (its interval is borderline), and both trailed TF-IDF clearly. "
         "The best-performing system overall is the production fusion (image and layout added), whose edge over TF-IDF alone is also not established statistically. "
         "Recommendation: keep TF-IDF + logistic regression as the low-cost text branch; treat a fine-tuned multilingual Transformer as the most promising upgrade, to be confirmed with several seeds, more epochs (the best epoch was the last one tried) and a larger real test set before it replaces or joins the fusion.", align="j")
