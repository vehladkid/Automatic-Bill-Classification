# Final verification audit - experiments/text_vs_neural and docs/report/PBL_Report

Scope: no model was retrained; the production pipeline was not modified (the only file touched outside `experiments/` is the report generator `scripts/make_pbl_report.py`). Backups of the report, its generator and the figures: `docs/report/backup_20261027_191731` (before the experiments section) and `backup_20261027_200844` (before this audit's corrections). Every check below has a script in this folder and a JSON output.

## 1. Verified claims
* **Classification metrics** - 1,271/1,273 independent checks agree (accuracy, macro P/R/F1, weighted-F1, ECE, NLL, Brier, per-class F1 and support, confusion matrices, temperatures re-fitted on validation, CSV vs recomputation) for TF-IDF, BiLSTM, BiLSTM+attention, LexRank-50%, XLM-R, LayoutLMv3 and the fusion (`classification_check.json`).
* **Splits** - val 634, test 629, real test 386 (382 groups); zero document-id or template-group overlap between train/val/test.
* **Bootstrap** - re-implemented with different seeds and 10,000 resamples: every endpoint within 0.012 of the stored 2,000-resample values, and the verdict (better/worse/unresolved) is identical for all six comparisons with TF-IDF. Extra checks: document-level bootstrap and exact McNemar agree (XLM-R vs TF-IDF: p = 0.42; fusion p = 0.11; LexRank p = 0.004 for lower accuracy).
* **OCR** - all 36 CER/WER values (3 recognisers x all/SROIE/FUNSD x 4 metrics) reproduced exactly (< 1e-9) by re-running inference and an independent Levenshtein (two independent implementations agree on known cases and 400 fuzz pairs). 29 test documents (11 SROIE, 18 FUNSD), 2,862 crops, no overlap with the 434 training documents; 2 masked ('***') GT lines.
* **Sizes** - every size matches the file on disk; the XLM-R and TrOCR checkpoint SHA-256 match Hugging Face.
* **Latency** - re-measured: all eight within 1.5x of the stored values.
* **Report** - 255 automated checks: Tables 5-9 cell-by-cell vs results, per-class table vs CSV, numbering, 13 embedded images byte-identical to saved figures, graphs regenerate pixel-identically, DOCX/PDF consistent. OCR (Table 5) has no LexRank row; fusion and LayoutLMv3 are marked as using image/layout; XLM-R is described as not established (interval includes zero); limitations are in 4.4.7 and 4.5.

## 2. Numerical discrepancies
* BiLSTM+attention raw NLL: stored 1.3848 (real) / 0.8499 (all) vs 1.3852 / 0.8501 recomputed. Cause: the project's `nll()` clips probabilities at 1e-9 and one test document has a true-class probability of 8.6e-10. Convention, not an error; no value changed.
* Re-running saved BiLSTM weights on CPU: logits differ by up to 0.12 (BiLSTM) / 0.01 (attention) from the stored GPU logits - float noise, 100% identical predictions.
* LayoutLMv3 production checkpoint on 40 test pages: logits differ by up to 1.9 from the stored ones (stored = training-time evaluation at 448 tokens; this run = the production 512-token path in bf16); predictions identical (40/40). Consistent with (but not proof of) the previously noted 448-vs-512 token mismatch plus bf16 noise; it changed logits, not labels, on these 40 pages.
* Latency ratios 0.97-1.46 (shared machine) - within the 2x tolerance.

## 3. Missing artefacts
* **XLM-R fine-tuned weights** and per-epoch checkpoints were not saved (disclosed). Its reported test metrics are verified as arithmetic on the stored logits only; the logits cannot be tied to a loadable model, and a re-run (one seed, bf16, GPU) is not guaranteed to reproduce them. Its model size and latency refer to the pretrained architecture/checkpoint.
* Per-line OCR hypotheses were not stored by the experiment (re-created and stored by this audit in `ocr_hypotheses_reproduced.json`).
* LexRank summaries/classifiers and TF-IDF re-fit are not stored (need re-running). Training times cannot be re-measured without retraining (only internal consistency checked).

## 4. Statistical limitations
* XLM-R vs TF-IDF: macro-F1 +0.057 (95% CI -0.004 to +0.127), accuracy +0.015 (-0.016 to +0.047), McNemar p = 0.42 -> **not established**; single seed, untuned, best epoch was the last one (6/6).
* **Attention vs BiLSTM is borderline, not established**: stored 2,000-resample interval +0.003..+0.174; independent 10,000-resample intervals -0.004..+0.174 and -0.001..+0.173; fails the Bonferroni correction (7 comparisons). The earlier summary and report text that said attention "helped" was an overclaim and has been corrected.
* Surviving Bonferroni correction (7 comparisons): both BiLSTMs worse than TF-IDF, LayoutLMv3 worse in macro-F1 (-0.127, CI -0.245..-0.008), LexRank worse in accuracy (-0.031). Not surviving/unresolved: XLM-R, fusion, attention-vs-BiLSTM, LexRank macro-F1.
* 386 real test documents; categories with 4-7 documents (gas, internet, medical, water) make macro-F1 noisy; ECE has no confidence interval.
* Seed variation of the BiLSTMs is large (accuracy 0.68-0.71 and 0.74-0.78 across 3 seeds); the reported seed was chosen on validation.
* **Unequal input length** (found in this audit): 47% of real test documents exceed 256 XLM-R tokens and 34% exceed 256 word tokens; TF-IDF reads everything. Now disclosed in the report (4.4.7).

## 5. Report formatting / content issues found
* Figure numbering jumped from Fig. 4 to Fig. 9 -> renumbered 1-13 (fixed). Table 1 had no caption line -> added.
* Methodology said XLM-R was "fully fine-tuned" although the word-embedding matrix was frozen -> corrected.
* "Test set scored once per final model" was inaccurate (all BiLSTM seeds and LexRank settings were scored for transparency) -> rewritten.
* Truncation of inputs was not disclosed -> added. The m5 script docstring said "median ~110 tokens"; the measured median is 184 -> comment corrected (no result changed).
* Abstract/4.3 wording on the fusion ("beats the best single model", "better calibrated") overstated what overlapping/paired intervals show -> softened in the previous revision; paired test confirms fusion vs TF-IDF is unresolved (+0.008, CI -0.040..+0.055).

## 6. Corrections applied / still required before submission
Applied (report regenerated; DOCX + PDF, 19 pages; backup kept): items in section 5, attention wording, truncation limitation.
Still required: (a) decide whether to keep the XLM-R row as "single recorded run" or re-run it with saving enabled and several seeds; (b) replace the placeholder author line; (c) the reference list still has only six journal papers of the 10-15 your template asks for; (d) do not state that attention helps or that XLM-R is better in the viva - say "suggestive, not established".
