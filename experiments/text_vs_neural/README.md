# Text-classification and OCR comparison: TF-IDF vs BiLSTM vs BiLSTM+attention vs LexRank vs Transformer vs LayoutLMv3

Everything here is experimental and separate from the production pipeline. Nothing in `scripts/`, `models/`, `datasets/` or `reports/` was modified; production checkpoints and splits are only read.

## What was compared (and what each experiment measures)
* **Experiment A - OCR recognition** (`ocr_exp.py`): image crop -> text, scored with CER/WER against the project's ground-truth SROIE/FUNSD transcriptions (test-split documents only). RapidOCR PP-OCRv4 recogniser (production), a CRNN (CNN+BiLSTM+CTC) trained here, TrOCR-small-printed (pretrained Transformer). LexRank cannot read images, so it has no OCR row.
* **Experiment B - text classification of the OCR text** (16 expense categories, project splits train 4,848 / val 634 / test 629; headline = REAL test documents, n = 386): `m1_tfidf.py`, `m2_rnn.py bilstm|bilstm_attn`, `m3_lexrank.py` (+ `m3b_lexrank_controls.py`), `m5_xlmr.py`, reference rows `m6_layoutlm.py` (production LayoutLMv3, uses image+layout) and `m7_fusion_reference.py` (production fusion).
* `aggregate.py` builds the tables (`table_A..E_*.csv`, `outputs/results.json`) and the graphs (`figures/fig1..fig9`); `report_section.py` writes section 4.4 of the PBL report from those files.

## Protocol
Same documents, same OCR text (`row["text"]`), same labels; hyper-parameters/epochs/early-stopping on validation only; test scored once per final model; one temperature per model fitted on validation logits only; ECE = 10 equal-width bins, top-label; paired group bootstrap (2,000 resamples, same resampled documents for every model) for differences against TF-IDF.

## Runs that failed, were repeated or are limited (full disclosure)
* CRNN OCR: the first run finished before the case/space-insensitive CER was added to the scorer; it was re-scored with `CRNN_REUSE=1` (same trained weights, no retraining).
* XLM-R: fine-tuning all 278M parameters with AdamW would not fit in 6 GB, so the 192M-parameter word-embedding matrix is frozen (a pre-run design decision; no out-of-memory run was attempted). One seed, no hyper-parameter search, and the best epoch was the last epoch tried (6 of 6), so it may still have been improving.
* BiLSTMs: three seeds each; the seed with the best validation macro-F1 is the reported one; per-seed test numbers are in `outputs/m_*.json` and `results.json` (`nn_seed_variation`) for transparency (large seed variance).
* A 507 MB partial checkpoint download was deleted by mistake and re-downloaded; unrelated to results.
* TrOCR-small-printed was fine-tuned on SROIE by its authors (model card), so its SROIE score is **non-comparable**; FUNSD is its clean subset.
* Latency/throughput figures were measured on a shared laptop (other jobs running), after warm-up, batch size 1; use them for orders of magnitude only.
* Not done: attention-based OCR decoder trained here (TrOCR covers the Transformer/attention OCR case); multi-seed Transformer; evaluation on any new data.

## Reproduce (from the repo root; GPU 6 GB; run GPU jobs one at a time and stop the demo servers first)
```bash
cd experiments/text_vs_neural
../../.venv/bin/python m1_tfidf.py                  # TF-IDF + LR (re-fit; checks it reproduces the production model exactly)
../../.venv/bin/python m2_rnn.py bilstm             # BiLSTM, 3 seeds
../../.venv/bin/python m2_rnn.py bilstm_attn        # BiLSTM + attention, 3 seeds
../../.venv/bin/python m3_lexrank.py                # LexRank summaries at 100/75/50/25 % + TF-IDF/LR
../../.venv/bin/python m3b_lexrank_controls.py      # random / first-lines controls
../../.venv/bin/python m6_layoutlm.py               # production LayoutLMv3 reference (+ latency)
../../.venv/bin/python m7_fusion_reference.py       # production fusion reference
# XLM-R checkpoint (1.1 GB, not in git): python pretrained/fetch_parallel.py https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/model.safetensors pretrained/xlm-roberta-base/model.safetensors
#   plus config.json, tokenizer.json, sentencepiece.bpe.model, tokenizer_config.json from the same repo
HF_HUB_OFFLINE=1 ../../.venv/bin/python m5_xlmr.py
../../.venv/bin/python ocr_exp.py rapidocr; ../../.venv/bin/python ocr_exp.py crnn; ../../.venv/bin/python ocr_exp.py trocr   # TrOCR needs microsoft/trocr-small-printed in pretrained/
PYTHONPATH=<dir with matplotlib> ../../.venv/bin/python aggregate.py            # tables + figures (matplotlib is not in the project venv)
# report: PYTHONPATH=<dir with python-docx and matplotlib> ../../.venv/bin/python ../../scripts/make_pbl_report.py
```

## Reproducibility status (from the final audit; scripts in `verification/`)
| Item | Reproducible from saved artefacts? |
|---|---|
| Every classification metric (accuracy, macro P/R/F1, weighted-F1, ECE, NLL, Brier, per-class, confusion) | **Yes** - recomputed independently from `outputs/*_logits.npz` + the project's split files: 1,271 of 1,273 checks agree; the 2 differences are the 1e-9 probability clip in the project's `nll()` (BiLSTM+attention raw NLL 1.3848 vs 1.3852) |
| Temperatures, paired bootstrap, McNemar | **Yes** - re-derived with independent code (10,000 resamples, different seeds); verdicts agree, except that the attention-vs-BiLSTM interval is borderline (stored +0.003..+0.174, independent -0.004..+0.174) |
| OCR CER/WER | **Yes by re-running inference** - hypotheses are not stored by the experiment scripts; re-running RapidOCR, the saved CRNN weights and the TrOCR checkpoint on the same crops reproduced all 36 stored values exactly (hypotheses saved in `verification/ocr_hypotheses_reproduced.json`) |
| TF-IDF, BiLSTM, BiLSTM+attention test logits from saved weights | **Yes** - TF-IDF exactly; BiLSTMs to within GPU/CPU float noise (max logit diff 0.12 and 0.01, identical predictions) |
| LayoutLMv3 (production) | Checkpoint re-run on 40 test pages gives identical predictions; logits differ by up to 1.9 (bf16 + 512-token inference vs the 448-token training-time evaluation) |
| Figures and CSV tables | **Yes** - regenerated byte/pixel-identically (`verification/verify_figures.py`) |
| Model sizes | **Yes** (file sizes match; XLM-R and TrOCR files match the Hugging Face SHA-256) |
| Latency | Re-measured within 1.5x of the stored values (shared machine); XLM-R latency re-measured with the **pretrained** weights (same architecture) because the fine-tuned weights were not kept |
| Training times | Only internal consistency can be checked (history vs stored value); they need re-training to re-measure |
| **XLM-RoBERTa fine-tuned model** | **NO** - the fine-tuned weights were not saved (to conserve disk), no per-epoch checkpoint exists. Its stored test logits give the reported metrics (arithmetic verified), but the logits cannot be tied to a model, and re-training (one seed, bf16 GPU) is not guaranteed to reproduce them bit-for-bit. Treat its numbers as a single recorded run |
| LexRank pipelines | Needs re-running (summaries and classifiers were not stored); logits are stored |
Input-length caveat found in the audit: 47% of real test documents exceed 256 XLM-R tokens and 34% exceed 256 word tokens, so the neural text models see truncated text while TF-IDF reads everything (`verification/truncation_check.json`).
