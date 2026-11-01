# Phase 1 — Audit: why did the LLM move F1 from 0.703 to 0.704?

Written before any app code was edited. Every number below was recomputed from files in this repo by `reports/extraction_audit/diagnose_val.py` (read-only; uses the cached LLM answers, no model run). Output: `diagnose_val.json`.

## 1. Execution flow (Streamlit UI → saved result)

| # | Stage | Code | Model / engine |
|---|---|---|---|
| 1 | Upload validation, decode, EXIF-rotate, cap long side at 3000 px, save `page_1.png` (PDF: `pdftoppm` 110 dpi JPEG, ≤10 pages) | `app/service.py::validate_upload/_prepare_pages` | PIL, poppler |
| 2 | OCR on the page file. PDF page 1 uses the PDF text layer when present (conf 1.0), else RapidOCR | `service._ocr`, `scripts/pdf_utils.py` | RapidOCR (onnxruntime GPU, `max_side_len=1600`) |
| 3 | Segments → words (`split_words`, `reading_order`) → lines (`baseline.lines_of`) → `ocr.txt` | `service._text_from_segments` | rules |
| 4 | **Task A** expense category (16 classes): text-LR + ConvNeXt-tiny + LayoutLMv3, temperature-calibrated weighted fusion, τ = 0.61 routing | `scripts/taskA/predict.py::Predictor` | trained, frozen |
| 5 | **Task B V1** field extraction (per-field choice of regex baseline / feature ranker / LayoutLMv3 token tagger, isotonic confidence, applicability gating) | `scripts/taskB/predict.py::Extractor.run` | trained, frozen |
| 6 | Optional **LLM** (`HRBC_LLM_MODE`, default `off`): Qwen2.5-VL-3B via Ollama; prompt = OCR lines + boxes (+ image); output parsed, each value *grounded* in OCR text, fused with V1 | `app/llm/{enhance,validate,fuse,service,client}.py` | qwen2.5vl:3b Q4_K_M |
| 7 | Tax suggestions, **document profile** (rule-based employee-document type / benefit category; LLM opinion counted only if its quoted evidence is in the OCR), field report | `app/tax/suggest.py`, `app/doc_profile.py`, `app/field_report.py` | rules |
| 8 | Save original + `record.json`, `task_b.json`, `ocr.txt`… under `outputs/` | `service._finish` | — |

Three different "classification" things exist and must not be conflated:
* **Task A** — 16-class *expense category*, trained, the only one that places the file in `classified/<category>/`.
* **Document type / benefit category** — rule-based (`doc_profile`, `config/taxonomy.json`), not trained, no labelled real examples for most classes.
* **LLM `classification.document_kind`** — an opinion about the document *kind* that the rules may confirm or send to review; it never changes the Task A category.

Traced example (existing record `outputs/results/20261012T143108_d9a587c9`, an IRCTC PDF): PDF text layer (285 words, no OCR) → Task A `train` 0.9999 auto → V1 fields from LayoutLMv3/ranker with calibrated confidences 0.31–0.995 (merchant 0.31, invoice no. 0.32) → LLM `off` (none run). A fresh trace of live documents is part of Phase 2.

## 2. Exact definition of the metric that moved 0.703 → 0.704

* **Metric:** pooled (micro) field-level F1: TP/FP/FN summed over 11 fields (`merchant_name, invoice_number, invoice_date, total_amount, subtotal, tax_total, cgst, sgst, igst, gstin, currency`; `hsn_sac` excluded) and over all **295 real validation documents** (`scripts/taskB/evaluate.py::micro`).
* TP = predicted and normalised-equal to ground truth (amounts equal within tolerance, dates ISO, ids upper-alnum, text = case/accents/non-alnum-stripped **exact** equality); FP = predicted and (GT absent or different); FN = GT present and (not predicted or different). Only fields the source dataset annotates are scored for that doc.
* 0.703 = V1: TP 637, FP 205, FN 333 (P 0.757, R 0.657). 0.704 = V1 + LLM, *LLM-only values kept as suggestions* (default policy): TP 638, FP 204, FN 332. **The whole improvement is one field out of 970 ground-truth fields.** (Reproduced exactly: `diagnose_val.json`.)

## 3. Most likely causes (each backed by a measurement)

1. **By construction the default policy cannot add recall.** `enhance.py` moves every LLM-only value to `suggestions` and keeps it out of the scored fields. Fields can then only change when V1 *and* the LLM both proposed a value and V1 failed grounding. Measured on the cached answers: 527 agreeing fields (identical), 312 V1-only, 134 conflicts where V1 was kept, **29 where the LLM overrode V1** — 6 fixed a wrong V1 value (4 hotel totals, 1 merchant, 1 date), 5 broke a correct V1 value (2 CORD amounts read at the wrong scale, 1 merchant, 1 date, 1 currency), the other 18 were wrong either way or on fields without ground truth. Net +1 TP (hand-checked from the printed override list). The score could not have moved much, whatever the LLM knew.
2. **The "obvious" alternative (write LLM-only values into the result) is worse, not better:** F1 0.692 (TP +26 but FP +77). LLM raw values for `total_amount` contradicted a correct V1 61 times, `subtotal` 22, `tax_total` 18 (`diagnose_val.json`, `llm_contradicts_correct_v1_raw`). Validation (grounding) rescues part of that.
3. **The headline number mixes two very different populations.** 168 of the 295 documents (SROIE 68, CORD 100) use the dataset's **ground-truth OCR text** (`ocr_source: ground_truth` in the annotation files), not RapidOCR. Slice results:

   | slice | n | V1 F1 | V1 + LLM (default) F1 | V1 + LLM (add) F1 |
   |---|---|---|---|---|
   | ground-truth OCR (SROIE+CORD) | 168 | 0.866 | 0.861 | 0.848 |
   | **RapidOCR (receiptbench+invoices_pt)** — what the app actually sees | 127 | **0.567** | 0.573 | 0.570 |
   | all | 295 | 0.703 | 0.704 | 0.692 |

   So the benchmark never tested the app's OCR for 57 % of its documents, and on the other 43 % V1 is already far weaker (0.567). The LLM neither helps on clean text (−0.005) nor materially on noisy text (+0.006).
4. **Ceiling / label issues, not only model issues.** Where V1 is wrong, the GT value is *not* visible in the OCR text for `invoice_date` 53 of 170, `currency` 53 of 96 (currency is inferred, never printed as a word → the grounding rule can never accept it), `total_amount` 30 of 281, `merchant_name` 25 of 177. Part of that is real OCR/text damage (glued tokens such as `16/03/202311:18:54AM`, `Me1bourne&Z1+2`, `TOTAL$36.60` — spaces are lost, which also defeats the date regex used for grounding), part is annotation convention. These cases cannot be fixed by any extractor reading this text.
5. **Headroom that *is* reachable from the text:** V1 wrong while GT *is* present in OCR: `merchant_name` 61, `invoice_number` 29, `total_amount` 21, `sgst` 5 — and V1 emits `total_amount = 0.0` on hotel bills where GT is 699 / 6272 / 2500 / 1450 (the LLM override fixes exactly these). Those are the targets.
6. **Evaluation methodology caveats** (not defects, but they bound the conclusion): exact-match scoring penalises merchant-name convention differences ("fgcgis" vs "fgcgasservicestation"); only one LLM prompt/model was tried; the benchmark prompt builds lines from annotated words (`llm_eval.segs_of`) while the app sends raw RapidOCR segments (slightly different prompt text); the test split has never been run with the LLM (and must not be used for tuning).

**Answer to "is the bottleneck OCR, extraction, classification or evaluation?"** (to be confirmed in Phase 2): evaluation design (mixed OCR provenance + a policy that cannot add recall) explains the *flat 0.703→0.704*; the real accuracy bottleneck is on the RapidOCR slice, where it is a mix of text loss (OCR) and V1 extraction errors. Classification is not part of this metric at all.

## 4. Proposed work and how each item will be evaluated

| # | Change / investigation | Evaluation (same docs, GT, scorer for baseline and candidate) | Gate to integrate |
|---|---|---|---|
| P2 | **Measure OCR separately.** Run the app's OCR path (same image preparation, RapidOCR engine) on all 295 real val images. CER/WER vs dataset GT text on SROIE+CORD (168); for all 295, % of GT field values *recoverable* from OCR text. | CER, WER (bag-of-words P/R), per-field recoverability: GT-OCR text vs RapidOCR text | diagnosis only |
| P2b | Run **V1 on live OCR** and compare with V1 on the dataset words → end-to-end degradation due to OCR. | micro-F1 by slice | diagnosis only |
| P2c | OCR-text-level fixes **only if** the measurements show a specific loss (e.g. glued tokens / date grounding) fixable with existing dependencies. | recoverability and V1 F1 before/after on val | ≥ gain on val RapidOCR slice, no regression on GT-OCR slice |
| P3a | **LLM on live OCR**, all 295 val docs, text mode, with the *app's* prompt input. Error taxonomy: wrong-over-correct, hallucinated (ungrounded), invalid JSON, subtotal/tax/total confusion. | field P/R/F1, wrong-over-correct count, rejected-as-ungrounded rate, JSON failures | diagnosis |
| P3b | **Conservative, rule-based correction policy** (no LLM invention): (i) treat V1 `total_amount == 0` as missing; (ii) allow a grounded LLM-only value only for fields whose measured LLM-only precision on validation is high *and* that pass cross-field arithmetic; (iii) keep everything else as suggestion. Per-field choices are made on **validation only**. | micro-F1 + bootstrap CI on val; then **one** frozen-config run on test (never tuned on) | val gain with CI excluding 0 on the RapidOCR slice, no field-level regressions; otherwise unchanged |
| P4 | **Classification audit**: Task A metrics on real val vs test, per-class P/R, confusion; routing τ; label-mapping review; LLM zero-shot category opinion on the same docs (cost-limited). | accuracy, macro-F1, per-class, real vs synthetic | integrate only if reproducible gain without material regression; else unchanged |
| P5 | Regression tests + selftest + demo smoke; frozen Task B test hash and V1 snapshot re-verified. | existing 395 tests + new focused tests | all pass |

Constraints honoured: no training or weight changes, no downloads, no V2 annotation, Ollama model untouched, one model-holding process at a time (6 GB GPU), test split not used for tuning.
