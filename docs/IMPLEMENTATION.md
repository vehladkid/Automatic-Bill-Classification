# Implementation status, results and what is left (as of 2026-10-11)

There is no separate `architecture.md`/`implementation.md` before this; the per-task reports are [TASK_A_REPORT.md](TASK_A_REPORT.md), [TASK_B_REPORT.md](TASK_B_REPORT.md), and data docs [DATASET.md](DATASET.md), [DATASET_STATS.md](DATASET_STATS.md), [DATA_QUALITY_REPORT.md](DATA_QUALITY_REPORT.md). Design: [ARCHITECTURE.md](ARCHITECTURE.md).

## Summary table
| Item | Status | Headline result |
|---|---|---|
| Dataset build (real + synthetic, Indian focus) | **Done** | 8,183 docs = 5,753 real + 2,430 synthetic (Task A uses 6,111 labelled; Task B 6,442); 0 leaked val/test near-duplicates; `rent` has **no real data** |
| Task A - expense category (16 classes) | **Done** | calibrated fusion: real test acc 0.907, macro-F1 0.839 |
| Task B v1 - field extraction (34 fields) | **Done (frozen baseline)** | hybrid real-test micro-F1 0.663, macro 0.578; Indian-real 0.599 |
| Task B v2 - real Indian GST data | **In progress, paused** | Tier A (54 pages) annotated; human QA pending; Tier B not started; **no retraining yet** |
| Application | **Local Streamlit demo built** (`app/`); 26/26 backend checks pass | upload -> OCR -> Task A -> Task B -> filed by category |

## 1. Dataset (done)
Counts re-verified 2026-10-11 against `datasets/metadata/manifest.jsonl` (the earlier 7,924/5,494 figures predate the India-ReceiptBench round). Built from user Indian bills, SROIE, CORD, ReceiptBench subset, PT invoices, FUNSD, RVL-CDIP subset, medical bills, Indian grocery GST, plus synthetic Indian bills. Group-based splits (Task A: train 4,848 / val 634 / test 629; real counts 2,904 / 391 / 386). Quality notes: weak real test sets for several classes (4-18 docs), `rent` synthetic-only, OCR via RapidOCR (errors on tiny or rotated text). Disk cleanup of verified duplicates done (about 10 GB freed earlier).

## 2. Task A results (real test unless stated; 95% CI over groups in the report)
| model | real acc | real macro-F1 |
|---|---:|---:|
| TF-IDF + LR | 0.883 | 0.831 |
| ConvNeXt-Tiny | 0.800 | 0.627 |
| LayoutLMv3 | 0.863 | 0.704 |
| **Calibrated fusion (final)** | **0.907** | **0.839** (real-Indian acc 0.955, ECE 0.020) |
HR routing at tau 0.61: about 93% auto-routed at about 94% precision. Label audit fixed 90 labels; synthetic data adds little for real accuracy (0.871 vs 0.883 acc, text model). Open issues: `rent` real accuracy unknown; per-class real support small for gap classes.

## 3. Task B v1 results (real test, same documents for all systems)
| system | real micro-F1 |
|---|---:|
| baseline (rules) | 0.444 |
| candidate ranker | 0.546 |
| LayoutLMv3 | 0.646 |
| **hybrid (final)** | **0.663** (macro 0.578; Indian-real 0.599) |
Critical fields (hybrid F1): total 0.82, tax_total 0.82, subtotal 0.76, date 0.75, merchant 0.54, invoice_no 0.42; Indian GST on tiny support: GSTIN 0.50 (n=13), CGST 0.73 (n=4), SGST 0.60 (n=4); IGST and HSN/SAC have **no real evaluation data**. Verdict in the report: research prototype, not production-grade. Largest error group: gold value not locatable in OCR text (224 of ~500 field errors) - an OCR/format problem more than a model one.

## 4. Task B v2 (in progress)
Done: V1 frozen and hashed; data-gap analysis; 169 grocery-GST photos oriented + OCR'd; supplier/layout analysis (one buyer, ~20-23 sellers, one region); tiers (A 54 / B 39 / C 76 not annotated); review tool; Tier A annotated (silver, by the model, not a human): GSTIN 54/54, CGST/SGST 49, HSN list 53, taxable_value 53, tax_total 32, subtotal 0 (none prints an explicit subtotal), IGST 0 (no inter-state invoice in the pool); V2 schema (`hsn_sac` list, `taxable_value`, `discount`, `round_off`); proxy QA by blind AI reviewers on 10 pages: 99.3% value agreement, 94% of boxes correct, one systematic error (derived round-off) fixed.
**Waiting on you:** the human answer sheet (`datasets/interim/v2_pool/qa/HUMAN_QA/answer_sheet.csv` is still blank). Tier B, the V2 split, retraining and the V1-vs-V2 comparison are all gated on it.
Not solvable with this pool: IGST supervision, broad Indian generalization (single buyer/region), real test evidence for GST fields (V1 test is frozen with 5 GSTIN / 3-4 CGST-SGST docs).

## 5. What is left, in order
1. Human QA of the 10 Tier-A pages -> score -> fix systematic issues (needs your answer sheet).
2. Tier B (39 pages) with the same conventions, then the 93-document audit (stop for review).
3. Leakage audit of V2 pages vs the frozen V1 test, V2 train/val split, retrain ranker + LayoutLMv3, rebuild calibration/hybrid, V1-vs-V2 on the same frozen test, Indian-real table, error analysis, unseen-document test, `docs/TASK_B_V2_REPORT.md`, keep/drop decision.
4. Separate real inter-state (IGST) invoice source; more real data for `rent` and weak classes.
5. Application: a local Streamlit demo exists (`app/`, see EXAMINER_PROJECT_WALKTHROUGH.md); ingest, policy engine, API, auth and deployment are not built.

## 6. Environment
RTX 4050 6 GB, `.venv` (torch 2.9.1+cu128, onnxruntime-gpu 1.22). Disk is tight (about 3 GB free on a 98%-full drive; cleanup options were proposed, nothing deleted yet). Reproduce: `bash scripts/run_pipeline.sh` (data), `scripts/taskA/*`, `bash scripts/taskB/run_taskB.sh`.
