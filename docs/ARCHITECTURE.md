# Architecture - ML-based Automated HR Expense Claim Intelligence (dataset + model layer)

Approved scope: Task A (expense category) and Task B (field extraction) only, plus a local Streamlit demo app (`app/`). No email ingestion, scraping backend, policy engine, auth, API or deployment exists. Full audited description: [EXAMINER_PROJECT_WALKTHROUGH.md](EXAMINER_PROJECT_WALKTHROUGH.md). Status and numbers: [IMPLEMENTATION.md](IMPLEMENTATION.md).

## 1. Intended end-to-end flow (what the models are for)
```
claim image / PDF
   -> OCR (RapidOCR, ONNX, GPU)  : words + boxes + confidences
   -> Task A  expense category   : 16 labels, calibrated confidence       [DONE]  -> HR-routing threshold
   -> Task B  field extraction   : 34 fields (merchant, date, totals, GSTIN, CGST/SGST...)  [DONE v1; v2 data in progress]
   -> app: results view + filing in outputs/classified/<category>/
```
Low-confidence outputs are routed to HR instead of auto-approved; every model reports calibrated confidence for that purpose.

## 2. Data layer (`datasets/`, `scripts/*.py`)
- **Canonical manifest** `datasets/metadata/manifest.jsonl` (one row per document: provenance, licence, country, language, synthetic flag, label quality, sha256, group id, split). Real and synthetic are always separable.
- **Sources**: user-supplied Indian bills (`datasets/indian/`, electricity/gas/water/food/fuel/train/parking...), public sets (SROIE, CORD, ReceiptBench subset, InvoicesReceiptsPT, FUNSD, RVL-CDIP subset, medical bills, Indian grocery GST), synthetic Indian bills (Task A: 2,430 incl. 180 hard negatives; Task B-only: 1,080).
- **Leakage control** (`deduplicate.py`, `create_splits.py`, `audit_leakage.py`): sha256 exact + pHash/dHash near-duplicate + merchant/layout grouping; splits are by group; independent audit found 0 val/test images within pHash <= 4 of train. Human label fixes in `label_overrides.json` (90).
- **Views**: `processed/classification` (Task A), `taskB` (train/validation/test jsonl); `indian_all/` = relative-symlink browse view.
- **OCR** (`run_ocr.py`, `taskA/run_ocr_gpu.py`, `taskB/run_ocr_gpu_b.py`): RapidOCR via onnxruntime-gpu 1.22 on an RTX 4050 (6 GB), ~170 img/min.

## 3. Task A - expense category classification (`scripts/taskA/`, `models/taskA/`)
Three independent models, fused late:
1. TF-IDF (char 2-4 + word 1-2) + logistic regression on OCR text.
2. ConvNeXt-Tiny (IN-22k weights, local safetensors) on the image.
3. LayoutLMv3-base (text + layout + image), micro-batch 2 x 8 accumulation, 448 tokens, bf16.
Per-model temperature scaling + isotonic calibration, weighted probability fusion (text 0.5 / image 0.3 / layoutlm 0.2), and an HR-routing threshold tau = 0.61 picked on validation only. Inference: `scripts/taskA/predict.py`.

## 4. Task B - field extraction (`scripts/taskB/`, `models/taskB/`)
34-field canonical schema (`schema.py`, frozen as V1). Four systems compared on the same real held-out test:
- **Baseline**: regex/keyword rules over OCR lines (`baseline.py`).
- **Candidate ranker**: generate candidate spans per field (`words.py`) + HistGradientBoosting ranker (`ranker.py`).
- **LayoutLMv3 token classifier**: BIO tags, 384-subtoken windows stride 96 (`ie_layoutlm.py`, `decode_ie.py`).
- **Hybrid** (final V1 system): per-field system chosen on real validation only; applicability gating by keyword evidence (`gating.py`); isotonic confidence calibration; real-validation threshold policy (MIN_REAL_VAL 30, default tau 0.5, low-support fields flagged). Inference: `scripts/taskB/predict.py <image|pdf|document_id>`.
**V2 schema** (`schema_v2.py`, `models/taskB/v2/field_schema_v2.json`): V1 + `hsn_sac` as a list of unique codes, new `taxable_value`, `discount`, `round_off`; `subtotal` = explicit labelled subtotal only. V1 stays frozen and reproducible (`reports/taskB/v1_snapshot/` with SHA256 manifest).

## 5. Task B V2 annotation pipeline (`scripts/taskB/v2_*.py`, `datasets/interim/v2_pool/`)
orientation-corrected OCR of 169 real Indian GST invoice photos -> supplier/layout analysis -> value tiers A/B/C -> SUGGESTION_ONLY candidates (regex, never model output) -> annotation (Tier A: 54 pages, silver) -> blind QA (proxy AI reviewers + human packet) -> migration to V2 schema. Static HTML review tool: `datasets/interim/v2_pool/review/index.html`.

## 6. Design rules that apply everywhere
Real-only metrics are the headline; synthetic results are reported separately and never as real-world evidence. All tuning on validation only, test touched once per system. Group-aware splits and bootstrap CIs over groups. No silent label or split changes. Disk/GPU: bf16, small micro-batches, no gradient checkpointing for LayoutLMv3.

## 7. Directory map
`datasets/` data and metadata | `scripts/` pipeline (`taskA/`, `taskB/`) | `models/` weights (`taskA/`, `taskB/`, `taskB/v2/`, `pretrained/`) | `reports/` machine-readable results (`taskA/`, `taskB/`, `taskB/v2/`) | `docs/` human-readable reports.
