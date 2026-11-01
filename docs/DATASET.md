# Expense-claim dataset: what's in it, where it came from, how it was built

> **Task B (field extraction, 2026-10-05): [TASK_B_REPORT.md](TASK_B_REPORT.md).** Canonical annotations in `datasets/processed/taskB/`, extra synthetic fields in `datasets/synthetic_b/` (Task-B-only, never mixed into Task A), code `scripts/taskB/`, models `models/taskB/`, metrics `reports/taskB/`.

> **Public dataset / pretrained-model research for Indian expense and tax-document types (2026-10-19): [DATASET_MODEL_RESEARCH.md](DATASET_MODEL_RESEARCH.md).** Normalized document-type manifest: `datasets/metadata/doc_type_manifest_v1.jsonl`; schema: `config/doc_type_schema_v1.json`; tools: `scripts/dataset_discovery/`.

> **Task A models, results and label-audit (2026-10-05): [TASK_A_REPORT.md](TASK_A_REPORT.md).** Code in `scripts/taskA/`, weights in `models/taskA/`, metrics in `reports/taskA/`.

> **Updated 2026-10-04 (real-data round):** 7,924 docs = 5,494 real + 2,430 synthetic. New sources (my_receipts, ReceiptBench subset, InvoicesReceiptsPT subset, hf medical bills, your electricity/gas/water/restaurant zips), hard negatives, tighter grouping, GPU OCR, leakage audit. Details, before/after table and rejected candidates: [REAL_DATA_ACQUISITION_2026-10-04.md](REAL_DATA_ACQUISITION_2026-10-04.md). The source table in section 3 below is the earlier round; counts in sections 1-5 that differ are superseded by DATASET_STATS.md.

Rebuild everything: `bash scripts/run_pipeline.sh` (OCR step ~30 min CPU, resumable). Numbers: [DATASET_STATS.md](DATASET_STATS.md) (generated).
Quality problems and gaps: [DATA_QUALITY_REPORT.md](DATA_QUALITY_REPORT.md). Labels: [TAXONOMY.md](TAXONOMY.md). Next steps: [TRAINING_RECOMMENDATIONS.md](TRAINING_RECOMMENDATIONS.md).

## 1. Layout (existing `datasets/` kept; nothing deleted)

```
datasets/
  indian_all/            GENERATED browsable view of every Indian doc as relative SYMLINKS (no duplicate bytes): real/<cat>/ + synthetic/<cat>/ + index.csv (source, license, split). Rebuilt by scripts/export_indian_view.py
  real/<category>/       user-supplied real NON-Indian bills + _provenance.json (country/license), e.g. real/food = 198 US restaurant receipts
  indian/<category>/     your manually collected real bills (folder name = label) - moved here from the repo root; empty folders = real-data gap
  public/<source>/       untouched raw downloads (sroie, cord, in_financial_docs, indian_grocery_gst, funsd, rvl_cdip_160, my_receipts, receiptbench (subset), invoices_receipts_pt (subset), hf_medical_bills)
  synthetic/             generated Indian bills: images/ + annotations.jsonl (exact OCR boxes + fields)
  interim/               derived only: annotations/<document_id>.json (words+bboxes+fields), decoded cord/ and rvl_cdip/ images
  metadata/              manifest_raw -> manifest_dedup -> manifest.jsonl (FINAL, one row per doc), stats, label_overrides, flags
  splits/                train.txt / val.txt / test.txt (document_ids)
  processed/             classification/ document_type/ information_extraction/ {train,validation,test}.jsonl, anomaly/ (empty, see README)
scripts/                 common, download_datasets, generate_synthetic_data, build_dataset, run_ocr, normalize_labels, deduplicate, create_splits,
                         validate_dataset, audit_leakage, acquire_real_foreign, detect_language, pdf_utils, augment (train-time only), sanity_baseline, run_pipeline.sh
```
Original images are never modified or deleted. OCR text, boxes and fields are kept beside every image (`ann_path` in the manifest).

## 2. Canonical record (`datasets/metadata/manifest.jsonl`)
`document_id` (`<source>__<id>`), `source_dataset`, `source_id`, `file_path` (relative to `datasets/`), `document_type`, `expense_category`,
`parent_group`, `subcategory`, `label_quality`, `label_source`, `language`, `country`, `currency`, `synthetic`, `collection`
(public_real / manual_indian / synthetic), `ocr_available`, `ocr_source` (ground_truth / rapidocr / render_ground_truth),
`field_annotations_available`, `ann_path`, `license`, `origin_split`, `group_id`, `sha256`, `width/height`, `split`.
`label_quality`: `manual_folder` (human), `source_label` (dataset authors), `manual_review` (my visual correction), `rule_high` (unambiguous rule),
`synthetic`, `unlabeled`. Task A uses only these trusted values except `unlabeled`.
Annotation file: `{"words":[{"text","bbox":[x0,y0,x1,y1]}], "fields":{...}, "ocr_source":...}` in original-image pixel coordinates.

## 3. Sources: selected (downloaded 2026-10-04 unless noted)

| Dataset | Source | License / terms | Size on disk | Docs | Role | Limitations |
|---|---|---|---|---|---|---|
| SROIE 2019 (scanned receipts) | ICDAR 2019 competition data (zip fetched earlier, 2026-09-20; `sroie-datasetv2.zip`) | no explicit license found in the archive; competition data, treat as research-only | 1.8 GB (zip+extracted) | 973 (958 after dedup) | IE (company/date/address/total) + OCR boxes; weak category via company-name rules | Malaysian retail; 973 docs from only ~296 merchant families (Gardenia, Unihakka... repeat dozens of times); no expense category |
| CORD v2 | HF `naver-clova-ix/cord-v2` | CC-BY-4.0 | 2.2 GB parquet (+0.4 GB decoded) | 1000 (998) | IE (menu, subtotal, tax, total) + word quads | Indonesian restaurant/shop receipts; **no category, no merchant/date fields**; excluded from Task A |
| Indian Grocery Tax Invoice Images | Kaggle `surajitsadhukhan/indian-grocery-tax-invoice-image-dataset` (2026) | CC-BY-4.0 | 652 MB | 169 | **Real Indian GST invoices** -> `miscellaneous/grocery`, doc type invoice | single region (West Bengal), wholesale grocery; no field labels (OCR by RapidOCR) |
| Financial Document Image Dataset (India) | Kaggle `mehaksingal/personal-financial-dataset-for-india` (2023) | DbCL-1.0 on the DB; images scraped from Google Images (copyright unclear -> academic use only) | 61 MB | 426 (390) | Indian bank statements / salary slips / ITR / cheques = `non_expense` negatives; `Utility/` = mixed-country utility bills, relabelled electricity/water/gas by OCR rule (31 kept after dedup: electricity 8, water 16, gas 7; 56 stay unlabeled) | scraped, mixed countries in Utility, 2 corrupt files |
| FUNSD | official https://guillaumejaume.github.io/FUNSD/ (Jaume et al., ICDAR-OST 2019) | research / non-commercial (verify page terms) | 44 MB | 199 | non_expense forms with word boxes | English US scans |
| RVL-CDIP (sample) | HF `nielsr/rvl_cdip_10_examples_per_class` (10/class x 2 splits, 16 classes) | IIT-CDIP; research use | 34 MB | 320 | non_expense negatives (15 classes) + 20 old-US `invoice` (doc-type only) | the full set is ~100 GB; only a sample is needed for negatives |
| Manual Indian bills | your folders -> `datasets/indian/` | owner-collected; redistribution rights unverified | 47 MB | 206 | real Indian: food 69, train 65, fuel 32, parking 22, water 15, internet 2, gas 1 (after review) | 17 independent layouts in `train`; see quality report |
| Synthetic Indian bills | `scripts/generate_synthetic_data.py` (seed 42) | generated here, CC0 | 136 MB | 2250 | 150 per leaf category x 15: GSTIN-valid-format, CGST/SGST/IGST, INR, +91 phones, Indian addresses, exact boxes+fields | `synthetic=true` always; single rendering engine; fictional merchant names; **not** evidence of real-world accuracy |

Citations: SROIE - Huang et al., ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction. CORD - Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing", NeurIPS 2019 Doc-Intelligence WS. RVL-CDIP - Harley, Ufkes, Derpanis, ICDAR 2015. FUNSD - Jaume, Ekenel, Thiran, ICDAR-OST 2019. Grocery - S. Sadhukhan, Kaggle 2026. Financial docs - M. Singal, Kaggle 2023.

## 4. Candidates evaluated and NOT used
| Candidate | Why not |
|---|---|
| Full RVL-CDIP (400k) | ~100 GB vs 8 GB free disk; not needed (negatives only) |
| `devp1866/noisy-medical-document-images-ocr`, `devp1866/high-quality-ocr-ready-invoice-pdfs` | synthetic US hospital bills (CPT/insurance) / one single-seller template of 100 electronics invoices with 10% "VAT" - wrong domain, would skew `medical` / `miscellaneous` |
| `snikhil17/hotel-bills` | 7 web images, license "unknown" |
| HF `mychen76/invoices-and-receipts_ocr_*`, `UniqueData/ocr-receipts-text-detection` | license unstated / CC-BY-NC-ND; mostly US/EU receipts, no categories |
| XFUND, DocVQA | forms/QA, not bills; no category value for this milestone |
| Public Indian hotel/flight/bus/taxi/electricity/medical/rent bill sets | **none found** (Kaggle, HF, GitHub searches; a 2026 HF forum thread is still asking for one). This is the main gap. |

## 5. Pipeline and decisions
1. **Ingest** (`build_dataset.py`): every source -> manifest row + annotation file. Source-provided OCR/boxes kept as ground truth; `run_ocr.py` (RapidOCR) fills images lacking it, tagged `ocr_source=rapidocr`.
2. **Label normalisation**: manual = folder name; SROIE = regex on company field (`rule_high` only for food/fuel/medical/hotel/office-type names, ~290 docs, else `unlabeled`); CORD/SROIE-unmatched/RVL invoices = `unlabeled` (used for Tasks B/C, never Task A); utility bills = OCR keyword rule requiring >=3 hits for one type and 0 for others; `label_overrides.json` = human-reviewed fixes (1 so far: HCG gas invoice found in `water/`).
3. **Task C doc type**: separate field. Public receipts keep `receipt`/`invoice`; manual/synthetic get the default for their category (`scripts/common.py`) - weaker than Task A labels for those sources.
4. **Dedup** (`deduplicate.py`; numbers superseded - see REAL_DATA_ACQUISITION): sha256 exact; near-dup = pHash<=4 OR (pHash<=6 AND dHash<=8) plus same-merchant groups (the first AND-rule left 10 val / 6 test near-twins of train; audit_leakage.py now reports 0), merged into groups, **kept** but forced into one split; SROIE additionally grouped by merchant family (958 docs -> 296 groups); cross-source overlap found: 3 groups where your folders contain images also in the scraped Indian financial set (kept together). Synthetic excluded from perceptual search (shared templates collide by design).
5. **Splits** (`create_splits.py`): 80/10/10, seed 42, assigned to **groups** (largest first), stratified by category x real/synthetic, so real data reaches val/test whenever a class has >=3 groups; classes with <3 groups go all-train (internet). Original benchmark splits are not preserved (`origin_split` kept) - numbers are not comparable to published SROIE/CORD leaderboards.
6. **Leakage guarantees** (hard-checked in `validate_dataset.py`, non-zero exit on failure; independent check in `audit_leakage.py`): hard negatives never carry an expense label; unique `document_id`; no group in >1 split; no identical image bytes in >1 split; valid labels; all files exist. Augmentation is never materialised.
7. **Augmentation** (`augment.py`, apply to TRAIN only inside the data loader): small rotation (+-8), mild perspective, brightness/contrast, soft shadow, blur, low-res, noise, JPEG q35-85, <=6% crop - these mimic phone photos/scans. Rejected: flips (mirrored text never occurs), hue shifts (logo colour is a class cue), large rotations (separate orientation step), over-aggressive cutout (can erase the total). Preview: `python scripts/augment.py` -> `datasets/interim/aug_preview.jpg`.
8. **Imbalance**: Task A is 21% `non_expense`, 12% `miscellaneous`, 4% for gap classes (all synthetic-heavy). Recommended: `class_weight` from `class_distribution.csv` + train-time augmentation; no oversampling file copies. Merge candidates if real data is never collected: `bus+train+flight` -> `intercity_travel` for the first release. Synthetic was generated to fill specific gaps (150/class), not to inflate: real-only categories are never padded beyond that.
