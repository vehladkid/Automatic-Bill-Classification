> **Update 2026-10-05: Task A is implemented and evaluated - see [TASK_A_REPORT.md](TASK_A_REPORT.md)** (text, image and LayoutLMv3 models, calibrated fusion, HR-routing threshold, label audit). Below is the original plan.

# Training recommendations

**Status: ready to start Task A (expense classification) and Task C (document type); Task B (IE) is ready for receipts and synthetic Indian bills; Task D not built.**
Caveat first (updated 2026-10-04): all classes except `rent` now have real documents (see REAL_DATA_ACQUISITION_2026-10-04.md). Evaluate real-only and synthetic-only separately; `rent` is still synthetic-only; weak real test sets (4-18 docs) need group cross-validation and confidence intervals. A GPU (RTX 4050, 6 GB) is now available: use mixed precision, batch ~8-16 at 384-512 px for CNN/ViT, LayoutLMv3-base fits with fp16 and gradient accumulation.

## Inputs
* Read `datasets/processed/classification/{train,validation,test}.jsonl`: `image` (path rel. to `datasets/`), `ocr_text`, `label`, `parent_group`, `synthetic`, `source`, `ann_path` (words + boxes for layout models).
* Labels: 16 leaves in `scripts/common.py: CATEGORIES`. Use `class_weight` from `datasets/metadata/class_distribution.csv`.
* Apply `scripts/augment.py: augment()` to train samples only.

## Models, in order
1. **Baseline (hour 1)**: TF-IDF (char 2-4 + word 1-2) + LogisticRegression on `ocr_text`. Already reproduced in `scripts/sanity_baseline.py` (real-test acc 0.94). Cheap, gives the floor and exposes label noise.
2. **Image baseline**: fine-tune a small CNN/ViT (e.g. EfficientNet-B0 / ConvNeXt-Tiny, 384px) on images, with augmentation + class weights. Tells you how much layout alone carries.
3. **Stronger**: multimodal LayoutLMv3-base (text + boxes + image, boxes in `ann_path`, normalise to 0-1000) for classification; same backbone gives Task B token classification (fields: company/date/address/total from SROIE, total/subtotal/tax from CORD, plus synthetic Indian fields incl. GSTIN/CGST/SGST). Alternative OCR-free: Donut. Needs a GPU (this machine has none).
4. Late-fusion of 1+2 before reaching for 3 if GPU time is scarce.

## Metrics
Macro-F1 (primary), per-class recall, confusion matrix; report **real-only** and **synthetic-only** separately, never pooled. Task B: field-level exact match / token F1 per field. Add calibration (ECE) because the product routes low-confidence claims to HR. For gap classes report "n_real_test" beside every number.

## Order of work
1. Add real bills for the empty folders (>=30 each, varied vendors; hotel, medical, electricity, flight, taxi first) and re-run `bash scripts/run_pipeline.sh`.
2. Baseline 1 -> audit errors with `label_review_flags.json` -> fix labels via `label_overrides.json`.
3. Image model -> compare. 4. LayoutLMv3 multimodal + IE heads. 5. Calibration and thresholds for auto-approve vs HR review.
6. Build Task D from train-only docs (see anomaly README).
Never tune on test; val for thresholds; touch test once per model family.
