#!/usr/bin/env bash
# Rebuild everything from raw. Run from repo root: bash scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"; PY=../.venv/bin/python
$PY download_datasets.py              # idempotent
$PY acquire_real_foreign.py           # diverse subsets of ReceiptBench + InvoicesReceiptsPT (remote range reads; ~25 min)
$PY generate_synthetic_data.py        # deterministic (seed 42)
$PY build_dataset.py                  # raw -> datasets/metadata/manifest_raw.jsonl + interim/annotations
$PY run_ocr.py                        # OCR for real images lacking GT OCR (GPU via onnxruntime-gpu==1.22.0, ~170 img/min; resumable)
$PY detect_language.py                # langdetect where the source states no language
$PY normalize_labels.py               # OCR-keyword labels for unlabeled utility bills
$PY deduplicate.py                    # sha256 + pHash/dHash + merchant groups -> manifest_dedup.jsonl
$PY create_splits.py                  # group-stratified split -> manifest.jsonl, datasets/splits/*.txt
$PY validate_dataset.py               # task views in datasets/processed/, stats, hard leakage checks
$PY audit_leakage.py                 # independent pHash leakage audit -> datasets/metadata/leakage_audit.json
$PY export_indian_view.py       # copies all Indian docs into datasets/indian_all/{real,synthetic}/<category>/ + index.csv
