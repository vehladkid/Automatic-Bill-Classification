#!/usr/bin/env bash
# Rebuild the document-type dataset v2 end to end. Run from anywhere:  bash scripts/dataset_discovery/run_all.sh [--download] [--ocr]
#   --download : fetch the verified external sources first (needs network; resumable; checks free disk; ~650 MB)
#   --ocr      : also run the OCR check of the generated documents (needs the project's OCR engine; a few minutes)
set -euo pipefail
cd "$(dirname "$0")/../.."; PY=.venv/bin/python; T=scripts/dataset_discovery
if [[ " $* " == *" --download "* ]]; then
  $PY $T/fetch_verified.py hf microsoft/dit-base-finetuned-rvlcdip --kind models --dest models/pretrained/dit-base-finetuned-rvlcdip --files config.json preprocessor_config.json pytorch_model.bin
  for t in Digital_Type1 Digital_Type2; do $PY $T/fetch_verified.py hf AgamiAI/Indian-Bank-Statements --dest datasets/public/agami_bank_statements --glob "train/India_Bank_Statement_$t/*.pdf" --with-sidecar-json --source-id agami_indian_bank_statements; done
  for t in Scanned_Type1 Scanned_Type2; do $PY $T/fetch_verified.py hf AgamiAI/Indian-Bank-Statements --dest datasets/public/agami_bank_statements --glob "train/India_Bank_Statement_$t/*.pdf" --limit 20 --with-sidecar-json --source-id agami_indian_bank_statements; done
  $PY $T/fetch_verified.py zenodo 8261508 --dest datasets/public/fatura --source-id fatura
fi
$PY $T/import_fatura_subset.py
[[ -f datasets/synthetic_doctype/v1/GENERATOR.json ]] || $PY $T/generate_synthetic_docs.py --out datasets/synthetic_doctype/v1 --n 110 --seed 20261019      # ~5.5 min; deterministic
[[ " $* " == *" --ocr "* ]] && $PY $T/validate_synthetic_ocr.py --per-type 9
$PY $T/build_license_register.py
$PY $T/build_doc_type_manifest_v2.py
$PY $T/validate_doc_type_manifest.py
$PY $T/audit_doc_type_manifest_v2.py > /dev/null
$PY $T/readiness_report.py > /dev/null
echo "done: datasets/metadata/doc_type_manifest_v2.jsonl, reports/doc_type/*, docs/DOC_TYPE_DATASET_READINESS.md"
# --- v2.1 (template-held-out splits, approval gate, audit, baseline); v2 files above are left untouched ---
$PY $T/build_doc_type_manifest_v2_1.py > /dev/null
$PY $T/validate_doc_type_manifest.py --manifest datasets/metadata/doc_type_manifest_v2_1.jsonl --schema v2_1 --tag v2_1 --report reports/doc_type/validation_v2_1.json > /dev/null
$PY $T/audit_doc_type_manifest_v2_1.py > /dev/null
if [[ " $* " == *" --baseline "* ]]; then $PY $T/ocr_for_baseline.py; $PY $T/tfidf_baseline_v2_1.py --final-test > /dev/null; $PY $T/write_v2_1_report.py; fi   # --baseline: OCR of generated docs (~15 min GPU) + TF-IDF experiment; --final-test reads the real test split once
echo "done v2.1: datasets/metadata/doc_type_manifest_v2_1.jsonl, reports/doc_type/{validation,audit}_v2_1.json, docs/DOC_TYPE_V2_1_AUDIT_AND_BASELINE.md"
