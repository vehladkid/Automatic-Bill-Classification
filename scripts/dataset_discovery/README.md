# Document-type dataset tooling (schema `doc-type-schema-v2`)

Builds, validates and audits a **document-type corpus** for Indian employee-expense, financial and tax-evidence documents. It is separate from Task A (16 expense categories) and Task B (field extraction) and never modifies their data, splits, snapshots or models.
Three label axes are kept apart (`config/doc_type_schema_v2.json`): **A document type**, **B expense category** (Task A, unchanged), **C tax evidence** (derived; eligibility is always `pending_verification`).

## Rebuild
```bash
bash scripts/dataset_discovery/run_all.sh              # offline rebuild from what is already on disk (generation ~5.5 min the first time, deterministic)
bash scripts/dataset_discovery/run_all.sh --download   # first fetch the verified external sources (~650 MB, resumable, refuses to use >60% of free disk)
bash scripts/dataset_discovery/run_all.sh --ocr        # also OCR-check the generated documents
.venv/bin/python -m unittest discover -s tests -p "test_dataset_discovery.py"
```
## Tools
| script | purpose |
|---|---|
| `hf_probe.py` | research capture: search/probe Hugging Face datasets & models and Zenodo records (licence, gating, size, files, retrieval date) -> `datasets/metadata/external_catalog.json` |
| `fetch_verified.py` | resumable downloader (IPv4), sha256/md5 verification, provenance log -> `datasets/metadata/provenance_external.jsonl` |
| `import_fatura_subset.py` | template-balanced FATURA subset (200 of 10,000 synthetic invoices) |
| `generate_synthetic_docs.py` | **synthgen-v1**: 11 Indian document types x 4 layouts x 3 image-quality modes (+ Hindi headers), fictional data, structured labels, deterministic by (seed, type, index) |
| `validate_synthetic_ocr.py` | OCR sanity check: is the class-defining text / main amount readable in the rendered images |
| `build_license_register.py` | curated licence + evidence register (`license_register_v2.json`); every manifest source must have an entry |
| `build_doc_type_manifest_v2.py` | unified manifest + real/synthetic manifests + split files + stats |
| `make_doc_type_splits.py` | per-split manifests; real and synthetic kept in separate files |
| `validate_doc_type_manifest.py` | schema/labels/licences/files/checksums/separation/grouping/generator-consistency checks (exit 1 on any error) |
| `audit_doc_type_manifest_v2.py` | byte-duplicate, group-leakage, pHash near-duplicate and cross-domain audit |
| `readiness_report.py` | coverage + training readiness -> `reports/doc_type/readiness_v2.json`, `docs/DOC_TYPE_DATASET_READINESS.md` |
| `build_doc_type_manifest.py`, `audit_doc_type_manifest.py` | v1 (kept unchanged for reproducibility) |
| `probe_text_baseline.py`, `probe_dit_zeroshot.py` | cheap feasibility probes from the earlier research round (validation split only) |

## Where things live (large data is not meant for Git)
`datasets/synthetic_doctype/v1/` generated images + `ann/` labels + `GENERATOR.json` (seed, script sha256, fonts) | `datasets/public/agami_bank_statements/`, `datasets/public/fatura/` verified external data | `datasets/metadata/doc_type_*` manifests |
`reports/doc_type/` validation, audit, OCR check, readiness | `models/pretrained/dit-base-finetuned-rvlcdip/` backbone weights (no training done).

## Rules this tooling enforces
- Original source labels are preserved next to the mapped label; mapping is by rule only (no visual re-labelling) and is flagged `needs_mapping_review` where the label comes from a collection rule.
- Real documents inherit the existing train/val/test split; the protected Task A/B test documents stay test-only (`in_protected_task_test`). Synthetic documents are `synthetic=true`, never eval-eligible, grouped by issuer/template, and reported separately - **never mix real and synthetic metrics**.
- Generated layout **D** is test-only: it is the only honest generalization slice for synthetic classes (layouts A-C are shared between splits by construction; the audit quantifies this).
- Synthetic documents are fictional (names/issuers checked against a real-brand deny-list; identifiers are random format-valid strings) and every page carries a "SYNTHETIC SAMPLE" footer. They support development only, never claims about real-world accuracy.
- No personal documents or extracted private values are stored in the repository tooling; no paid API or LLM is used.

## v2.1 additions (audit + corrected splits + baseline)
`build_doc_type_manifest_v2_1.py` (template-held-out generated splits, Agami template split, approval gate from `config/approved_sources_v1.json`, subtype-aware tax candidates, OCR-cue review queue) -> `validate_doc_type_manifest.py --schema v2_1 --tag v2_1` -> `audit_doc_type_manifest_v2_1.py` (independent audit) -> `ocr_for_baseline.py` -> `tfidf_baseline_v2_1.py [--final-test]` -> `write_v2_1_report.py` (renders `docs/DOC_TYPE_V2_1_AUDIT_AND_BASELINE.md`). v2 files are never rewritten. Use only `datasets/metadata/doc_type_v2_1_approved/` for training; `in_protected_task_test` rows are test-only.
