# XLM-RoBERTa vs TF-IDF + LR - final study

Questions: **Q1** is XLM-R demonstrably better than the production TF-IDF + LR text classifier on the same documents/splits? **Q2** does any gain justify the cost?
Pre-registration (written before training; with a dated addendum): `PREREGISTRATION.md`. Decision summary: `DECISION_SUMMARY.md`. Audit: `AUDIT.md`. Report: `docs/report/PBL_Report.docx|pdf`, Sections 4.4-4.6 and the Executive Summary.

## Environment (recorded)
Python venv `.venv` (torch 2.9.1+cu128, transformers 5.18.0, scikit-learn 1.9.1, numpy 2.5.2); NVIDIA RTX 4050 Laptop GPU 6 GB (shared with idle demo servers); Intel i5-13450HX; CPU benchmarks use 4 threads. `HF_HUB_OFFLINE=1`. Model: `FacebookAI/xlm-roberta-base`, pretrained safetensors sha256 `6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb` (`tokenizer/`). Each run's `runs/<name>_s<seed>/config.json` stores its exact arguments, seed, times, memory.

## Configuration
Primary (`primary256`): max length 256 incl. 2 special tokens, truncation keeps text start, <=6 epochs, patience 2, AdamW lr 3e-5, wd 0.01, 10% warm-up + linear decay, batch 16, bf16 autocast, class-balanced CE, 250k-row word-embedding matrix frozen (86,054,416 of 278,055,952 params trainable). Checkpoint = best **validation** real-only macro-F1 epoch; test scored once afterwards. Exploratory: `len512`, `chunk256`, `ep10` (selected on seed-0 validation, `configs/selected_variant.json`).
Data: project Task A splits, unchanged (train 4,848 / val 634 / test 629; headline = 386 real test documents, 382 template groups).

## Reproduce
```
cd experiments/xlmr_vs_tfidf_final
python phase1_reconcile.py ; python tfidf_repro.py ; python truncation_audit.py
./run_queue.sh            # primary256 seeds 0-4 + exploratory variants (GPU, ~7 min/run)
./run_extra_seeds.sh      # sensitivity analysis: seeds 5-9
python unseen_synthetic/gen_unseen.py ; python unseen_synthetic/ocr_unseen.py     # synthetic unseen set
python unseen_real/fetch_parallel.py <zenodo 6371710 1_Images.zip URL> unseen_real/raw/1_Images.zip 410222137 ; unzip ...; python unseen_real/build_real_set.py ; python unseen_real/ocr_real.py
python unseen_synthetic/score_unseen.py synthetic ; python unseen_synthetic/score_unseen.py real ; python unseen_stats.py synthetic ; python unseen_stats.py real ; python unseen_extras.py ; python hybrid_eval.py   # hybrid cascade -> outputs/hybrid/hybrid_stats.json
python select_variant.py ; python final_stats.py primary256:0-4 ep10 len512 chunk256 primary256:0-9   # primary256:0-9 writes primary256_seeds0to9_stats.json
python other_models_analysis.py ; python power_analysis.py
./run_bench.sh ; python cost_analysis.py
PYTHONPATH=<matplotlib dir> python final_figures.py
python ../../scripts/make_pbl_report.py                       # DOCX + PDF
python verification/verify_final.py ; python verification/verify_report_final.py
```
Not bit-for-bit reproducible: GPU kernels/bf16 are non-deterministic; reruns with the same seed will differ slightly (reported seed spread is the relevant uncertainty).

## Artifact inventory
`runs/*/` per-run config, history (per-epoch curves), logits (val+test), fp16 trainable-weight checkpoint, log; `outputs/tfidf/` refit baseline + predictions; `outputs/stats/` per-variant statistics, per-seed CSV, bootstrap distributions; `outputs/resources/` raw benchmarks; `outputs/*.json` reconciliation, truncation, other models, power, cost; `figures/final_fig1-7`; `verification/`.

## Known limitations / deviations (all disclosed in the report)
* Checkpoints are fp16 files of the trainable parameters only (disk was ~1-3 GB free); the frozen embedding matrix is the pretrained one (hash above). A reload check against stored logits is in `outputs/resources/xlmr_primary256_s0.json`.
* The original single-run XLM-R from `experiments/text_vs_neural` has no saved checkpoint; it is compared only via its stored predictions.
* `unseen_synthetic/`, `unseen_real/`: unseen test sets (see PREREGISTRATION addenda 2-3); `raw/` holds 410 MB of downloaded third-party images (CC BY 4.0, Cruz & Castelli, Zenodo 6371710) and can be deleted.
* Primary = 5 seeds (pre-registered). Seeds 5-9 are a **secondary sensitivity analysis** run after the primary result was known (addendum); they cannot change the primary verdict.
* Exploratory variants other than `ep10` are single-seed; `len512` single-seed CI verdict is borderline (the one failing check in `verify_final.py`, documented).
* LayoutLMv3 GPU batch-32 benchmark OOMs on the shared 6 GB GPU (recorded as OOM).
* Pre-study report backup `docs/report/backup_20261027_223641_pre_final_study` was reconstructed (generator was edited before the backup was taken); earlier backups are untouched.
* Latency/memory measured on a shared laptop; training times include contention with idle servers.
* Historical (not re-measured) values are labelled historical (BiLSTM, LayoutLMv3 training time).
