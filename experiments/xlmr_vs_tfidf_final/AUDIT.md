# Audit note - XLM-R vs TF-IDF final study

## Verified (artifact exists and was re-checked)
* 10 completed XLM-R runs (primary256 seeds 0-9) + exploratory runs; every table in report Section 4.5 is compared cell-by-cell with outputs/*.json by `verification/verify_report_final.py` (82 checks); statistics re-derived by `verification/verify_final.py`.
* TF-IDF refit reproduces the production model to 5.8e-14.
* Unseen sets: 576 synthetic documents (OCR'd, scored once); 336 real documents (Zenodo 6371710, 813 images downloaded, excluding the project's images and companies, of 813 images, 81 were already used by the project, 58 had a blank company, 48 shared a company with the project subset and 290 matched no labelling rule; 336 were kept).

## Deviations / disclosures
* Seeds 5-9 were run AFTER the five-seed result was known (disk space was freed); declared in the pre-registration addendum; secondary only.
* Pre-study report backup was reconstructed (generator edited before backup was taken); a fresh backup `docs/report/backup_*_pre_unseen_sets` was taken before the unseen-set update.
* 6 of 192 noisy synthetic documents have empty OCR; kept as pre-registered.
* The synthetic long-preamble condition does not isolate truncation (boilerplate drives both models to non_expense).
* Real-set labels are silver (keyword rules on company names, written before scoring); only 2 classes have >= 10 documents; Portuguese language; groceries labelled miscellaneous (project convention).
* Checkpoints are fp16 trainable-weights only; original single XLM-R run has no checkpoint.

## Added later
* Hybrid cascade (TF-IDF first, XLM-R for uncertain documents), pre-registered in Addendum 4 before scoring; thresholds chosen on validation only; hybrid seed-0 result and threshold independently recomputed in verify_report_final.py. Its cost is an ESTIMATE from measured component latencies, not a measurement of the combined system.

## Not done
* No quantisation/ONNX latency, no human-annotated real unseen set, no ablation of why TF-IDF falls back to 'food' on shifted wording (hypothesis only).
* `verification/verify_final.py` single known borderline check: len512 single-seed interval (exploratory).

## Report format and literature check (2026-10-28)
* Format: faculty template sections kept (1-6); IEEE conventions applied (single abstract paragraph with "Abstract-" lead and "Index Terms", roman-numbered table captions above tables, "Fig. n." captions below figures, IEEE-style numbered references, no bullet points, abbreviations spelled out; metric abbreviations CER/WER/ECE/CI/LR are never used, others are expanded at first use). The layout stays single-column (the template is single-column); it is not an IEEE two-column conference layout.
* References: all 27 entries were checked against Crossref (DOI) or the arXiv API (see verification/references_check.txt, verification/verify_references.py). Journal papers from 2021-2025: [4], [5], [8], [16]-[27] (15 papers). Two candidate journal papers were dropped because their content could not be verified (Limam et al. 2025, Engineering Applications of AI; Ren et al. 2021, Neurocomputing). [24] (survey, online-first) is described from its title and venue only; [23] is described from its publicly available abstract summary.
* Figures of Section 4.4 were redrawn with spelled-out labels by scripts/relabel_comparison_figures.py into docs/report/figures_spelled/; experiments/text_vs_neural is unchanged (results.json hash identical).
