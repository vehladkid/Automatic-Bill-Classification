# Dataset and model research: Indian expense and tax-document classification

Date: 2026-10-19. Every fact below was read from the primary source named next to it (Hugging Face Hub API / dataset or model card, Zenodo record API, project files) on the retrieval date; items I could not verify are marked **UNVERIFIED**. Nothing was downloaded except what section 6 lists.
Machine-readable catalog of everything probed: `datasets/metadata/external_catalog.json` (45 records, retrieval-dated). Tools: `scripts/dataset_discovery/`.

## 0. Bottom line
1. **No public dataset found with labelled EV, Sodexo/meal-voucher, rent-receipt/HRA, school-fee, 80C, 80D or loan-certificate documents.** Hub name searches ("rent receipt", "salary slip", "form 16", "insurance premium", "school fee", "loan", "tuition", "voucher", "indian invoice", "gst invoice") returned nothing relevant; web searches (Kaggle, GitHub, Zenodo, papers) found no such dataset either, and a Hugging Face forum thread shows others are asking for the same thing ([Seeking Indic document dataset](https://discuss.huggingface.co/t/seeking-indic-document-dataset-india-invoices-receipts-utility-bills-payment-advices-packing-lists-commercial-invoices-credit-notes/177055)). Kaggle and Roboflow Universe need accounts and were not searched exhaustively - absence there is **not** proven.
2. The only labelled tax/financial documents available are already in the project: `in_financial` (Form 16 x37, Salary Slip x89, Bank Statement x85, Cheque x92, Utility x87), country tagged `mixed_unverified`, licence "DbCL-1.0 database; underlying web images academic use only".
3. A dedicated document-type classifier should **complement** Task A and the profile rules, not replace them (section 5). Training it for the seven requested categories is **blocked by missing labelled data**, not by models or compute.
4. Built: a versioned schema, a normalized doc-type manifest (8,423 docs, inherited splits), a verified downloader, a leakage audit (clean), and two cheap feasibility probes. No model was trained or fine-tuned; no existing model, split, snapshot or V2 file was changed.

## 1. What the project already has
| Capability | Status |
|---|---|
| Task A: 16 expense categories | **trained** (3 models + calibrated fusion) |
| Task B: field extraction | **trained** (V1 frozen) |
| Document type (invoice/receipt/utility/...) | **labels only** (dataset `document_type`); never trained (Task C was removed from scope) |
| EV, Sodexo, HRA/rent, school fee, 80C, 80D, loan documents | **rules only** (`config/taxonomy.json`, `app/doc_profile.py`) - not a trained classifier; 0 real labelled examples |
| Missing labels | all seven requested categories; real **Indian** examples of Form 16, salary slip, bank statement, cheque (country unverified) |

**Taxonomy decision.** The 16 expense categories cannot hold the new requirements without conflating concepts: a school-fee receipt is expense category `miscellaneous`; an insurance premium receipt may be `medical` or `miscellaneous`; Form 16 and salary slips are not expenses at all. `config/doc_type_schema_v1.json` therefore separates (A) document type (single-label, hierarchical, with `needs_review`/`unknown`), (B) expense category (Task A, unchanged), and (C) tax evidence (multi-label, **derived** from type + rules + employee data, always `pending_verification`). A second, overlapping taxonomy was not created: `config/taxonomy.json` stays the rule layer; the new schema is the label space for a future trained classifier and the evaluation manifest.

## 2. Dataset inventory (ranked by usefulness for this goal)
Labels are document-level unless stated. "Use" = what the dataset can legitimately support here.
| # | Dataset (source) | Documents and labels | Country / language | Images / OCR | Splits | Licence (as stated by source) | Size | Use |
|---|---|---|---|---|---|---|---|---|
| 1 | **in_financial** (project, `datasets/public/in_financial_docs`; original public source URL not recorded in the project - **provenance gap**) | 390: Check 92, Salary Slip 89, Bank Statement 85, Utility 87, ITR/Form 16 37 (counts from `manifest.jsonl`) | `mixed_unverified`, en | images; OCR run by project | inherited 305/46/39 | "DbCL-1.0 (database); underlying web images - academic use only" (project record) | local | train + eval of tax/financial **types**; not Indian-verified; academic-only |
| 2 | Project Indian bills (manual_in 368, in_grocery_gst 169, receiptbench-IN 285) | receipts, invoices, utility bills, travel, hotel, fuel; labels by folder/source | IN | images + OCR | inherited | manual_in: "redistribution rights unverified"; grocery GST CC-BY-4.0; receiptbench MIT, **per-image consent not documented** | local | train + eval of commercial-bill types; Indian |
| 3 | [AgamiAI/Indian-Bank-Statements](https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements) | **synthetic** business current-account statements; 4 template folders x 100 PDFs + JSON (Digital Type1/2 ~20 MB each, Scanned Type1/2 ~660 MB each) - verified from Hub file listing | IN, en/Hindi (romanized) | PDF (+JSON ground truth) | `train` only | Apache-2.0 (card) | 1.37 GB full; **294 MB downloaded** (200 digital + 20 + 20 scanned PDFs, with JSON) | train-only bank_statement augmentation; never an eval set (single generator, 4 templates) |
| 4 | [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | receipts, **field-level** labels | Indonesia | image + parse | train/val/test | CC-BY-4.0 (card) | 2.3 GB (already local) | extraction; its documents are all "receipt" - gives no type contrast |
| 5 | [ICDAR2019-SROIE (HF re-release)](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) | 987 receipts (626 train / 361 test), field-level | Malaysia, en | image + text | train/test | **conflict**: card says CC-BY-4.0 citing the ICDAR RRC site; project recorded "research use". Primary RRC page not fetched - **UNVERIFIED** | 0.5 GB (local) | extraction; receipts only |
| 6 | [FUNSD (nielsr/funsd)](https://huggingface.co/datasets/nielsr/funsd) | 199 forms (149/50), token labels HEADER/QUESTION/ANSWER (field-level) | US, en | image + words | train/test | none on the HF card; original recorded as non-commercial research by the project - **UNVERIFIED here** | 17 MB | form-understanding; no document-type labels |
| 7 | [RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) | 400,000 grayscale images, 16 classes x 25,000: letter, form, email, handwritten, advertisement, scientific report, scientific publication, specification, file folder, news article, budget, invoice, presentation, questionnaire, resume, memo (card + model config); **25,000 `invoice` documents**; document-level labels | US (tobacco-litigation scans), en | images | official 320,000 / 40,000 / 40,000 (card) | `other` (derived from IIT-CDIP; research use) | **38.8 GB** - cannot fit (6.1 GB free); 320-doc subset local | general doc-type pretraining; only `invoice` overlaps our types; US scans, not Indian |
| 8 | [DocLayNet v1.1](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1) | 80,863 pages; **layout boxes** (11 classes); 6 document categories (finance, science, patents, tenders, law, manuals) - not our types | language not verified | images + boxes | train 69,375 / val 6,489 / test 4,999 pages | CDLA-Permissive-1.0 (card text; Hub metadata field says "other") | 31.8 GB | layout pretraining only; not feasible or needed |
| 9 | [FATURA](https://zenodo.org/record/8261508) | 10,000 **synthetic** invoice images, **50 templates x 200**, 24 box classes | en | jpg + json (3 formats) | no splits in the record (it proposes two training/evaluation protocols) | CC-BY-4.0 (Zenodo API) | 363.5 MB, md5 published | invoice-only; template leakage risk (must group by template); not downloaded: adds nothing for the new classes |
| 10 | [Voxel51 high-quality-invoice-images-for-ocr](https://huggingface.co/datasets/Voxel51/high-quality-invoice-images-for-ocr) | 8,181 **synthetic** invoices (1,489 fully annotated) | en | jpg | none | ODbL (card) | 2.4 GB | invoice-only; not downloaded |
| 11 | [drew-ipp/invoice-extraction-benchmark](https://huggingface.co/datasets/drew-ipp/invoice-extraction-benchmark) | 181 **synthetic** invoices + answer keys | en | pdf/jpg | test only | CC-BY-4.0 (card) | 37 MB | extraction test only |
| 12 | [DocILE](https://docile.rossum.ai/) | 106,680 docs (6,680 annotated + 100k synthetic) + ~1M unlabeled; invoices/orders | en (languages not stated) | pdf | benchmark splits | code MIT; **data via access-request form, academic use** | n/a | **blocked: needs the form; not bypassed** |
| 13 | [buthaya/payslips](https://huggingface.co/datasets/buthaya/payslips) | 611 pages, 9 entity types, **text+layout only (no images)**, insurance domain | country not stated | no images | 485/126 | MIT (card) | 8 MB | NER only; unsuitable for image classification |
| 14 | [nutrientdocs/document-classification-benchmark](https://huggingface.co/datasets/nutrientdocs/document-classification-benchmark) | 1,274 open-vocab docs, test only | en | images | test | `nutrient-eval-restricted` | 0.45 GB | **restricted - not usable for training** |
| - | WildReceipt, ReceiptSense, Kleister, Kaggle/Roboflow sets, Zindi Tunisia bill classification | seen in search results only | - | - | - | **UNVERIFIED** (WildReceipt page gave no licence; others not fetched; Kaggle/Roboflow/Zindi need accounts) | - | not evaluated |

Per-category document counts for the **requested** types are in section 4 (all zero).

## 3. Pretrained models (verified from the Hub API / config.json)
| Model | Architecture / task | Labels | Licence | Weights | Inputs | Local? | 6 GB GPU alongside pipeline |
|---|---|---|---|---|---|---|---|
| [microsoft/dit-base-finetuned-rvlcdip](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip) | DiT-base image classifier | 16 RVL-CDIP classes | MIT | 343 MB | image only | **downloaded + sha256 verified** | measured: 5.5 ms/image, 193 MB GPU |
| microsoft/dit-large-finetuned-rvlcdip | DiT-large | 16 | MIT | 1.2 GB | image | no | likely, not tested |
| microsoft/dit-base | self-supervised DiT backbone | none | MIT | 369 MB | image | no | fine-tuning backbone candidate |
| naver-clova-ix/donut-base-finetuned-rvlcdip | OCR-free encoder-decoder | RVL-CDIP via generation | MIT | 803 MB | image | no | heavier; not tested |
| **microsoft/layoutlmv3-base** | multimodal (text+layout+image) | none (backbone) | **CC-BY-NC-SA-4.0 (non-commercial)** | 501 MB | OCR words + boxes + image | **local; already used by Task A and Task B** | proven (Task A trains/infers on it) |
| microsoft/layoutlm-base-uncased | text+layout (no image) | none | MIT | 453 MB | words + boxes | no | likely |
| impira/layoutlm-invoices | doc-QA on invoices | n/a | CC-BY-NC-SA-4.0 | 511 MB | image/words | no | extraction, not classification |
| HAMMALE/vit-tiny-classifier-rvlcdip | ViT-tiny, 6M params | 16 RVL-CDIP | Apache-2.0 | 22 MB | image | no | trivial |
| fedihch/InvoiceReceiptClassifier (+_LayoutLMv3) | image classifier | **20 labels named LABEL_0..19 (undocumented)** | none / "other" | 802 / 501 MB | image | no | **unusable: unnamed labels, unclear licence** |
| distilbert/distilbert-base-uncased, FacebookAI/xlm-roberta-base, microsoft/deberta-v3-small, google/muril-base-cased, ai4bharat/IndicBERTv2-MLM-only | text encoders (pretraining only) | none | Apache-2.0 / MIT | 268 MB / 1.1 GB / 286 MB / 953 MB / 1.1 GB | OCR text | no | text classification candidates; IndicBERT/MuRIL only matter if documents contain Indic scripts |

**Flag:** the LayoutLMv3 weights the project already depends on are **non-commercial** (CC-BY-NC-SA-4.0). That constrains any commercial deployment of Task A/B and of a LayoutLMv3-based document-type model; MIT-licensed alternatives (DiT, LayoutLM-v1, text encoders) avoid it.

## 4. Coverage matrix (from `datasets/metadata/doc_type_manifest_v1_stats.json`; real docs train/val/test; synthetic separately)
| Requested category -> document type | Real train/val/test | Real Indian | Synthetic | Verdict |
|---|---|---|---|---|
| EV expense/benefit -> `ev_document` | 0/0/0 | 0 | 0 | **no data** |
| Sodexo / meal voucher -> `meal_voucher_statement` | 0/0/0 | 0 | 0 | **no data** |
| HRA / rent receipt, rental agreement -> `rent_receipt`, `rental_agreement` | 0/0/0 | 0 | 0 (the 150 synthetic "rent" documents are rent *invoices*, not receipts) | **no data** |
| School-fee receipt | 0/0/0 | 0 | 0 | **no data** |
| 80C evidence -> `investment_certificate` | 0/0/0 | 0 | 0 | **no data** |
| 80D / insurance premium -> `insurance_premium_receipt` | 0/0/0 | 0 | 0 | **no data** |
| Loan principal/interest -> `loan_interest_certificate` | 0/0/0 | 0 | 0 | **no data** |
| Form 16 | 27/8/2 | 0 verified | 0 | too few real examples |
| Salary slip | 67/10/12 | 0 verified | 0 | too few real examples |
| Bank statement | 70/5/10 | 0 verified | 240 (train-only) | too few real examples |
| Cheque | 76/10/6 | 0 verified | 0 | too few real examples |
| Invoice | 200/45/49 | 99/31/39 | 300 | trainable + evaluable |
| Receipt | 2,724/317/314 | 138/23/29 | 450 | trainable + evaluable (generic and Indian) |
| Utility bill | 281/35/34 | 148/15/20 | 600 | trainable + evaluable |
| Travel document | 442/62/53 | 144/20/8 | 450 | trainable; Indian test too small |
| Hotel bill / fuel receipt / medical bill | 177/22/22, 112/13/15, 26/4/4 | partial | 150 each | too few val/test or real docs for per-class claims |
| Other | 402/47/50 | 0 | 180 | trainable + evaluable |
Policy used for "trainable/evaluable": >=150 real train, >=30 real val and test per class, >=30 Indian docs for Indian claims (a project policy, not a statistical guarantee). Class imbalance: receipt 3,355 vs medical bill 34 real docs (ratio 98.7:1).

## 5. Should a dedicated document-type classifier be built?
Yes, **as a complement**: Task A stays the expense-category model; the document-profile rules stay as the explainable evidence layer; a trained document-type classifier would supply a calibrated type probability that the rules and the LLM-as-adjudicator policy can use. It must not replace Task A, and a high score on RVL-CDIP or invoice-vs-receipt would say nothing about HRA/80C/80D.

## 6. What was acquired and where
| Item | Location | Size | Integrity |
|---|---|---|---|
| DiT-base-finetuned-rvlcdip weights/config | `models/pretrained/dit-base-finetuned-rvlcdip/` | 343 MB | sha256 verified against the Hub's LFS checksum |
| Indian synthetic bank statements (240 PDFs + 240 JSON) | `datasets/public/agami_bank_statements/` | 294 MB | PDFs sha256 verified; JSON size-only (no checksum published) |
| Provenance (source, revision, licence, checksum, time) | `datasets/metadata/provenance_external.jsonl` (483 records: 241 sha256-verified, 242 size-only) | | |
| Catalog of 45 probed datasets/models | `datasets/metadata/external_catalog.json` | | |
Deliberately **not** downloaded: RVL-CDIP (38.8 GB), DocLayNet (31.8 GB), FATURA, Voxel51 invoices, other models. Free disk after acquisition: ~6 GB.
Tooling (`scripts/dataset_discovery/`): `hf_probe.py` (search/probe), `fetch_verified.py` (resumable, IPv4, checksum + provenance, refuses to use >60% of free disk; also supports Zenodo with md5), `build_doc_type_manifest.py`, `audit_doc_type_manifest.py`, `probe_text_baseline.py`, `probe_dit_zeroshot.py`.

## 7. Normalized manifest and leakage audit
`datasets/metadata/doc_type_manifest_v1.jsonl`: 8,423 rows; each keeps the original labels, mapping rule, label basis (`source_dataset_label` / `collection_rule` / `synthetic_generator`), country, licence, group id, inherited split, `eval_eligible` (real documents only), derived tax-evidence candidates and `eligibility_status`. Splits are **inherited** from the existing manifest, so the frozen test split is unchanged; external synthetic data is train-only. Label mapping is by source label only - **the mapping still needs a human review** before training (`mapping_rule`/`label_basis` columns make that review easy; notably the `receipt` class is "whatever receipt datasets contain", so its label basis is the collection, not an annotation of document type).
Independent audit (`reports/doc_type/leakage_audit_v1.json`): 0 byte-identical duplicates (0 across splits, 0 across sources); 0 group ids spanning splits; 0 undecodable images; pHash nearest-train-neighbour for real val/test docs: **0 at distance <=4** (val 578, test 571), 29 val (5.0%) and 37 test (6.5%) at <=8 ("similar template"); the 240 synthetic bank statements are >=12 bits from any real val/test document (0 at <=8).

## 8. Feasibility probes (cheap, no model files saved, TEST never read)
1. **Text probe** (`reports/doc_type/text_probe_val_v1.json`): TF-IDF + logistic regression on OCR text, train 4,599 / val 576 real docs, 12 types: accuracy 0.934, macro-F1 0.923 (0.925 over the classes with >=30 val docs); Indian val (102 docs, 6 types) accuracy 0.941; 7.8% of docs below 0.5 confidence. **Read with caution:** receipts are 55% of val; bank_statement, cheque, form_16, salary_slip, medical_bill, fuel_receipt have 4-13 val docs (marked untrustworthy in the report); most classes come from one source, so the probe may partly learn source style (largest confusion: travel_document -> receipt, 11 docs). It shows the labels are separable, **not** that the model generalizes or recognizes HRA/80C/80D documents.
2. **DiT zero-shot** (`reports/doc_type/dit_zeroshot_v1.json`, 465 images, 40 per type): 5.5 ms/image, 193 MB GPU. It labels only 45% of true invoices as `invoice` (18/40), and also 45% of receipts, 42% of salary slips, 28% of bank statements and 45% of cheques as `invoice`; Form 16 mostly `form` (26/35). An off-the-shelf RVL-CDIP classifier cannot distinguish our document types; it is only a fine-tuning backbone.

## 9. Proposed training and evaluation design
Comparison: (1) **image-only DiT fine-tune** - MIT licence, 343 MB, fits easily, no OCR needed, but needs many images per class and is blind to the keywords ("RENT RECEIPT", "Form 16", "premium") that identify these types; (2) **OCR-text classifier** - strongest signal for these types, CPU-friendly, OCR already runs in the pipeline, start from the TF-IDF probe and, if data allows, DistilBERT (Apache-2.0) or a multilingual encoder when Indic scripts appear; (3) **multimodal LayoutLMv3** - likely best accuracy and already proven on this GPU, but the weights are non-commercial.
**Recommendation for the next classifier:** an OCR-text document-type classifier (TF-IDF baseline -> small transformer), evaluated against the existing rule layer, with DiT image features as an optional second head later. Reason: cheapest, license-clean, explainable, and the type signal is textual.
**Gates before any training run** (status): licences verified for selected sources (done for selected; in_financial remains academic-only) / label mapping reviewed by a human (**pending**) / duplicate and leakage audit passes (**done, clean**) / manifests generated (**done**) / enough labelled examples (**NOT met for all seven requested categories**).
**Data needed (project policy targets):** >=150 train + >=30 val + >=30 test real documents per requested type (about 210 x 7 = 1,470), with >=30 Indian documents per type, grouped by issuer/template so one issuer never spans splits; double-annotate 20% for agreement; collect only authorized, anonymized documents (employees' own receipts with consent) - no personal bills in the repository. Synthetic fixtures may support rule/pipeline tests and train-only augmentation but **cannot** support accuracy claims.
**Evaluation:** grouped splits; per-class precision/recall/F1 with support, macro-F1, confusion matrix, calibration (ECE) once supports allow, manual-review rate at the routing threshold, bootstrap CIs; Indian and generic documents reported separately; RVL-CDIP/invoice-vs-receipt results reported separately and never as evidence for tax-class accuracy; independent test set used once; the new model enters production inference only after an independent evaluation and the existing regression tests.

## 10. Blockers and open items
- **No labelled data** for EV, Sodexo, HRA/rent, school-fee, 80C, 80D, loan documents (and few real Form 16 / salary slips / bank statements / cheques, none verified Indian).
- `in_financial` licence is academic-only, its country is unverified and its original download URL is not recorded in the project.
- SROIE licence conflict (CC-BY-4.0 per HF card vs "research use" per project record); FUNSD licence not on the card.
- LayoutLMv3 weights are non-commercial.
- DocILE needs an access-request form; Kaggle, Roboflow and Zindi need accounts - not attempted.
- Disk: ~6 GB free; network ~1.4 MB/s with no IPv6 route (downloader forces IPv4 and resumes).
- Human review of the label mapping is outstanding.


---
# Update 2 (2026-10-19, build mission): what was actually built, downloaded and generated

**Research round 2 (primary sources).** Zenodo API searches ("indian invoice", "receipt dataset", "invoice dataset", "bank statement", "rent receipt", "insurance premium receipt"), GitHub repository searches and further Hub searches again found **no real dataset for EV, Sodexo/meal-voucher, rent/HRA, school-fee, 80C, 80D or loan documents**. New facts: FATURA exists as Zenodo 8261508 (363.5 MB, CC-BY-4.0, md5 published; a newer record 10371464 is listed at 691 MB); Zenodo 6371710 / 7213544 are the Portuguese invoice sets already partly in the project (CC-BY-4.0); MIDD (Zenodo 5113009) is a 1 MB archive of IOB text annotations without images; the only GitHub hits were generator repos, one of which depends on a paid image API (not used). Full catalog: `datasets/metadata/external_catalog.json` (50 records). Licence evidence per source: `datasets/metadata/license_register_v2.json` (24 entries: 8 usable, 5 usable with restrictions, 1 conflict, 2 unverified, 3 blocked, 5 looked at and not acquired).

**Downloaded (verified):** FATURA archive 363.5 MB (md5 verified) -> 200-image template-balanced subset imported; AgamiAI Indian bank statements: 240 PDFs + 240 JSON (294 MB, PDFs sha256-verified); DiT-base weights (343 MB). Provenance: `datasets/metadata/provenance_external.jsonl`.
**Generated:** `synthgen-v1`, **1,210 documents** (110 per type) for 11 types: rent_receipt, rental_agreement, insurance_premium_receipt (health/life), school_fee_receipt, investment_certificate (PPF/ELSS/tax-saver FD/savings certificate/SSY), meal_voucher_statement (statement/invoice), ev_document (purchase invoice/charging receipt/lease letter), loan_interest_certificate, loan_statement, salary_slip, form_16 - 4 layouts, clean/scan/phone-photo quality, 131 with Hindi headers, 221 MB. Sample images were inspected; a pagination bug that painted continuation content onto page 1 was found visually and fixed before the final batch.
**Unified manifest v2:** 9,833 documents = 5,753 real + 4,080 synthetic (2,430 older Task A/B synthetic + 240 Agami + 200 FATURA + 1,210 generated); 0 excluded; real/synthetic manifests and split files kept separate. Validation passes (0 errors); audit: 0 byte duplicates, 0 groups spanning splits, no synthetic document near-identical (<=4 bits) to a real val/test document.
**Honest limits:** all seven requested categories still have **zero real labelled documents**; synthetic coverage is development-only. In the generated corpus the test slice for layouts A-C is near-identical to training pages (116/161), so only layout D (174 test docs, 3 near-identical) measures generalization - and even that is synthetic. OCR check on 110 sampled generated documents: class-defining text readable 100%, main amount recovered 95.5% (clean 91%, scan 97%, photo 100%; OCR confuses 0/o and drops some lines). The label mapping of real data still needs human review. See `docs/DOC_TYPE_DATASET_READINESS.md` for per-class readiness and `scripts/dataset_discovery/README.md` to rebuild.
