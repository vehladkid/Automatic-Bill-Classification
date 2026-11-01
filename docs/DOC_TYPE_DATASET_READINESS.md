# Document-type dataset v2: coverage and training readiness
_Generated 2026-10-19T20:46:59 by `scripts/dataset_discovery/readiness_report.py` from the machine-readable reports in `reports/doc_type/` and `datasets/metadata/`. Nothing here was trained._

**Corpus:** 9,833 documents = **5,753 real** + **4,080 synthetic** (excluded by mapping: 0). Schema `doc-type-schema-v2`. Validation: **PASSED** (9,833 rows, 9,769 images + 240 PDFs opened, 2,126 checksums recomputed, 1,210 generated records checked for internal consistency).

## Per document type (real and synthetic reported separately; splits train/val/test)

| document type | real | real Indian | synthetic | readiness |
|---|---|---|---|---|
| bank_statement | 70/5/10 | 0 | 240/0/0 | PARTIAL: real examples below policy (train>=150, val>=30, test>=30); synthetic available for development |
| cheque | 76/10/6 | 0 | 0/0/0 | PARTIAL: real examples below policy (train>=150, val>=30, test>=30) |
| ev_document | 0/0/0 | 0 | 67/12/31 | NOT READY for real-world claims: no real labelled documents; synthetic-only development possible |
| form_16 | 27/8/2 | 0 | 72/9/29 | PARTIAL: real examples below policy (train>=150, val>=30, test>=30); synthetic available for development |
| fuel_receipt | 112/13/15 | 43 | 120/15/15 | PARTIAL: real examples below policy (train>=150, val>=30, test>=30); synthetic available for development |
| hotel_bill | 177/22/22 | 60 | 120/15/15 | PARTIAL: real examples below policy (train>=150, val>=30, test>=30); synthetic available for development |
| insurance_premium_receipt | 0/0/0 | 0 | 59/17/34 | NOT READY for real-world claims: no real labelled documents; synthetic-only development possible |
| investment_certificate | 0/0/0 | 0 | 65/13/32 | NOT READY for real-world claims: no real labelled documents; synthetic-only development possible |
| invoice | 200/45/49 | 169 | 380/58/62 | READY (real train/val/test meet policy) |
| loan_interest_certificate | 0/0/0 | 0 | 61/14/35 | NOT READY for real-world claims: no real labelled documents; synthetic-only development possible |
| loan_statement | 0/0/0 | 0 | 66/15/29 | NOT READY for real-world claims: no real labelled documents; synthetic-only development possible |
| meal_voucher_statement | 0/0/0 | 0 | 65/14/31 | NOT READY for real-world claims: no real labelled documents; synthetic-only development possible |
| medical_bill | 26/4/4 | 20 | 120/15/15 | PARTIAL: real examples below policy (train>=150, val>=30, test>=30); synthetic available for development |
| other | 402/47/50 | 0 | 144/18/18 | READY (real train/val/test meet policy); Indian-specific claims need more Indian docs |
| receipt | 2724/317/314 | 190 | 360/45/45 | READY (real train/val/test meet policy); Indian-specific claims need more Indian docs |
| rent_receipt | 0/0/0 | 0 | 68/12/30 | NOT READY for real-world claims: no real labelled documents; synthetic-only development possible |
| rental_agreement | 0/0/0 | 0 | 64/13/33 | NOT READY for real-world claims: no real labelled documents; synthetic-only development possible |
| salary_slip | 67/10/12 | 0 | 71/19/20 | PARTIAL: real examples below policy (train>=150, val>=30, test>=30); synthetic available for development |
| school_fee_receipt | 0/0/0 | 0 | 62/17/31 | NOT READY for real-world claims: no real labelled documents; synthetic-only development possible |
| travel_document | 442/62/53 | 172 | 360/45/45 | READY (real train/val/test meet policy); Indian-specific claims need more Indian docs |
| utility_bill | 281/35/34 | 183 | 480/60/60 | READY (real train/val/test meet policy); Indian-specific claims need more Indian docs |

Policy for READY: >=150 real train, >=30 real val, >=30 real test per class; Indian-specific claims need >=30 Indian documents per class. Synthetic documents never count toward real readiness.

## Subtypes (real | synthetic)

- **bank_statement**: real {'null': 85} | synthetic {'digital_type1': 100, 'digital_type2': 100, 'scanned_type1': 20, 'scanned_type2': 20}
- **cheque**: real {'null': 92} | synthetic -
- **ev_document**: real - | synthetic {'charging_receipt': 61, 'purchase_invoice': 28, 'lease_letter': 21}
- **form_16**: real {'null': 37} | synthetic {'part_a_b': 110}
- **fuel_receipt**: real {'null': 140} | synthetic {'null': 150}
- **hotel_bill**: real {'null': 221} | synthetic {'null': 150}
- **insurance_premium_receipt**: real - | synthetic {'health': 60, 'life': 50}
- **investment_certificate**: real - | synthetic {'elss': 26, 'tax_saver_fd': 26, 'ppf': 22, 'savings_certificate': 20, 'ssy': 16}
- **invoice**: real {'null': 294} | synthetic {'null': 300, 'fatura_template': 200}
- **loan_interest_certificate**: real - | synthetic {'summary': 59, 'with_schedule': 51}
- **loan_statement**: real - | synthetic {'account_statement': 110}
- **meal_voucher_statement**: real - | synthetic {'statement': 83, 'invoice': 27}
- **medical_bill**: real {'null': 34} | synthetic {'null': 150}
- **other**: real {'null': 499} | synthetic {'null': 180}
- **receipt**: real {'null': 3355} | synthetic {'null': 450}
- **rent_receipt**: real - | synthetic {'single_month': 89, 'multi_month': 21}
- **rental_agreement**: real - | synthetic {'residential': 110}
- **salary_slip**: real {'null': 89} | synthetic {'monthly': 110}
- **school_fee_receipt**: real - | synthetic {'with_other_fees': 79, 'tuition_only': 31}
- **travel_document**: real {'null': 557} | synthetic {'null': 450}
- **utility_bill**: real {'null': 350} | synthetic {'null': 600}

## Provenance

- Real by source: {'my_receipts': 1162, 'cord': 998, 'sroie': 958, 'receiptbench': 867, 'in_financial': 390, 'manual_in': 368, 'rvl_cdip': 320, 'funsd': 199, 'user_real': 198, 'in_grocery_gst': 169, 'invoices_pt': 109, 'hf_medical': 15}
- Synthetic by source: {'synthetic_in': 2430, 'synthgen_v1': 1210, 'agami': 240, 'fatura': 200}
- Real by country: {'ID': 998, 'MY': 969, 'US': 964, 'IN': 837, 'DE': 714, 'mixed_unverified': 390, 'null': 384, 'PT': 109, 'CA': 95, 'ES': 34, 'NL': 29, 'CN': 24, 'GB': 24, 'SE': 22, 'AU': 22, 'HR': 20, 'GR': 20, 'HK': 17, 'AT': 17, 'CZ': 10, 'IR': 8, 'FR': 7, 'PL': 5, 'BE': 5, 'AE': 5, 'EE': 4, 'LT': 4, 'PH': 4, 'SG': 3, 'BD': 2, 'BR': 2, 'EG': 2, 'HU': 1, 'KE': 1, 'NZ': 1}
- Real by language: {'en': 4055, 'de': 664, 'null': 661, 'pt': 112, 'id': 85, 'ja': 42, 'ca': 24, 'vi': 18, 'fr': 10, 'zh-cn': 9, 'es': 9, 'et': 8, 'sv': 7, 'hr': 7, 'nl': 6, 'ro': 5, 'sl': 5, 'el': 5, 'so': 4, 'ko': 4, 'it': 3, 'ar': 3, 'lt': 2, 'no': 1, 'pl': 1, 'sk': 1, 'cs': 1, 'hu': 1}
- Synthetic by country / language: {'IN': 3880, 'null': 200} / {'en': 3949, 'en+hi': 131}
- Review status: {'needs_mapping_review': 5130, 'generator_exact': 4080, 'source_label': 623}

## Licences (`datasets/metadata/license_register_v2.json`)

- **usable**: cord, in_grocery_gst, invoices_pt, my_receipts, synthetic_in, agami_indian_bank_statements, fatura, synthgen_v1
- **conflict**: sroie
- **usable_with_restrictions**: funsd, in_financial, manual_in, receiptbench, rvl_cdip
- **unverified**: hf_medical, user_real
- **blocked**: docile, nutrient_doc_cls, kaggle_roboflow_zindi
- **not_acquired**: doclaynet, voxel51_invoices, drew_ipp_invoices, buthaya_payslips, midd

## Duplicate and leakage audit (`reports/doc_type/leakage_audit_v2.json`)

- Byte-identical duplicates: **0** groups (spanning splits: 0; real+synthetic: 0). Groups spanning splits: **0** of 7,676.
- Real val/test vs real train (pHash): <=4 bits: val 0/578, test 0/571; <=8 bits: val 29, test 37.
- Generated documents, test split: **seen layouts A-C** are near-identical to a training page for 116/161 documents (expected: shared layouts), **unseen layout D** for 3/174 (<=8 bits: 73). **Only the layout-D test slice is a meaningful generalization check for synthetic classes.**
- Synthetic vs real val/test: no near-identical (<=4) pair for any synthetic source; closest: { synthgen_v1: 8 bits, agami: 12 bits, fatura: 6 bits, older_synthetic_in: 10 bits }.

## OCR check of the generated documents (does the image support its label?)

- Sample 110 documents (clean / scan / phone-photo), project OCR: class-defining phrase readable **100%**, title readable **100%**, main amount recovered **96%** (OCR confusions such as 0/o and dropped lines explain the rest; the images themselves were inspected).

## Storage

- generated synthetic documents: 226 MB
- agami bank statements: 306 MB
- fatura archive and subset: 372 MB
- dit base weights: 343 MB
- free disk: 5.1 GB

## Readiness verdict

- **Real-data ready:** invoice, other, receipt, travel_document, utility_bill (Indian-specific claims limited to documents with country IN).
- **Partial real data:** bank_statement, cheque, form_16, fuel_receipt, hotel_bill, medical_bill, salary_slip.
- **No real labelled data (synthetic only):** ev_document, insurance_premium_receipt, investment_certificate, loan_interest_certificate, loan_statement, meal_voucher_statement, rent_receipt, rental_agreement, school_fee_receipt.
- **The dataset is ready for a development/feasibility training run, NOT for claims about real-world accuracy on the no-data classes.** Any model trained on the synthetic classes must be evaluated on real documents before such claims.

## Recommended next step

- Architecture: OCR-text document-type classifier first: TF-IDF (char+word) + logistic regression baseline, then a small transformer (DistilBERT-class; multilingual encoder only if Indic text matters); optional DiT image features later. All fit the 6 GB GPU (the baseline is CPU).
- Before training: human review of the real-label mapping (collection-rule labels); collect authorized, anonymized REAL documents for the no-data classes (>=30 val + >=30 test each, ideally >=150 train); report synthetic-layout-D and real metrics separately; never present synthetic scores as real-world accuracy; resolve licence conflicts (SROIE, FUNSD, in_financial provenance).
