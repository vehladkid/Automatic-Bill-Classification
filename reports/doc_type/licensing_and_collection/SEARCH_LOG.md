# Source search log (2026-10-19)
Searches were web searches and read-only page views; nothing was downloaded. Earlier rounds (HF/Zenodo/Kaggle catalogue of 50 records, `docs/DATASET_MODEL_RESEARCH.md`) were not repeated.

| need | queries (summary) | outcome |
|---|---|---|
| rent receipts / agreements | "Indian rent receipt dataset images HRA", "rental agreement India dataset legal document" | none. Only rent-price tables, a CC0 receipt-OCR set (192 receipts, not verified as Indian), and VidhikDastaavej (anonymised private legal documents; release status not verified) |
| salary slips / Form 16 | "Indian salary slip payslip dataset images Form 16", "huggingface payslip OCR India", "Form 16 OCR dataset github" | no open image dataset. Found: Roboflow `drumil-shah/form-16` (45 images, CC BY 4.0 label, sign-in to download, provenance undocumented, PAN/name fields) -> blocked; AgamiAI Indian-Income-Tax-Returns (synthetic ITR, not real) |
| insurance / school fee | "Indian insurance premium receipt OR school fee receipt dataset" | none modern. A British Library EAP collection holds historical Indian receipts (1913-47): wrong domain for current documents |
| bank statements / cheques | "Indian bank statement cheque image dataset" | IDRBT Cheque Image Dataset: 112 real, volunteer-written cheques, no licence stated -> blocked pending permission. Synthetic cheques (Apache-2.0, gated) and annotations-only (CC BY-SA 4.0) exist but are synthetic/derived. AgamiAI bank statements already used (synthetic) |
| EV charging / meal vouchers | "EV charging receipt invoice dataset India OR meal voucher Sodexo" | none. Only a government EV-station consumption table (no documents) and commercial paid OCR-invoice vendors (paid: excluded) |
| loan / investment certificates | "Indian loan statement OR home loan interest certificate OR PPF dataset" | none; only explanatory pages |
| general Indic documents | "AIKosh India document images invoices receipts" | AIKosh holds layout/OCR sets (IndicDLP, gated; newspapers/books domains) - not our classes |

Pages read for decisions: Kaggle (in_financial), opendatacommons DbCL, FUNSD licence page, RRC pages, HF cards (SROIE mirror, hf_medical x2, ReceiptBench), IDRBT page, arXiv 2606.20682, Roboflow Form-16 page. huggingface.co was not fetchable by the fetch tool for two pages (royayush, AgamiAI ITR): recorded as unverified. No gated page, login, form or paywall was used.
Conclusion: **no public source for the 12 target classes passes the approval gate; real data must be collected under consent.**
