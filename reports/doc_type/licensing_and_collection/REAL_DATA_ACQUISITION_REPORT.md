# Real-data acquisition and dataset preparation: report (2026-10-19)

Scope: existing local data, unresolved sources, new public sources, ingestion automation, collection operations. No model was trained or tuned, no synthetic data generated, no dataset or model downloaded, no message sent, no terms accepted. Task A, the Task B V1 snapshot, the frozen test (sha256 fd8bafb3ad13...) and the blocked V2 assets are untouched (checked at the end: 34/34 snapshot checksums OK).

## 1. Existing data recovered or newly approved
- **Recovered locally: nothing new.** `local_inventory.md` accounts for every image/PDF under the real-data directories: each is either in manifest v2.1 or a byte-identical duplicate of a manifested file (152 duplicates = the 152 rows dropped at dedup/corruption time; 2 undecodable `in_financial` images). `indian_all/real` (837 files) is a copy of already-manifested files. `datasets/interim/v2_pool` belongs to the blocked V2 track and was not opened.
- **Newly approved:** FUNSD (non-commercial research only; 199 forms, class `other` only). **Approval withdrawn:** ReceiptBench (867 docs; empty README, no provenance/consent). Net approved real documents: **3,325 of 5,753** (3,993 under approved-sources-v1; see section 4 for the effect).
- Approval is now enforced at acquisition time: `approval_gate.py` + `fetch_verified.py --source-id`.

## 2. Sources investigated and evidence
Decision table with exact terms and links: `licensing_decisions.md` (unchanged this round; evidence for IDRBT and Roboflow added to the licence register, 28 entries). Search results and conclusions: `SEARCH_LOG.md`. Draft permission requests for every blocked source plus an owner-attestation template: `PERMISSION_REQUESTS.md` (**not sent**).
New candidates found this round: IDRBT Cheque Image Dataset (112 real volunteer-written cheques, no licence -> unverified, permission needed), Roboflow `drumil-shah/form-16` (45 images, CC BY 4.0 label, sign-in required, provenance undocumented, PAN fields -> blocked), AgamiAI Indian-Income-Tax-Returns and `royayush/cheque-synthetic-images` (synthetic, not acquired). **No public source for the 12 target classes passes the approval gate.**

## 3. Usable real documents per class (approved sources only, manifest v2.1 under approved-sources-v2)
"Groups" = distinct `group_id` (merchant/template family where the source defined one, otherwise one per document); true issuer ids exist only for the collected data.

| class | real (all) | real approved | train / val / test | groups | approved sources |
|---|---|---|---|---|---|
| bank_statement | 85 | 0 | 0 / 0 / 0 | 0 | - |
| cheque | 92 | 0 | 0 / 0 / 0 | 0 | - |
| form_16 | 37 | 0 | 0 / 0 / 0 | 0 | - |
| fuel_receipt | 140 | 71 | 63 / 1 / 7 | 51 | invoices_pt, manual_in, my_receipts |
| hotel_bill | 221 | 81 | 77 / 1 / 3 | 33 | my_receipts |
| invoice | 294 | 294 | 200 / 45 / 49 | 250 | in_grocery_gst, my_receipts, rvl_cdip |
| medical_bill | 34 | 11 | 10 / 0 / 1 | 6 | my_receipts |
| other | 499 | 499 | 402 / 47 / 50 | 497 | funsd, rvl_cdip |
| receipt | 3355 | 1942 | 1605 / 167 / 170 | 1444 | cord, invoices_pt, manual_in, my_receipts |
| salary_slip | 89 | 0 | 0 / 0 / 0 | 0 | - |
| travel_document | 557 | 247 | 217 / 22 / 8 | 30 | manual_in, my_receipts |
| utility_bill | 350 | 180 | 146 / 15 / 19 | 122 | manual_in |

Classes with **zero** approved real documents: bank_statement, cheque, form_16, salary_slip, and (as before) ev_document, insurance_premium_receipt, investment_certificate, loan_interest_certificate, loan_statement, meal_voucher_statement, rent_receipt, rental_agreement, school_fee_receipt. Collected-and-validated documents so far: **0** (`collection_manifest.jsonl` is empty, validator passes).

## 4. Blocked documents and the effect of tightening
By licence/provenance: sroie 958, receiptbench 867, in_financial 390, user_real 198, hf_medical 15 (2,428 real docs). By label ambiguity: 78 flagged for review (34 of them in approved sources: manual_in 16, my_receipts 18; keep them out of any evaluation until a human decides) and 2,971 approved real docs whose label is only a collection-rule mapping (`mapping_review_queue_v2_1.jsonl`, 286 prioritised). Privacy: nothing with personal data is in the collection; the sources above may contain real people's data and stay unused.
**Consequence of withdrawing ReceiptBench:** real val/test support collapsed for the classes it carried: hotel_bill 1/3, fuel_receipt 1/7, medical_bill 0/1, travel_document 22/8 (val/test). Only receipt, invoice, other and utility_bill keep >=15 val and test documents. The earlier baseline (run under approved-sources-v1) therefore overstates what the approved data can currently evaluate; do not rerun it until more real data exists.

## 5. Scripts and automation implemented (all new, reuse the existing schema and register)
- `approval_gate.py`: one decision function (config + register must agree) used by CLI and by `fetch_verified.py`; dataset downloads without an approved `--source-id` are refused; approved ones log the evidence into the provenance record.
- `collection_ops.py`: `status` (validate + per-class progress with unique issuers, annotation and split counts, review/redaction queues, explained blockers; exit 1 on errors), `import` (dry-run by default; licence/permission gate, redactor != verifier, metadata PII scan, local-OCR PII scan of the redacted copy, sha256 + pHash duplicates against the collection and the real corpus, nothing written without `--apply`, backup kept), `assign-splits` (deterministic issuer-group assignment honouring every gate; specimens train-only; val/test wait for double annotation).
- `collection_manifest.py` (schema v1.1: optional `redacted_sha256`/`redacted_phash`; duplicate-hash check), `local_inventory.py`, `write_licensing_decisions.py`, licence-register overlay + 4 new entries.
- Reports never contain document text, names, numbers or file names (records are referenced by opaque id or batch line; a test asserts this).
- A bug found by the new tests and fixed: the PII scan flagged hex hash fields as phone numbers.

## 6. Tests and resources
Full suite: **294 tests OK** (was 266; 28 new in `tests/test_collection_ops.py` covering licence gating, acquisition refusal, privacy checks, duplicates, split isolation, dry-run safety, progress/queue reporting). Collection validator on the real manifest: passed, 0 records. Disk free at the end: 3.7 GB (was 4.7 GB at the start of the audit session; this task added only reports and a 5,753-entry hash cache). Network use: read-only page views and web searches.

## 7. Remaining tasks that need a human
1. Send the permission requests (a person must choose the channel, send, and keep the replies): `in_financial` uploader, RRC organisers, IDRBT, Hugging Face uploaders (hf_medical, ReceiptBench), Roboflow uploader (needs a signed-in account).
2. Sign owner attestations for `manual_in` and `user_real` (project owner).
3. Review the 286-record mapping queue and the 78 cue conflicts.
4. Obtain consent and real documents (the only route for the 12 target classes), run redaction with a second-person verifier, annotate (double for val/test).
5. Have counsel check the consent wording (DPDP Act 2023).
6. Decide whether to restore ReceiptBench if the uploader provides provenance.

## 8. Highest-value next action
Collect the first real batch: one consenting volunteer set (rent receipts, insurance and school-fee receipts, loan-interest certificates) with signed consent records, then run `collection_ops.py import` in dry-run mode. Nothing else raises real coverage; public sources are exhausted for these classes.
