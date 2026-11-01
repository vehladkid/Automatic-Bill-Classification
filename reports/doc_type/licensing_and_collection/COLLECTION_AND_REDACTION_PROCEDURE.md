# Real-document collection, intake and redaction procedure (v1)

Scope: real, consented, anonymised documents for the classes without approved real data (priority list and per-class minimums: `collection_targets_v1.json`). Records follow `collection_manifest_schema_v1.json` and are checked with `scripts/dataset_discovery/collection_manifest.py`. This is an engineering procedure, not legal advice: have counsel confirm the consent wording against India's Digital Personal Data Protection Act, 2023 and your organisation's data-protection policy before collecting from anyone.

## 1. Principles
1. No document enters training or evaluation without (a) a recorded permission, (b) a verified redacted copy, (c) an issuer-group id and split.
2. Originals never leave the vault, are never committed, never uploaded to a cloud service, OCR API or hosted LLM. All OCR/annotation runs locally on the redacted copy. Anything external needs explicit written authorisation from the data owner and the data-protection lead.
3. The manifest stores opaque ids and review states only. Names, account/policy/loan numbers, PAN, Aadhaar, addresses, phone, e-mail, employee ids and GSTIN of individuals are never written to the manifest, notes, file names or commit messages. The validator rejects such patterns.
4. Document type, expense category and tax-evidence candidates are annotated as three separate decisions; eligibility for a tax benefit is never labelled.
5. Synthetic documents never count toward real-data targets; blank specimens/templates are flagged `specimen_or_template` and may be used for training only.

## 2. Roles
Collector (obtains documents and consent) - Redactor (applies redaction) - Verifier (different person; checks redaction) - Annotators A and B (label the redacted copy) - Data-protection lead (owns the vault, consent register and deletion). One person must not both redact and verify the same document.

## 3. Sourcing (in order of preference)
1. Documents the project owner holds about themselves (`owner_own_document`).
2. Volunteers who sign an individual consent (`individual_written_consent`): rent receipts, insurance, school-fee, loan-interest certificates, investment statements, salary slips, Form 16, bank pages.
3. Employers/HR teams under a written permission and data-processing agreement (`employer_written_permission`); employees' consent is still required for their payslips/Form 16.
4. Issuers (insurers, schools, meal-card providers, EV charging operators, lenders) supplying real anonymised samples under written permission (`issuer_written_permission`).
5. Public sources only after the licence is verified from the primary page and added to the licence register (`public_licence_verified`). Scraped web images are not a source (see `in_financial`).
Never obtain documents by scraping, by asking employees informally, or from HR systems without a documented purpose.

## 4. Consent record (kept in the vault, referenced by `permission_ref`)
Purpose (training and evaluating an internal document-classification model; no sale, no publication of the documents); what is collected; that only a redacted copy is used; where it is stored and who can access it; retention period and deletion on request; right to withdraw before the redacted copy is frozen into a test set (after which only deletion of the copy is possible - say so); no external transfer; signature/date. One record per individual or organisation; the record id is the only link stored in the manifest.

## 5. Storage layout (outside the repository)
```
vault/            encrypted volume (e.g. LUKS or age/gpg); originals, consent register, id map; offline HMAC key for issuer groups; never mounted on a networked build host
redacted_store/   flattened redacted copies only (git-ignored or outside the repo); path recorded in `redacted_copy`
repo manifests    collection_manifest.jsonl (opaque ids and states only)
```
Access: vault - Collector, Redactor, Verifier, DP lead. Redacted store - annotators and ML engineers. The id map (opaque doc id <-> vault file) is in the vault only.

## 6. Intake steps
1. Collector checks the consent record, then ingests the file into the vault; a random `doc_id` (`COL-<8hex>-<4hex>`) is generated; file names with personal data are discarded.
2. Collector sets `source_id`, `permission_basis`, `permission_ref`, `sensitive_review_status=pending`, `split=unassigned`.
3. Issuer group: normalise the issuer name + template family (e.g. same bank/employer/school and same layout) and compute `IG-` + first 10 hex of HMAC-SHA256(offline key, normalised string). Same issuer/template family => same id => same split.
4. Redactor creates the redacted copy (section 7) and sets `redaction_required`; Verifier checks (section 8); only then `redacted_verified` and `redacted_copy` are filled.
5. Annotators label `doc_type`, `expense_category`, `tax_evidence` independently on the redacted copy; 20% of documents (all val/test) are double-annotated; disagreements are adjudicated and noted with an `ambiguity_flags` value.
6. Split assignment (only after redaction and annotation): per document type, assign whole issuer groups deterministically (stable hash of the group id) so that each split has the minimum distinct issuers (`collection_targets_v1.json`); test groups are frozen and never revisited.
7. Run the validator; fix every error. A record that fails or is `rejected_unredactable` stays out of all training/evaluation manifests.

## 7. Redaction rules
Remove, by burning opaque boxes into the pixels (no removable overlay layers): person names (holder, employee, tenant, landlord, student, parent), addresses, phone/e-mail, PAN, Aadhaar, account/customer/policy/loan/folio/employee numbers, UAN/PF numbers, signatures and stamps with names, photographs, QR codes and barcodes, MICR line and cheque numbers, transaction rows of bank statements (keep header/summary or redact descriptions and counterparties). Keep what defines the class: issuer logo/name of institutions (not individuals), titles ("Rent Receipt", "Form 16 Part A"), table structure, headings, currency symbols, amount formats (replace actual amounts only if they could identify a person; otherwise keep, because amounts are class-relevant). Then export as a flattened PNG/PDF-image, strip EXIF/PDF metadata, and re-save under the redacted store name.

## 8. Verification (second person)
Open the redacted copy at 100%; check every region listed in section 7; run local OCR on the redacted copy and confirm no PAN/Aadhaar/phone/e-mail/long-number/IFSC/GSTIN pattern remains (the same patterns the validator uses); check metadata is empty; record `redacted_verified` or `rejected_unredactable`. Any doubt = reject.

## 9. Quality, ambiguity and review flags
Use `ambiguity_flags` instead of guessing: `type_ambiguous` (receipt vs invoice vs specific type), `expense_category_ambiguous`, `tax_mapping_ambiguous` (insurance life vs health, loan purpose unknown, EV purchase vs lease), `partial_page`, `multi_document_pdf` (split into separate records first), `low_quality_scan`, `language_mixed`, `specimen_or_template`. `reviewer_notes` is short free text without personal data.

## 10. Retention, deletion, external services
Originals: keep only while redaction/verification is pending or as the consent record requires; then delete from the vault unless the data-protection lead documents a reason. Withdrawal requests: delete the original and the redacted copy and mark the manifest row removed in a new manifest version. No upload of originals or redacted copies to external services without explicit written authorisation recorded in the consent register.

## 11. Acceptance for a class to support a real metric
>= 30 val and >= 30 test redacted_verified, double-annotated documents from >= 6 distinct issuer groups each (`ready_for_per_class_metric` in `collection_status.json`), >= 90% Indian. Until then report that class as "no real evaluation" regardless of any synthetic score.

## 12. Example record (illustrative only, not data)
```json
{"doc_id":"COL-3f9a1c0d-7be2","doc_type":"rent_receipt","expense_category":"rent","tax_evidence":["hra_evidence"],"real":true,"source_id":"COLL-ORG-01","permission_basis":"individual_written_consent","permission_ref":"CR-0042","issuer_group_id":"IG-9c41ab07e3","split":"unassigned","annotation_status":"pending","sensitive_review_status":"pending","redacted_copy":null,"ambiguity_flags":[],"reviewer_notes":"","eligibility_status":"pending_verification","country":"IN","language":"en","created":"2026-10-19"}
```
