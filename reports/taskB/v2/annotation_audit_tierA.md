# Task B v2 - Tier-A annotation audit (54 pages of `in_grocery_gst`)
Files: `annotations_tierA.json` (this folder; working copy `datasets/interim/v2_pool/annot/annotations_tierA.json`), machine-readable audit `annotation_audit_tierA.json`, decisions with per-field notes `datasets/interim/v2_pool/annot/decisions.jsonl`, scripts `scripts/taskB/v2_annot_{view,build,find,audit}.py`.

## IMPORTANT: who annotated
The values were read by **me (Claude) from image crops** of every page (full page, header, totals, HSN column), not by an independent human. OCR/regex suggestions were kept as `original_suggestion` (SUGGESTION_ONLY), no Task B model output was used. Every record says `annotator: claude (AI-assisted ...)`, `verified_against_image: true` means "checked against the image by this annotator". Treat these as **silver, not gold**: before they enter V2 training a human should re-check a sample (suggest 10 pages, all fields) - see "next step".

## Conventions applied (as instructed)
- `cgst`/`sgst` = invoice-level totals (sum row/box); a multi-rate invoice with no printed invoice-level total gets **null** (not summed by me).
- `hsn_sac`: scalar field in the canonical schema -> only filled when the invoice has exactly one HSN/SAC code; otherwise status `unsupported_multi_value`, value null, and the **list of unique codes is stored in `extra.hsn_sac_list`** (with boxes where locatable).
- `subtotal`: only when the invoice prints an amount-before-tax/taxable total with no discount/scheme line that would make it ambiguous; gross, scheme/discount, taxable value, round-off are kept in `extra`. Otherwise null.
- `total_amount`: printed final payable amount after round-off (not the unrounded Net Amt). `tax_total`: printed total-tax figure only (not derived).
- Nulls are explicit with a reason in `note`.

## Results
1. **Tier-A documents completed: 54/54** (status verified 54, unusable 0). Two pairs are the *same invoice photographed twice* (GTJ70931: 235614/235738; EV/25-26/5315: 000815/000838) -> they must stay in the same split group.
2. **Field coverage (non-null / 54)**: merchant_name 54, gstin 54 (all 54 checksum-valid), invoice_number 52, invoice_date 53, total_amount 53, cgst 49, sgst 49, tax_total 32, subtotal 16, hsn_sac 10 (+43 multi-code docs with list in `extra`), igst 0.
3. **Null fields**: invoice_number 2 (ink blot, torn page), invoice_date 1 (torn), total_amount 1 (page 1 of 3), cgst/sgst 5 each (multi-rate without invoice-level total, page 1 of 3, one combined-GST layout), tax_total 22, subtotal 38 (mostly: discount/scheme/gross vs taxable ambiguity - 37 docs print such lines), hsn_sac 44 (43 multi-code + 1 none printed), igst 54 (all intra-state).
4. **Corrected OCR suggestions** (a suggestion existed and my value differed): merchant_name 30 of 30 (suggestions were whole header lines), total_amount 34 of 48, invoice_number 11 of 48, gstin 10 of 53, invoice_date 2 of 41, cgst 3, sgst 3, subtotal 2, hsn 1. Suggestions agreed: gstin 43, invoice_date 39, invoice_number 36, total_amount 14. A human value with *no* suggestion at all: tax_total 32 (the regex never produced one), cgst 42, sgst 43, merchant 24. Suggestions I rejected by setting null: hsn 38 (per the new HSN rule), subtotal 5, invoice_number 1.
5. **Bbox**: corrected/overridden boxes 19 (total 5, date 8, invoice_no 3, merchant 1, cgst 1, subtotal 1, ...). Quality of boxes is the weak point: amounts are OCR-word-exact (`ocr_exact`: total 52, cgst/sgst 46+46, subtotal 16, tax_total 24), but **text fields mostly use `ocr_substring_estimate`** - OCR glued labels and values into one word and the box is a character-proportion estimate (merchant 51, gstin 44, invoice_no 45, date 48). 7 values have no box (merchant 2, gstin 4, total 1; listed per doc in the JSON); 3 docs share one ambiguous box for cgst and sgst (OCR glued two numbers; flagged `bbox_ambiguous`). Visual overlay check of 6 random pages: boxes sit on the right printed values; two merchant boxes are wider than the name (whole OCR line).
6. **Unusable documents: 0** (9 pages are partial or mid-document: B R Enterprise page 1 of 2/3 (4), Scootsy web-print pages with only some line items (3, totals are invoice-level), one torn Binoy page, and the Malati 'page 2 of 2' copies; their uncertain fields are null).
7. **Supplier/layout**: 20 sellers (18 identified by printed name, 1 by GSTIN only), max 4 pages per seller; top seller share 7.4%. Layout families: Marg-style (Shiv Shankar, Kirtiraaj, Lokenath, Malati...), Tally-style (Shaw, Shankar, Manshi, S S Agencies), web-printed (Scootsy, Deva), distributor ERP (Ekadant, K D, B G, B R, Suchitra, Durga).
8. **GSTIN / CGST / SGST / HSN**: GSTIN 54/54 (20 distinct seller GSTINs, 54 checksum-valid; buyer GSTIN deliberately excluded), CGST 49 & SGST 49 (equal in all 49), HSN scalar 10 / list 53.
9. **Multi-rate**: 20 docs have >= 2 GST rates (rates seen 0, 2.5, 5, 6, 9, 12, 18 %); 4 have 0%-rate lines. All-in: 28 docs have tax_total == cgst+sgst exactly; the one subtotal arithmetic mismatch (000429, -6.41) is explained by a printed discount (-6.00) and round-off (-0.41).
10. **Schema problems found**
   - `hsn_sac` is a scalar: **43/54 invoices have 2-23 HSN codes**, so only 10 labelled documents; a list-valued field (or per-line codes) is needed for real HSN supervision. V1 code (`id` type, one value) cannot ingest `extra.hsn_sac_list` without a schema change.
   - No fields for taxable value, discount/scheme, round-off, gross: `subtotal` is null on 38/54 although taxable value is printed on most. Decision needed whether `subtotal` may mean taxable value (V1 convention: unknown for Indian bills).
   - No per-rate tax representation; 5 multi-rate docs without a printed total have cgst/sgst null.
   - **IGST: none of the 54 pages is inter-state.** The pool cannot add IGST supervision, whatever else is done.
   - Single-buyer pool: GSTIN labelling needs the seller-vs-buyer rule (done), but a model trained on it may learn "the GSTIN that is not 19CPQPS9464F1ZX".
   - OCR label/value gluing makes word-level box alignment impossible for ~85% of text fields (substring estimates), which matters for LayoutLMv3 token labels. Needs word-level re-OCR or manual boxes before training.
   - merchant_name keeps printed suffixes ("(2023-24)", "PVT. LTD."); V1 normaliser handles case/punctuation only.
   - Letter/digit ambiguity (1 vs I) in invoice numbers/GSTIN handled by checksum for GSTIN; invoice numbers 'BGBLN-25-I...' and 'DVL...' are best readings.

## Recommended next step (waits for your approval)
(a) a human re-checks ~10 of these 54 pages (suggest: 2 each of the three layout families + the 3 partial ones) against `annotations_tierA.json`; (b) you decide the `hsn_sac`-list and `subtotal` schema questions; (c) then Tier B (39 pages) or go straight to the leakage audit / V2 split with these 54 as silver data. No retraining, no split, no touch of the V1 test set so far.
