# Task B v2 - independent QA of Tier-A annotations (10 of 54 pages)
## What this QA is - and is not
The project owner asked for an **independent person**. No human was available in this session, so I ran a **proxy**: three fresh-context AI reviewer agents (same model family as the annotator, so errors can be correlated) that never saw my annotations:
- Reviewers A and B: blind re-annotation of 5 pages each from the images only (values, null decisions, seller GSTIN, HSN lists...).
- Reviewer C: judged every drawn bbox on overlay images (field-name labels only, no values).
**This is NOT the requested human QA.** A ready blind packet for a real person is in `datasets/interim/v2_pool/qa/HUMAN_QA/` (pages, overlays, `answer_sheet.csv`, `INSTRUCTIONS.md`); returned sheet is scored with `python scripts/taskB/v2_qa_human_packet.py score <sheet.csv>`. Tier B was **not** started, awaiting your decision.

Pages (10): Marg-style (Shiv Shankar; Malati page 2/2 = edge), Tally-style (Shaw; S.S. Agencies), web-printed (Scootsy; Deva), ERP-style (Ekadant; Durga torn corner; B R Enterprise page 1/3 = edge; Binoy torn = edge). Covers all 4 families and the 3 edge cases.

## Results (14 fields x 10 pages = 140 comparisons)
| metric | value |
|---|---|
| value agreement (after fixes below) | 139/140 = 99.3% (first pass 138/140 = 98.6%) |
| null-decision agreement | 140/140 after fixes (139/140 first pass) |
| field-level disagreements | 2 first pass -> 1 remaining (convention) |
| page-level disagreements | 2 of 10 pages first pass (Scootsy `discount`, Durga `round_off`) |
| seller-GSTIN agreement | 10/10 (no buyer GSTIN mix-ups) |
| HSN list agreement (set equality) | 10/10 |
| bbox: correct / loose / wrong | 125 / 5 / 3 of 133 boxes = 94.0% correct, 97.7% correct-or-loose |
Disagreements (kept separate in `qa_disagreements_tierA.json`):
1. p07 Durga `round_off`: I had annotated -0.30, computed as the difference between two printed totals; the invoice prints only "Roundoff To 2994.00". **Systematic error (derived value)** - found in 8 docs (B G Distributors x4, Durga x3, K D x1). Fixed: set to null with a note, all 8.
2. p02 Deva `discount`: reviewer wrote 0.00 (the row prints 0.00), I had null. Convention, not an error: zero-valued discount/round-off are recorded as null with "printed zero"; remains listed as a disagreement.
Bbox problems found: p04 invoice_date box on the 'Printed on' stamp (fixed: now on DATE), p04 sgst box identical to the cgst box because OCR glued the two numbers (fixed: sgst bbox null `glued_with_cgst_not_boxed`; same for 2 other docs), p10 discount box on a serial-number cell (fixed: located lower-right, prefers right/lower occurrences), merchant boxes loose on 3 pages (p04 cuts 'LTD.', p06 includes address top, p07 only the visible 'ENTERPRISE') and p06 date/HSN boxes loose - left as is, they are character-proportion estimates and can't be improved without word-level re-OCR. The fixes were not re-reviewed.

## Conclusion on Tier A
- The error found (derived round-off) was systematic and has been fixed across all 54 pages; the other defects are bbox-quality issues, not value errors.
- Agreement is high but the reviewers share my model family and the same reading conventions; 10 pages (18%), one reviewer per page. It supports "no gross systematic value errors" - it does not replace a human check.
- Remaining bbox risk: values outside the 10 QA pages have boxes chosen by the same automatic matching that produced the 3 wrong boxes; those 44 pages are unreviewed for boxes.

## V2 schema / annotation changes applied (V1 schema untouched; V1 hashes verified)
`scripts/taskB/schema_v2.py`, `models/taskB/v2/field_schema_v2.json`: `hsn_sac` -> list of unique digit codes (per-code boxes in `hsn_sac.per_code_bbox`, `list_complete` flag, partial lists flagged); new `taxable_value`, `discount` (with `discount_kind`), `round_off` (signed, printed only); `subtotal` keeps its meaning = explicit labelled sub total only. No invoice in Tier A prints a literal sub total, so **subtotal = null on 54/54** (the 16 previous footing / 'AMOUNT BEFORE TAX' values moved to `taxable_value`). Note for you: 4 Shiv Shankar invoices print "AMOUNT BEFORE TAX" - I treated it as taxable value, not subtotal; say if you want it as subtotal.
Tier-A v2 coverage (non-null / 54): hsn_sac list 53, taxable_value 53, discount 34, round_off 27, tax_total 32, cgst 49, sgst 49, subtotal 0, igst 0.
Files: `annotations_v2_tierA.json` here; code `v2_migrate_annotations.py`, `v2_qa_compare.py`, `v2_qa_human_packet.py`, `v2_qa_overlays.py`.
