# Task B v2 - Phase 1/2 state and data-gap analysis (2026-10-05)

## V1 freeze
Snapshot: reports/taskB/v1_snapshot/ (split jsonl, final_results, errors, inference_check, cv_ranker, V1 report, SHA256SUMS of models/taskB/* and scripts/taskB/*.py).
V1 split (datasets/processed/taskB): train 5160 (real 2352, Indian-real 232), val 646 (real 295, IN 26), test 636 (real 285, IN-real 27). test.jsonl sha256 fd8bafb3...b88c. Nothing retrained.

## Indian-real annotated data in V1 (all from ReceiptBench; travel/hotel/ticket heavy)
| split | docs | merchant | date | total | inv_no | GSTIN | CGST | SGST | IGST | subtotal | tax_total | HSN/SAC |
| train | 232 | 184 | 109 | 192 | 97 | 54 | 42 | 43 | 1 | 0 | 0 | 0 |
| val | 26 | 18 | 18 | 23 | 15 | 7 | 7 | 7 | 0 | 0 | 0 | 0 |
| test | 27 | 17 | 16 | 24 | 14 | 5 | 3 | 4 | 0 | 0 | 0 | 0 |
Categories (train): bus 53, transport 52, hotel 44, flight 44, food 20, fuel 10; zero utility bills, zero retail/GST tax invoices.

## Gaps
1. No annotated real Indian retail/B2B GST invoices -> no subtotal, tax_total, HSN/SAC, multi-rate CGST/SGST, IGST supervision. Cannot be fixed by retraining.
2. Frozen test has 27 Indian docs, 5 GSTIN, 3-4 CGST/SGST: Indian GST field evaluation on the frozen test stays INSUFFICIENT REAL TEST SUPPORT whatever V2 does (test must not change). Only CV on new data can speak to GST fields.
3. V1 errors: 224 of ~500 field errors are "GT not locatable in OCR" (OCR/format problem, not model), then FP 50, wrong value 41, abstained-wrong 37+31, wrong box 31, partial span 28. Identifier confusion errors are small in count (invoice_no vs other ID 6, subtotal<-total 2, date swaps ~6), so more data targets mainly merchant/invoice_no/GST fields rather than these.

## Real Indian documents available locally (all UNANNOTATED for fields)
| source | docs | licence/rights | notes |
| in_grocery_gst | 169 (165 groups) | CC-BY-4.0 | real GST tax invoices, but phone photos, mostly rotated 90 deg, dense multi-page distributor invoices (36+ lines), tiny text. RapidOCR finds a GSTIN in only 17/169; the ones found show 7 distinct GSTINs (one supplier = 10 docs) -> low merchant diversity, mostly West Bengal. |
| manual_in (utility, food, fuel, train, parking) | 368 | project-owner collected; redistribution rights unverified | only 3 GSTINs detected; no subtotal/GST structure for most (bills, tickets) |
| hf_medical | 15 | see manifest | |

## Why I stopped
Target was 100-200 REAL Indian documents with verified field annotations + boxes. The documents exist, the annotations do not. Producing them needs (a) rotation-corrected OCR (the current OCR cannot read most grocery photos), and (b) a human to verify every field/box on small-print, 36-line, multi-rate invoices. Labels pre-filled by the V1 model and "verified" only by me would be circular with the model being evaluated and would not meet the Phase 4 bar; I will not substitute them or synthetic labels.
