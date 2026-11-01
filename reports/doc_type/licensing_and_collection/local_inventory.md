# Local data inventory (read-only)

| disk root | files | in manifest | exact duplicate of a manifested file | undecodable | unmanifested unique |
|---|---|---|---|---|---|
| public/sroie | 973 | 958 | 15 | 0 | 0 |
| public/receiptbench | 873 | 867 | 6 | 0 | 0 |
| interim/my_receipts/images | 1252 | 1162 | 90 | 0 | 0 |
| public/indian_grocery_gst | 169 | 169 | 0 | 0 | 0 |
| public/funsd | 199 | 199 | 0 | 0 | 0 |
| public/rvl_cdip_160 | 0 | 0 | 0 | 0 | 0 |
| public/invoices_receipts_pt | 109 | 109 | 0 | 0 | 0 |
| interim/hf_medical_bills/images | 15 | 15 | 0 | 0 | 0 |
| public/in_financial_docs | 426 | 390 | 34 | 2 | 0 |
| public/cord | 0 | 0 | 0 | 0 | 0 |
| real | 200 | 198 | 2 | 0 | 0 |
| indian | 369 | 368 | 1 | 0 | 0 |
| indian_all/real | 837 | 0 | 837 | 0 | 0 |

## Real documents by source (manifest v2.1)
| source | rows | approved | flagged | needs mapping review | no OCR |
|---|---|---|---|---|---|
| cord | 998 | 998 | 0 | 998 | 0 |
| funsd | 199 | 199 | 0 | 199 | 0 |
| hf_medical | 15 | 0 | 0 | 15 | 0 |
| in_financial | 390 | 0 | 1 | 86 | 7 |
| in_grocery_gst | 169 | 169 | 0 | 169 | 0 |
| invoices_pt | 109 | 109 | 0 | 109 | 0 |
| manual_in | 368 | 368 | 16 | 352 | 0 |
| my_receipts | 1162 | 1162 | 18 | 1144 | 0 |
| receiptbench | 867 | 0 | 34 | 833 | 0 |
| rvl_cdip | 320 | 320 | 0 | 0 | 0 |
| sroie | 958 | 0 | 9 | 949 | 0 |
| user_real | 198 | 0 | 0 | 198 | 0 |

## Groups and exact action
| group | docs | action |
|---|---|---|
| real docs excluded only by licence/provenance | {'in_financial': 390, 'sroie': 958, 'hf_medical': 15, 'user_real': 198, 'receiptbench': 867} | none possible locally; each needs the exact permission in licensing_decisions.md (draft requests in PERMISSION_REQUESTS.md). Do not use until approved. |
| approved real docs whose label is only a collection-rule mapping | 2971 | human review of mapping_review_queue_v2_1.jsonl (286 records: 78 cue conflicts + spot checks) |
| real docs flagged for review (OCR cue conflicts) | 78 | human decision per record; keep out of evaluation until resolved |
| real docs without OCR text | 7 | in_financial only (blocked source); no OCR spent on unapproved data |
