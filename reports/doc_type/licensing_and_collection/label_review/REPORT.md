# Label review of the 286-record backlog

Method (`scripts/dataset_discovery/label_review.py`): OCR cue terms per document type (whole-word, distinct terms), strength of the original label's source (`manual_folder` / `source_label` / `manual_review` strong; `rule_high` / `unlabeled` weak), and the independent expense category used only as corroboration. No filenames or generator conventions are used. Only `doc_type` can be proposed for change; expense category (Task A) and tax evidence are never changed (tax: "insufficient evidence", candidates untouched). Manifests, the mapping queue, Task A, the Task B frozen test and snapshot, and V2 assets were not modified (a test checks their hashes).

## Result (actual counts)
| status | records |
|---|---|
| supported by OCR and source metadata | 78 |
| likely incorrect | 6 |
| ambiguous, needs human review | 70 |
| blocked by provenance/licensing | 132 |
| **total** | **286** |

Approved sources only (154 records): supported 78, likely incorrect 6, ambiguous 70. "Resolved" = supported (78) + the 1 high-confidence change applied to the overlay = **79 of 286**; the human queue holds **75** records (5 medium-confidence proposals + 70 ambiguous); 132 are blocked and untouched until their source is cleared. By queue origin: {'cue_conflict/ambiguous': 17, 'cue_conflict/blocked_provenance': 44, 'cue_conflict/likely_incorrect': 6, 'cue_conflict/supported': 11, 'spot_check/ambiguous': 53, 'spot_check/blocked_provenance': 88, 'spot_check/supported': 67}.

Blocked records by source (excluded in `config/approved_sources_v2.json`; their OCR verdict is stored as `evidence_verdict` — {'ambiguous': 54, 'supported': 74, 'likely_incorrect': 4} — but never applied):

| source | records |
|---|---|
| in_financial | 9 |
| receiptbench | 82 |
| sroie | 25 |
| hf_medical | 8 |
| user_real | 8 |

## Proposed changes (approved sources)
| id | record | old doc_type | proposed | confidence | status |
|---|---|---|---|---|---|
| LR1-0001 | manual_in__gas_Screenshot_2026-09-20_at_ | utility_bill | invoice | low | not applied: confidence below high |
| LR1-0002 | my_receipts__2022_gr_divani_20220917 | receipt | hotel_bill | medium | not applied: confidence below high |
| LR1-0003 | my_receipts__2022_ir_hilton_20220718 | receipt | hotel_bill | medium | not applied: confidence below high |
| LR1-0004 | my_receipts__2022_us_amtrak_20221016 | receipt | travel_document | medium | not applied: confidence below high |
| LR1-0005 | my_receipts__2023_de_motelone_20230819 | receipt | hotel_bill | medium | not applied: confidence below high |
| LR1-0006 | my_receipts__2024_de_motelone_20240627 | receipt | hotel_bill | medium | not applied: confidence below high |
| LR1-0007 | my_receipts__2024_nl_motelone_20240203 | travel_document | hotel_bill | high | overlay only (manifest untouched) |
| LR1-0012 | my_receipts__2018_ca_retail_safeway_2018 | invoice | receipt | low | not applied: confidence below high |

Only the high-confidence change is in the reversible overlay (`label_overlay_v1.jsonl`; `apply_label_overlay.py` produces effective labels and `revert_overlay` restores the originals exactly; stale or non-doc_type overlays are refused). It was checked visually: a Motel One Amsterdam hotel invoice typed `travel_document` while its expense category is `hotel` (human override) and its OCR has 4 hotel cues and none for travel; sibling Motel One documents are typed `hotel_bill`. The 5 medium proposals are hotel folios or a rail purchase receipt typed generic `receipt` (type overlap, e.g. Hilton Dublin folio, Amtrak ticket receipt): left for a human. 11 further proposals concern blocked sources (7 low, 4 medium) and are not applied.

## What stays ambiguous and why
- LPG/gas cylinder tax invoices labelled `utility_bill/gas` by the project owner (GSTIN, HSN, CGST cues): utility bill vs invoice overlap — 8 records.
- Generic `receipt` vs specific types (hotel, rail) — type overlap in the taxonomy rather than an error.
- Records with a single cue or little text (< 15 words) and a rule-derived label.
- `invoices_pt` fuel/receipt labels derived from a seller-name regex with no corroborating cue (16).
- 2 records where the expense category looks unusual for the type (flagged, Task A label not changed).

## Limits
The cue lexicon is a heuristic, not a classifier: "supported" means the label is consistent with OCR and source metadata, not verified by a person. Calibration was visual on 12 images (4 contested, 8 random "supported"); one over-generous rule (an `invoice` with zero invoice cues) was found that way and fixed (18 records moved from supported to ambiguous). Human review remains required before any label is called verified; the 154 approved-source records cover only 154 of the 2,971 collection-rule-labelled real documents (the queue was a prioritised sample).

Files: `review_results.jsonl` (all 286 with old labels, evidence, status, confidence, reason), `proposed_changes.jsonl`, `human_review_queue.jsonl` (ranked), `label_overlay_v1.jsonl`, `effective_labels.jsonl`, `summary.json`.
