# Taxonomy

Three **independent** label axes are stored per document (never conflate them):

| Axis | Field | Values | Used by |
|---|---|---|---|
| What was bought | `expense_category` (+ `parent_group`) | see below | Task A |
| What kind of paper it is | `document_type` | receipt, invoice, utility_bill, travel_document, hotel_bill, fuel_receipt, medical_bill, other, non_expense_document | Task C |
| Provenance / trust | `label_quality`, `synthetic`, `collection` | see DATASET.md | filtering |

## `expense_category` (16 values) and parent groups

| parent_group | expense_category | notes |
|---|---|---|
| travel | flight, train, bus, **transport** | `transport` = cab / auto / taxi / ride-hailing **and local transit tickets (metro, subway, tram)** - decision 2026-10-05; matches the `datasets/indian/transport/` folder |
| accommodation | hotel | guest-house / lodging invoices fall here too |
| food | food | restaurants, meals, catering (subcategory optional) |
| utilities | electricity, water, gas, internet | `internet` includes telecom/postpaid |
| transport_fuel | fuel, parking | petrol / diesel / EV charging are all `fuel` |
| medical | medical | hospital, pharmacy/chemist, diagnostics |
| other_business | rent, miscellaneous | `miscellaneous` = office supplies, equipment, groceries, subscriptions |
| non_expense | non_expense | letters, forms, memos... negatives so a gate model can reject non-bills |
| (not a class) | `unlabeled` | real receipts whose source gives no category (CORD, unmatched SROIE, RVL invoices). Kept for Tasks B/C, **excluded from Task A**. |

## Why this and not the brief's hierarchy verbatim

* Public taxonomies surveyed: SROIE (no category; fields only), CORD (menu/total schema, no category), RVL-CDIP (16 *document* classes - an
  invoice is a doc type, not a spending category), Expensify/Zoho/SAP-Concur-style expense types (Meals, Fuel, Lodging, Airfare,
  Ground transport, Office supplies, Telecom, Utilities, Medical, Misc.). None is trainable on public data; the HR-style list is.
* **Flat 15+1 leaf label for the classifier, hierarchy as metadata.** A softmax over leaves trains best with the data we can get;
  `parent_group` lets the later policy engine reason at group level without retraining.
* Merged vs. the brief: *Hotel* appears under both Travel and Accommodation in the brief -> single `hotel` leaf (a duplicate leaf
  would make labels contradictory). Petrol+Diesel+EV -> `fuel` (indistinguishable on most receipts; sub-type kept in `subcategory`/fields).
  Restaurant/Meals/Catering -> `food`. "Hospital/Pharmacy/Services" -> `medical` (doc type `medical_bill` either way).
* Added `non_expense` (needed at inference to reject non-bills) and `unlabeled` (honesty about missing labels).
* Document type is derived from the category only for manual/synthetic sources (default map in `scripts/common.py`);
  public receipts carry their own (`receipt`/`invoice`/`non_expense_document`).
