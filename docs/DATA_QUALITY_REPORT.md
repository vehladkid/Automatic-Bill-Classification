# Data quality report (updated 2026-10-04, after the real-data round)

Numbers: [DATASET_STATS.md](DATASET_STATS.md), before/after + sources + rejected candidates: [REAL_DATA_ACQUISITION_2026-10-04.md](REAL_DATA_ACQUISITION_2026-10-04.md).

## Headline
* 7,924 docs after dedup: **5,494 real, 2,430 synthetic**. Task A: 5,852 trusted samples (3,422 real). 2,072 docs have no category (CORD 998, SROIE 669, transit/tourist and other unlabeled my_receipts 327, utility 56, RVL invoices 20, 2 ReceiptBench).
* Every class except **rent** has real data now (rent: none exists publicly). Weakest real coverage: gas (40 docs / 30 groups), water (71 / 30, mostly one Delhi Jal Board template), internet (63 / 27), medical (60 / 43; only 15 genuine Indian, license unstated).
* Real test is still small for weak classes (4-18 docs each). Sanity model (TF-IDF+LogReg on OCR text): real-only test acc 0.81 / macro-F1 0.68; synthetic-only test 1.00 (meaningless: templated).
* Leakage: independent audit finds 0 val/test images within pHash<=4 of a train image (37 val / 26 test at <=8, similar templates).

## Issues found
| Issue | Evidence | Handling |
|---|---|---|
| Mislabelled user file | `datasets/indian/water/Screenshot from 2026-09-06 21-03-10.png` is an HCG piped-gas invoice | `label_overrides.json` -> gas; source untouched. Please move it. |
| Very low layout diversity | `train`: 293 docs but 76 groups (162 DB-ticket template, 38 IRCTC); water: ~38 Delhi Jal Board bills; fuel handwritten cash bills; PT/my_receipts repeat merchants | grouped into one split each; `independent_groups` reported per class |
| Grouping gap found by audit | first rule left 10 val / 6 test docs within pHash<=4 of train (same-merchant receipts) | rule widened + filename/seller grouping; re-audited = 0 |
| Duplicates | 149 exact dropped, 15,121 near-dup pairs -> 440 groups | dropped / grouped |
| Corrupt images | 2 (`in_financial` Utility/53.jpg, 86.jpg) | dropped |
| Cross-source overlap | in_financial+manual_in 3, receiptbench+user_real 4, receiptbench+sroie 2, 3-way 1 | grouped |
| Roboflow-style filenames | user electricity/water zips look like a Roboflow export; a few bills appear 2-3x (augmented variants) | near-dup grouping keeps variants together |
| Duplicate upload | teammate's `grocery.zip` is byte-identical to the Kaggle grocery set | not unpacked |
| Label noise risk | SROIE/PT/my_receipts labels come from regex (`rule_high`), transit docs split train/transport by text rules; ReceiptBench types come from the authors | `label_quality` kept per row; `label_review_flags.json` |
| Foreign vs Indian | restaurant.zip is US; marked country US under `datasets/real/` | never counted as Indian |

## Limitations
* **Geography**: real Indian = your folders (electricity 90, gas 33, water 55, food 69, train 65, fuel 32, parking 22, internet/other small) + grocery 169 + hf medical 15 + scraped financial docs. Hotel, flight, bus, taxi are almost entirely foreign. GSTIN/CGST/SGST field extraction still has only synthetic + ReceiptBench Indian examples (tax_number JSON), not real field labels.
* **OCR**: GT for SROIE/CORD/FUNSD/synthetic; PDF text layer for my_receipts and some medical PDFs; RapidOCR (GPU) elsewhere. No OCR accuracy measured. Handwritten cash bills and tilted photos are weak.
* **Licensing**: 581 real documents have unstated or unverified rights (your zips, hf medical bills, scraped financial images, ReceiptBench images are real personal receipts without per-image consent info; one shows a social-media watermark). SROIE, FUNSD, RVL-CDIP research-only. -> academic/internal use only; no redistribution or commercial release of data/weights without a rights review.
* **Task D**: not built; `nikita2998/find-it-again-dataset` (SROIE forgeries) is the best public starting point.
* **Synthetic**: one renderer, 4 layout families, fictional vendors; hard negatives share positive layouts so only text separates them.
* Original SROIE/CORD benchmark splits are not preserved. Disk: ~4.3 GB duplicate CORD in `~/.cache/huggingface` (user's, untouched), pip CUDA libs ~2.5 GB in `.venv`.
