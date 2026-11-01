# Automatic Bill Classification (HR Expense Claim Intelligence)

A two-stage machine-learning system that reads the bills Indian employees submit (photos, scans, PDFs, e-mail attachments), decides what kind of bill each one is, extracts its key fields, estimates the income-tax effect, and routes uncertain cases to a human-resources (HR) reviewer. It was built as a PBL (project-based learning) project and comes with a measured comparison of text classifiers (TF-IDF, BiLSTM, LexRank, XLM-RoBERTa) and a full written report.

**Authors:** Tejvir Singh (24BAI1040), Chitwan Singh (24BAI1629), Dasari Durga Naga Sri Surya Vignesh (24BAI1589)

> **Important honesty notes.** The tax figures are estimates only (rules taken from secondary sources, not the official gazette). The second-stage classifier was trained on synthetic documents only; its 1.000 synthetic score reflects the generator, not real-world accuracy. The XLM-RoBERTa study is inconclusive on the real test set. All of this is stated again where relevant below.

---

## 1. What the system does (end-to-end walk-through)

```
 bill upload (JPG/PNG/PDF) or e-mail attachment
        |
 validation + page rendering
        |
 OCR (RapidOCR / PP-OCRv4, ONNX)  ->  words + boxes
        |
 STAGE 1  Task A: expense category (16 classes)
          calibrated fusion of  TF-IDF + logistic regression (0.5)
                                ConvNeXt-Tiny image model      (0.3)
                                LayoutLMv3 text+layout+image   (0.2)
          confidence >= tau (0.61) -> auto-processed, else -> HR review
        |
        +-- a real expense label  -> file under classified/<category>/
        |
        +-- "non_expense" / failed / low confidence
                 |
            STAGE 2  document-type classifier (12 tax-evidence types)
                     TF-IDF + logistic regression, trained on synthetic data
                     accepted only if probability >= 0.60, always flagged for HR review
                     else the bill stays "non_expense"
        |
 TASK B: field extraction (up to 34 fields: merchant, date, totals, GSTIN, CGST/SGST/IGST ...)
         hybrid of rules + gradient-boosted candidate ranker + LayoutLMv3 token classifier
        |
 document profile + tax-section suggestions + per-bill rebate estimate
        |
 Streamlit UI   |   Gmail ingestion worker   |   HR review folders + claims database
```

### The 16 expense labels (Task A)
flight, train, bus, transport (cab/auto), hotel, rent, food, electricity, water, gas, internet, fuel, parking, medical, miscellaneous, non_expense.

### The 12 second-stage document types
rent_receipt, rental_agreement, insurance_premium_receipt, school_fee_receipt, loan_interest_certificate, loan_statement, investment_certificate, meal_voucher_statement, ev_document, form_16, salary_slip, bank_statement.

## 2. Repository layout

| Path | Contents |
|---|---|
| `app/` | The application: `service.py` (pipeline glue), `streamlit_app.py` (UI), `second_classifier.py` (stage 2), `doc_profile.py`, `field_report.py`, `tax/` (tax estimate, quota, claims store, HR folders), `tax_ui.py`, `ingest/` (Gmail API / IMAP e-mail worker), `llm/` (optional local language-model assist), `selftest.py` |
| `scripts/` | Dataset pipeline (`run_pipeline.sh`, `build_dataset.py`, `create_splits.py`, ...), `taskA/` (text, image, LayoutLMv3, fusion and calibration training), `taskB/` (extraction models), `doc_type_new/` (stage-2 training and the saturation check), `make_pbl_report.py` (builds the report) |
| `config/` | Taxonomy, tax rules per financial year (`tax_rules/`), `bill_tax_quota.json`, `tax_suggest.json`, document-type schemas |
| `experiments/text_vs_neural/` | Controlled comparison on identical OCR text: TF-IDF, BiLSTM, BiLSTM + attention, LexRank + classifier, XLM-RoBERTa, LayoutLMv3, plus an OCR comparison (RapidOCR, CRNN, TrOCR) |
| `experiments/xlmr_vs_tfidf_final/` | Pre-registered final study: XLM-RoBERTa vs TF-IDF over 10 seeds, truncation audit, resource and cost analysis, power analysis, unseen synthetic and real test sets, TF-IDF-first hybrid cascade, verification scripts, `DECISION_SUMMARY.md`, `AUDIT.md` |
| `docs/` | Design and data documents (`ARCHITECTURE.md`, `DATASET.md`, `TASK_A_REPORT.md`, `TASK_B_REPORT.md`, `TAX_AND_LLM.md`, `EMAIL_INGESTION*.md`, `DEMO.md`) and the PBL report in `docs/report/` |
| `reports/` | Metrics and audits produced by the training and evaluation scripts |
| `tests/` | Unit and smoke tests (standard library only) |
| `demo/` | Synthetic sample bills for each category and the twelve new document types |
| `deploy/` | Example environment file and a systemd unit for the e-mail worker |

## 3. Models and why they were chosen

* **TF-IDF + logistic regression (text).** TF-IDF (term frequency - inverse document frequency) turns OCR text into a weighted vector of character 2-5-grams and word 1-2-grams; logistic regression learns one weight per feature per class. It scored 0.831 macro-F1 on real test documents, trains in under a minute, answers in about 2 ms on a CPU and is explainable. Code: `scripts/taskA/t1_text_baseline.py`; used again for stage 2 (`scripts/doc_type_new/train_new_classes.py`, `app/second_classifier.py`).
* **ConvNeXt-Tiny (image)** looks at the page picture (`scripts/taskA/t2_image_convnext.py`).
* **LayoutLMv3 (text + layout + image)** reads words with their positions (`scripts/taskA/t3_layoutlmv3.py`). Its weights are CC-BY-NC-SA (non-commercial).
* **Calibrated fusion** combines the three with per-model temperature scaling and a validation-chosen confidence threshold (`scripts/taskA/t4_fusion_calibration.py`).
* **Gradient-boosted decision trees (GBDT)** are used in Task B, not for classification: rules propose candidate values, one tree ensemble per field scores each candidate from its value, position and neighbouring words, and the field is left empty when the best score is below a threshold (`scripts/taskB/ranker.py`).
* **BiLSTM, BiLSTM + attention, LexRank + classifier, XLM-RoBERTa** are comparison models answering "why not an RNN / summarisation / transformer?" (`experiments/text_vs_neural/m2_rnn.py`, `m3_lexrank.py`, `m5_xlmr.py`; multi-seed study in `experiments/xlmr_vs_tfidf_final/`).

## 4. Results in brief

| Item | Result (real held-out documents unless stated) |
|---|---|
| Task A, calibrated fusion | accuracy 0.907, macro-F1 0.839; 93.0% auto-processed at 94.4% precision (tau = 0.61) |
| Task A, TF-IDF + LR alone | accuracy 0.883, macro-F1 0.831 |
| Task B, hybrid extractor | micro-F1 0.663 (rule baseline 0.444, GBDT ranker 0.546, LayoutLMv3 0.646) |
| Stage 2 (synthetic only) | 1.000 on a held-out synthetic layout - saturated, not a real-world estimate (see `docs/report`, Table IV) |
| XLM-RoBERTa vs TF-IDF | mean macro-F1 0.883 vs 0.831 (+0.052), 95% interval -0.005 to +0.127: **inconclusive**; about 45 times the CPU latency and 25 times the size |

Full numbers, statistics, limitations and the cost analysis are in `docs/report/PBL_Report.docx` / `.pdf` and in `experiments/xlmr_vs_tfidf_final/DECISION_SUMMARY.md`.

## 5. Tax module

`app/tax/` estimates the employee's yearly income tax from a saved salary profile (default profiles E1 and E2; HR user HR1) using rules in `config/tax_rules/`, suggests the tax section a bill may support, shows a per-bill rebate from the fixed quota in `config/bill_tax_quota.json`, and keeps claims, HR decisions and an audit log in a local SQLite database. A temporary demo switch (`demo_rebate` in `bill_tax_quota.json`) gives every label a notional allowance; set `enabled` to false for statutory-only behaviour. **Everything is an estimate; GST on a bill is not an income-tax rebate.** The store has no real authentication - it is a single-machine demo.

## 6. Setup and running

1. Python 3.12 with a virtual environment; install the pinned versions in `requirements.txt` (install `torch`/`torchvision` from the CUDA index you use) and `poppler-utils` for PDF input. A CUDA GPU is used if present.
2. **Data and models are not in this repository** (size and licences). Rebuild the dataset with `bash scripts/run_pipeline.sh`, train with the scripts in `scripts/taskA`, `scripts/taskB` and `scripts/doc_type_new`, and fetch pretrained weights with `scripts/taskA/fetch_weights.py`. Expected locations are `datasets/`, `models/taskA`, `models/taskB`, `models/doc_type_new`.
3. Start the interface: `bash app/run_app.sh`, then open http://127.0.0.1:8501.
4. Optional Gmail worker (never commit credentials): copy `deploy/worker.env.example` to `~/.config/hrbc/worker.env`, add your own OAuth client and authorised sender addresses (see `docs/EMAIL_INGESTION.md` and `docs/EMAIL_INGESTION_OPERATIONS.md`), then run `cd app && set -a && . ~/.config/hrbc/worker.env && set +a && ../.venv/bin/python -m ingest.run worker --gmail --interval 60`.
5. Tests: `.venv/bin/python -m unittest discover -s tests -v` and the backend self-test `HRBC_OUT=/tmp/hrbc_test .venv/bin/python app/selftest.py`.

## 7. Reproducing the experiments

* Comparison on identical OCR text: scripts `experiments/text_vs_neural/m1_tfidf.py` ... `m7_fusion_reference.py`, then `aggregate.py` for tables and figures.
* Final XLM-RoBERTa study: `experiments/xlmr_vs_tfidf_final/README.md` lists every command (training queue, statistics, unseen sets, hybrid, cost analysis, report build and verification). The analysis plan is pre-registered in `PREREGISTRATION.md` (with dated addenda).
* Report: `PYTHONPATH=<dir with python-docx and matplotlib> .venv/bin/python scripts/make_pbl_report.py` writes `docs/report/PBL_Report.docx`.

## 8. Limitations

Small real test sets for several classes (medical n = 6); Task B is a research-grade baseline (merchant name and invoice number are weak); stage 2 has seen no real documents and has no reject class beyond its 0.60 threshold; tax rules are unverified secondary sources; the e-mail worker and database are single-machine demo components without real authentication; some source datasets have licences that restrict redistribution, which is why datasets, trained models and real sample bills are excluded from this repository.
