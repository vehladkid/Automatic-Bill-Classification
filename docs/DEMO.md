# College demo: HR Bill Classifier (local)

Scope: upload a bill -> OCR -> expense category (Task A) -> field extraction (Task B V1) -> result page + saved file. Local only, one machine. This is a research demo, not a production system.

## Start / stop
```bash
cd /home/tejvir-singh/Desktop/ML-project
HRBC_OUT=$PWD/outputs_demo bash app/run_app.sh        # clean demo data in outputs_demo/ ; omit HRBC_OUT to use (and keep adding to) outputs/
# open http://127.0.0.1:8501   (first page load takes ~10 s while the models load)
# stop: Ctrl+C in that terminal
```
Needs the project `.venv`, `poppler-utils` (PDFs) and, for speed, the GPU. **Run only one copy of the app/models at a time** (6 GB GPU): do not run the test suite while the demo is open.
Backend self-test: `HRBC_OUT=/tmp/hrbc_test .venv/bin/python app/selftest.py` (26 checks). Demo smoke test (run with the app closed, ~30 s): `cd tests && ../.venv/bin/python -m unittest test_demo_smoke`.

## Demo script (about 6 minutes) - inputs are in `demo/sample_bills/`
1. Sidebar shows the models loaded. Page **Upload Bill**: drop `synthetic_electricity.png`, check the preview, click **Process Bill** (progress box, then it jumps to **Results**).
2. **Results**: left = the original with an optional box around any extracted field; right = *A. Expense classification* (category electricity, confidence, Auto vs HR-review routing, top-3), *Saved document* (path under `classified/electricity/`), *C. Extracted fields* (vendor, invoice number/date, subtotal, tax, total, currency, GSTIN, payment method with confidence/evidence), *Document profile* (document type is shown separately from the expense category), and the OCR text.
3. Upload `synthetic_fuel.png` and `synthetic_hotel.png` (categories fuel, hotel) and `cafe_receipt_cc0.pdf` (a PDF with a text layer, a real receipt; shows page tabs).
4. Upload `restaurant_receipt_owner_screenshot.png` (a real photo-like receipt): category food is confident but few fields are extracted. Say so: Task B is the frozen V1 research baseline (real-test micro-F1 about 0.66).
5. Failure handling: `blank_page_unreadable.png` -> "Classification not available: ocr_empty ... NOT filed in any category" (saved in `unclassified/`); `corrupt_image.png` -> "The image could not be decoded"; `unsupported_notes.txt` -> "text/plain files are not allowed".
6. **Classified Bills** (counts per category), then stop and restart the app: all results are still there (they are files under `outputs*/`).
Optional: **Tax Declaration / HR Review / Tax Summary / Email Inbox** exist (enter an employee ID in the sidebar). They are not part of this demo script.

## Honest limitations (say these if asked)
- The synthetic bills are fictional and come from this project's own generator, which also trained the models: their near-perfect extraction says nothing about real-world accuracy. Real-document field extraction is weak (merchant name and invoice number especially).
- Task A categories: 16 expense classes. The newer tax-document types (rent receipt, salary slip, Form 16, ...) are NOT trained classes; "document type" is a rule-based suggestion.
- PDFs: all pages are OCR'd (up to 10), classification and fields use page 1 only. Rotated photos are not auto-rotated.
- Identity in the sidebar is a demo selector, not authentication. One app instance at a time (GPU memory).
