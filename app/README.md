# HR Bill Classifier (demo app)
Upload bill -> OCR -> Task A expense category -> Task B V1 field extraction -> saved in `outputs/classified/<category>/`.

Launch (from the repo root):  `bash app/run_app.sh`   then open  http://127.0.0.1:8501
(Stop with Ctrl+C. First start loads the models, ~10 s. Needs the project `.venv`, a CUDA GPU is used if present, `poppler-utils` for PDFs.)

Files: `app/service.py` (pipeline glue, reuses scripts/taskA/predict.py and scripts/taskB/predict.py), `app/streamlit_app.py` (UI), `app/selftest.py` (backend tests: `HRBC_OUT=/tmp/hrbc_test .venv/bin/python app/selftest.py`).

Output layout (all under `outputs/`, never inside datasets/ or reports/):
- `classified/<category>/<doc_id>__<safe_name>.<ext>` - original upload (never overwritten)
- `unclassified/...` - OCR found no text or classification failed (no category is guessed)
- `results/<doc_id>/` - `record.json` (everything), `ocr.txt`, `ocr_page1_words.json`, `task_b.json`, `page_<n>.jpg`

Limits: JPG/PNG/PDF up to 20 MB; PDFs: OCR on up to 10 pages, classification + extraction on page 1 only. Task B is the frozen V1 research baseline (weak on merchant name / invoice number; no multi-code HSN, taxable value, discount or round-off). Photos rotated by 90 degrees are not auto-rotated (EXIF rotation only).

## Added: LLM assist + HR tax module
Pages: Upload Bill, Results, Classified Bills, **Tax Declaration** (employee), **HR Review**, **Tax Summary**. See `docs/TAX_AND_LLM.md` for design, rules verification status, setup of the optional local LLM (`HRBC_LLM_MODE=off|auto|always`, `config/env.example`) and tests (`.venv/bin/python -m unittest discover -s tests -v`).
Claims DB: `outputs/hr/hr.db`. Tax output is an estimate only.

## College demo
Start commands, demo script and honest limitations: `docs/DEMO.md` (inputs in `demo/sample_bills/`, smoke test `tests/test_demo_smoke.py`).
