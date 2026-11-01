"""Backend self-test for the HR Bill Classifier service (uses a temporary output dir via HRBC_OUT, never the real outputs/).
Run:  HRBC_OUT=/tmp/hrbc_test .venv/bin/python app/selftest.py     Reads sample bills from datasets/ (read-only)."""
import os, sys, json, tempfile, shutil
from pathlib import Path
assert os.environ.get("HRBC_OUT"), "set HRBC_OUT to a scratch directory"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import service as S
ROOT = S.ROOT; R = []
def check(name, ok, detail=""):
    R.append((name, bool(ok))); print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
svc = S.BillService().load(); print("model load errors:", svc.errors)
food = ROOT / "datasets/indian/food/Screenshot from 2026-09-20 14-01-02.png"; elec = ROOT / "datasets/indian/electricity/adrija_bescom_jpg.rf.pXvXN5v0rENFr4EdHbdW.jpg"
pdf = ROOT / "datasets/public/hf_medical_bills/train_sample_2.pdf"
# 1. normal image -> category folder, OCR text, Task A label from the real label set, Task B fields
r = svc.process(food.read_bytes(), food.name)
f = S.OUT / r["saved_path"]
check("image: processed, status done", r["status"] == "done", r["status"])
check("image: OCR text non-empty", (S.OUT / r["result_dir"] / "ocr.txt").read_text().strip() != "", f"{r['ocr_words_page1']} words")
check("image: Task A label in real taxonomy", r["task_a"] and r["task_a"]["label"] in svc.labels, str(r["task_a"] and (r["task_a"]["label"], r["task_a"]["confidence"])))
check("image: file saved under classified/<label>/", f.exists() and f.parent.name == r["task_a"]["label"] and f.parent.parent.name == "classified", str(f.relative_to(S.OUT)))
check("image: original bytes preserved", f.read_bytes() == food.read_bytes())
check("image: Task B fields returned (dict)", isinstance(r["task_b"], dict) and "fields" in r["task_b"], f"{len(r['task_b']['fields'])} fields")
# 2. duplicate upload does not overwrite
r2 = svc.process(food.read_bytes(), food.name)
check("duplicate: new id and new file, old file intact", r2["doc_id"] != r["doc_id"] and (S.OUT / r2["saved_path"]).exists() and f.exists() and S.OUT / r2["saved_path"] != f)
check("duplicate: warns about identical earlier upload", any("already processed" in w for w in r2["warnings"]))
# 3. hostile filename
r3 = svc.process(elec.read_bytes(), "../../../etc/pass wd;rm -rf.jpg")
p3 = (S.OUT / r3["saved_path"]).resolve()
check("path traversal: file stays inside outputs/", str(p3).startswith(str(S.OUT.resolve())) and ".." not in r3["saved_filename"] and "/" not in r3["saved_filename"], r3["saved_filename"])
check("path traversal: original filename kept in metadata", r3["original_filename"].startswith("../../../etc"))
# 4. multi-page PDF
r4 = svc.process(pdf.read_bytes(), "multi.pdf")
txt4 = (S.OUT / r4["result_dir"] / "ocr.txt").read_text()
check("pdf: all pages OCR'd and labelled", r4["n_pages"] == 3 and r4["pages_ocred"] == 3 and "===== Page 3 =====" in txt4, f"n_pages={r4['n_pages']}")
check("pdf: page-1-only limit disclosed", any("page 1 only" in w for w in r4["warnings"]))
# 5. Task B failure keeps OCR + Task A
orig = svc.B.run
svc.B.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated Task B failure"))
r5 = svc.process(food.read_bytes(), "tb_fail.png"); svc.B.run = orig
check("Task B failure: status done_without_fields, error recorded", r5["status"] == "done_without_fields" and "simulated" in (r5["task_b_error"] or ""))
check("Task B failure: Task A + OCR + category still saved", r5["task_a"] is not None and (S.OUT / r5["saved_path"]).exists() and (S.OUT / r5["result_dir"] / "ocr.txt").read_text().strip() != "")
# 6. Task A failure -> unclassified, not an arbitrary category
origA = svc.A.predict
svc.A.predict = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated Task A failure"))
r6 = svc.process(food.read_bytes(), "ta_fail.png"); svc.A.predict = origA
check("Task A failure: saved in unclassified/, no category", r6["category"] is None and (S.OUT / r6["saved_path"]).parent.name == "unclassified" and r6["status"] == "classification_failed")
check("Task A failure: OCR + Task B results still present", r6["task_b"] is not None and r6["ocr_chars"] > 0)
# 7. blank image -> OCR empty -> unclassified
import io
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (400, 300), "white").save(buf, "PNG")
r7 = svc.process(buf.getvalue(), "blank.png")
check("blank image: ocr_empty, unclassified, nothing invented", r7["status"] == "ocr_empty" and r7["category"] is None and r7["task_a"] is None)
# 8. invalid uploads
def rejects(name, data, fn, frag):
    try: svc.process(data, fn); check(name, False, "no error")
    except S.UserError as e: check(name, frag in str(e), str(e)[:90])
rejects("reject unsupported extension", b"hello", "x.txt", "Unsupported file type")
rejects("reject empty file", b"", "x.jpg", "empty")
rejects("reject non-image content", b"not an image at all", "x.png", "not a JPEG")
rejects("reject extension/content mismatch", elec.read_bytes(), "x.png", "extension says")
rejects("reject oversized (>20 MB)", b"\xff\xd8\xff" + b"0" * (S.MAX_BYTES + 10), "big.jpg", "too large")
rejects("reject truncated/corrupt JPEG", elec.read_bytes()[:3000], "t.jpg", "could not be decoded")
rejects("reject corrupt PDF", b"%PDF-1.4\ngarbage", "bad.pdf", "PDF could not be read")
# 8b. second-stage classifier (models/doc_type_new): Task A has no class for these; synthetic-only model
rent = ROOT / "datasets/synthetic_doctype/v1/images/rent_receipt/synth_rent_receipt_0000_p1.jpg"
r8 = svc.process(rent.read_bytes(), rent.name); sc = r8.get("second_classifier") or {}
check("second stage: loaded without error", svc.C is not None and "second_classifier" not in svc.errors, str(svc.errors.get("second_classifier", "")))
check("second stage: Task A non_expense -> rent_receipt accepted and filed", r8["task_a"]["label"] == "rent_receipt" and r8["task_a"]["source"] == "second_classifier" and r8["task_a_first_stage"]["label"] == "non_expense" and sc.get("accepted") and sc["label"] == "rent_receipt" and r8["category"] == "rent_receipt" and (S.OUT / r8["saved_path"]).parent.name == "rent_receipt", str((sc.get("label"), sc.get("confidence"))))
check("second stage: silent (no extra banner), routed to HR review", not any("second classifier" in w for w in r8["warnings"]) and r8["task_a"]["route"] == "hr_review")
check("second stage: not run for a confident expense bill", r.get("second_classifier") is None and r["category"] == r["task_a"]["label"])
check("second stage: below threshold -> no label, stays non_expense", svc.C.predict("Thank you for visiting. Have a nice day.")["label"] is None)
check("second stage: empty text -> no label", svc.C.predict("")["label"] is None)
# 9. records / listing reflect only real saved files
recs = S.list_records(); ok = all((S.OUT / x["saved_path"]).exists() for x in recs)
check("listing: every record points at an existing file; rejected uploads left no record", ok and len(recs) == 8, f"{len(recs)} records")
check("no leftover dirs from rejected uploads", len([d for d in S.RESULTS.iterdir()]) == len(recs))
print(f"\n{sum(ok for _, ok in R)}/{len(R)} checks passed"); sys.exit(0 if all(ok for _, ok in R) else 1)
