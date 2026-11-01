"""Thin service layer for the HR Bill Classifier demo.
Upload bytes -> validate -> pages -> OCR (existing RapidOCR/GPU engine, run once) -> Task A (scripts/taskA/predict.py::Predictor)
-> Task B V1 (scripts/taskB/predict.py::Extractor) -> save under outputs/. No model is trained or modified; nothing here touches datasets/ or reports/.

Outputs (all under <repo>/outputs/):
  classified/<category>/<doc_id>__<safe_name>.<ext>   the ORIGINAL uploaded bytes, never overwritten (exclusive create)
  unclassified/<doc_id>__<safe_name>.<ext>           OCR returned no text / classification failed -> NOT put in any category
  results/<doc_id>/record.json, ocr.txt, ocr_page1_words.json, task_b.json, page_<n>.jpg   sidecars
"""
import sys, io, os, re, json, time, uuid, hashlib, shutil, subprocess, threading, importlib.util, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("HRBC_OUT", ROOT / "outputs"))   # override only for tests
CLASSIFIED, UNCLASSIFIED, RESULTS = OUT / "classified", OUT / "unclassified", OUT / "results"
for _p in (ROOT / "scripts", ROOT / "scripts" / "taskA", ROOT / "scripts" / "taskB"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 60_000_000
MAX_PDF_PAGES = 10
LONG_SIDE = 3000
OCR_MAX_SIDE = int(os.environ.get("HRBC_OCR_MAX_SIDE", "2400"))       # RapidOCR down-scales pages above this long side. 2400 (was 1600) measured better on validation: reports/extraction_audit
OCR_REPAIR = os.environ.get("HRBC_OCR_REPAIR", "1") != "0"           # app/ocr_text.py: re-insert the space RapidOCR drops between a date and the time after it
ALLOWED = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".pdf": "pdf"}
LOCK = threading.Lock()

class UserError(Exception):
    """Problem with the upload itself (shown as an error, nothing is saved)."""

def _now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

def safe_name(name):
    """Strip any path, keep [A-Za-z0-9._-], bounded length. Never returns an empty or dot-only stem."""
    base = Path(str(name).replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._-")[:60]
    return stem or "bill"

def sniff(data):
    if data[:3] == b"\xff\xd8\xff": return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n": return "png"
    if data[:5] == b"%PDF-": return "pdf"
    return None

def validate_upload(data, filename):
    """Returns (kind, ext). Raises UserError with a readable message."""
    ext = Path(str(filename)).suffix.lower()
    if ext not in ALLOWED: raise UserError(f"Unsupported file type '{ext or '(none)'}'. Upload a JPG, JPEG, PNG or PDF.")
    if not data: raise UserError("The uploaded file is empty.")
    if len(data) > MAX_BYTES: raise UserError(f"File is too large ({len(data) / 1e6:.1f} MB). Maximum is {MAX_BYTES // 1_000_000} MB.")
    kind = sniff(data)
    if kind is None: raise UserError("The file content is not a JPEG, PNG or PDF (unreadable or corrupt, or the extension is wrong).")
    if kind != ALLOWED[ext]: raise UserError(f"The file extension says {ext} but the content is {kind.upper()}; please re-save it with the right extension.")
    return kind, ext

def _pdf_pages(path):
    r = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, timeout=60)
    m = re.search(r"Pages:\s+(\d+)", r.stdout)
    if r.returncode != 0 or not m: raise UserError("The PDF could not be read (corrupt or password-protected).")
    return int(m.group(1))

def render_pdf_page(path, n, out_jpg):
    from pdf_utils import DPI
    r = subprocess.run(["pdftoppm", "-r", str(DPI), "-f", str(n), "-l", str(n), "-jpeg", "-jpegopt", "quality=90", "-singlefile", str(path), str(Path(out_jpg).with_suffix(""))], capture_output=True, timeout=120)
    if r.returncode != 0 or not Path(out_jpg).exists(): raise UserError(f"PDF page {n} could not be rendered: {r.stderr.decode(errors='ignore')[:200]}")

def preview_first_page(data, kind):
    """Bytes of a JPEG preview (for the upload screen). Raises UserError if unreadable."""
    if kind == "pdf":
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.pdf"; p.write_bytes(data); _pdf_pages(p); render_pdf_page(p, 1, Path(d) / "p.jpg"); return (Path(d) / "p.jpg").read_bytes()
    try:                                  # fully decode now: a corrupt image must produce a readable message here, not a traceback in the preview widget
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        with Image.open(io.BytesIO(data)) as im: im.load()
    except Exception as e: raise UserError(f"The image could not be decoded ({type(e).__name__}); it may be corrupt or too large.")
    return data

def _load_image(path):
    try:
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        with Image.open(path) as im: im.verify()
        im = Image.open(path); im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA"): im = im.convert("RGBA" if "transparency" in im.info or im.mode in ("LA", "PA") else "RGB")
    except UserError: raise
    except Exception as e: raise UserError(f"The image could not be decoded ({type(e).__name__}); it may be corrupt or too large.")
    if max(im.size) > LONG_SIDE:
        s = LONG_SIDE / max(im.size); im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return im

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

class BillService:
    """Models are loaded once (load()). Each component reports its own load error so a failure in one does not hide the others."""
    def __init__(self):
        os.umask(0o077)      # everything this service writes (results, OCR text, page images, originals) is private to the service account
        self.loaded = False; self.errors = {}; self.A = None; self.B = None; self.C = None; self.ocr_engine = None; self.labels = []
        self.tauA = None

    def load(self):
        if self.loaded: return self
        t0 = time.time()
        try:
            self.ocr_engine = self._make_engine()
        except Exception as e: self.errors["ocr"] = f"{type(e).__name__}: {e}"
        try:
            mA = _load_module("taskA_predict", ROOT / "scripts/taskA/predict.py"); self.A = mA.Predictor(); self.labels = list(self.A.cfg["labels"]); self.tauA = self.A.cfg.get("tau")
        except Exception as e: self.errors["task_a"] = f"{type(e).__name__}: {e}"
        try:
            mB = _load_module("taskB_predict", ROOT / "scripts/taskB/predict.py"); self.B = mB.Extractor(use_layoutlm=True); self.mB = mB
        except Exception as e: self.errors["task_b"] = f"{type(e).__name__}: {e}"
        try:
            from second_classifier import SecondClassifier; self.C = SecondClassifier()
        except Exception as e: self.errors["second_classifier"] = f"{type(e).__name__}: {e}"
        try:
            from tax import quota as Q; Q.seed_default_profiles()      # default employee profiles E1/E2 (insert-only), so e-mail and upload flows always find a profile
        except Exception as e: self.errors["default_profiles"] = f"{type(e).__name__}: {e}"
        self.load_seconds = round(time.time() - t0, 1); self.loaded = True; return self

    # ---------------- OCR ----------------
    @staticmethod
    def _make_engine():
        """Same RapidOCR models and settings as scripts/taskA/run_ocr_gpu.get_engine() (which the training/eval scripts keep using), except max_side_len."""
        if OCR_MAX_SIDE == 1600:
            from run_ocr_gpu import get_engine; return get_engine()
        from run_ocr import _gpu_libs; _gpu_libs()
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, intra_op_num_threads=4, inter_op_num_threads=1, max_side_len=OCR_MAX_SIDE)

    def _ocr(self, path):
        """OCR on the page FILE (same call as scripts/taskA/predict.py::ocr), so Task A sees the same OCR input as in training/CLI use."""
        if self.ocr_engine is None: raise RuntimeError("OCR engine is not available: " + self.errors.get("ocr", "not loaded"))
        res, _ = self.ocr_engine(str(path))
        segs = [{"text": t, "bbox": [int(min(p[0] for p in b)), int(min(p[1] for p in b)), int(max(p[0] for p in b)), int(max(p[1] for p in b))], "conf": round(float(c), 3)} for b, t, c in (res or [])]
        if OCR_REPAIR:
            from ocr_text import repair_segments
            segs = repair_segments(segs)
        return segs

    @staticmethod
    def _text_from_segments(segs, img_h):
        from words import split_words, reading_order
        from baseline import lines_of
        ws = reading_order(split_words(segs))
        if not ws: return "", ws
        L = lines_of([w["text"] for w in ws], [w["bbox"] for w in ws])
        return "\n".join(l["text"] for l in L), ws

    # ---------------- main entry ----------------
    def process(self, data, filename, owner=None, source=None):
        """owner: employee id the document belongs to (None = legacy/unowned demo upload). source: provenance dict (e.g. email message id) - metadata only."""
        with LOCK:
            self.load()
            kind, ext = validate_upload(data, filename)
            doc_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
            want = (source or {}).get("doc_id")                      # deterministic id supplied by the ingestion worker: makes crash-and-retry idempotent
            if want:
                if not re.fullmatch(r"\d{8}T\d{6}_[0-9a-f]{8}", str(want)): raise UserError("invalid document id")
                done = RESULTS / want / "record.json"
                if done.exists():                                    # already fully processed: return it, create nothing
                    try:
                        prev_rec = json.loads(done.read_text())
                        if prev_rec.get("finished_at"): return prev_rec
                    except Exception: pass
                shutil.rmtree(RESULTS / want, ignore_errors=True)    # partial leftovers of an interrupted run of THIS id
                for stale in list(CLASSIFIED.glob(f"*/{want}__*")) + list(UNCLASSIFIED.glob(f"{want}__*")): stale.unlink(missing_ok=True)
                doc_id = want
            sname = safe_name(filename); rdir = RESULTS / doc_id; rdir.mkdir(parents=True, exist_ok=False)
            orig_tmp = rdir / f"original{ext}"
            with open(orig_tmp, "xb") as f: f.write(data)
            rec = {"doc_id": doc_id, "original_filename": str(filename), "safe_name": sname, "kind": kind, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                   "uploaded_at": _now(), "owner": owner, "source": source or {"channel": "web_upload"}, "status": "processing", "warnings": [], "task_a": None, "task_a_error": None, "task_b": None, "task_b_error": None, "ocr_error": None,
                   "n_pages": 1, "pages_ocred": 0, "models_used_page": 1, "saved_path": None, "category": None, "result_dir": str(rdir.relative_to(OUT)),
                   "pipeline": "Task A fusion (models/taskA) + Task B V1 hybrid (models/taskB, frozen). Predictions only - not ground truth, not human-verified."}
            prev = [r["doc_id"] for r in list_records() if r.get("sha256") == rec["sha256"]]
            if prev: rec["warnings"].append(f"Identical file was already processed as {prev[-1]}; saved again under a new id (nothing overwritten).")
            try:
                pages = self._prepare_pages(orig_tmp, kind, rdir, rec)
            except UserError as e:
                shutil.rmtree(rdir, ignore_errors=True); raise
            # --- OCR (all pages; page-1 words feed the models) ---
            ocr_chunks = []; seg1 = None; ws1 = []
            for i, pth in enumerate(pages, 1):
                try:
                    img = Image.open(pth).convert("RGB")
                    if i == 1 and kind == "pdf":
                        from pdf_utils import words as pdf_words
                        tl = pdf_words(orig_tmp); segs = [dict(s, conf=1.0) for s in tl] if tl else self._ocr(pth); rec["ocr_source_page1"] = "pdf_text_layer" if tl else "rapidocr"
                    else:
                        segs = self._ocr(pth); rec.setdefault("ocr_source_page1", "rapidocr")
                    txt, ws = self._text_from_segments(segs, img.height); rec["pages_ocred"] += 1
                    ocr_chunks.append((i, txt))
                    if i == 1:
                        seg1, ws1 = segs, ws; (rdir / "ocr_page1_words.json").write_text(json.dumps(segs, ensure_ascii=False))
                except Exception as e:
                    rec["ocr_error"] = f"page {i}: {type(e).__name__}: {e}"; rec["warnings"].append(f"OCR failed on page {i}.")
                    if i == 1: break
            full = "\n\n".join((f"===== Page {i} =====\n" if len(pages) > 1 else "") + t for i, t in ocr_chunks)
            (rdir / "ocr.txt").write_text(full, encoding="utf-8"); rec["ocr_chars"] = len(full); rec["ocr_words_page1"] = len(ws1)
            if rec["n_pages"] > len(pages): rec["warnings"].append(f"PDF has {rec['n_pages']} pages; only the first {len(pages)} were OCR'd.")
            if rec["n_pages"] > 1: rec["warnings"].append("Classification and field extraction use page 1 only (current pipeline limit); OCR text covers all processed pages.")
            page1 = pages[0]
            try: return self._models(rec, orig_tmp, ext, rdir, page1, seg1, ws1, doc_id)
            except Exception as e:      # unexpected: never lose the upload, never invent a category
                rec["status"] = "error"; rec["task_a_error"] = rec["task_a_error"] or f"unexpected: {type(e).__name__}: {e}"; rec["task_a"] = None
                return self._finish(rec, orig_tmp, ext, None)

    def _models(self, rec, orig_tmp, ext, rdir, page1, seg1, ws1, doc_id):
        if True:
            if seg1 is None or not ws1:
                rec["status"] = "ocr_failed" if rec["ocr_error"] else "ocr_empty"
                if not rec["ocr_error"]: rec["ocr_error"] = "OCR found no text on page 1."
                return self._finish(rec, orig_tmp, ext, None)
            # --- Task A ---
            cat = None
            if self.A is None: rec["task_a_error"] = "Task A model not loaded: " + self.errors.get("task_a", "unknown")
            else:
                try:
                    ws_a = [(s["text"], s["bbox"]) for s in seg1]
                    r = self.A.predict(str(page1), words=ws_a)
                    if r["label"] not in self.labels: raise RuntimeError(f"unexpected label {r['label']!r}")
                    rec["task_a"] = {"label": r["label"], "parent_group": r["parent_group"], "confidence": r["confidence"], "route": r["route"], "top3": r["top3"], "tau": self.tauA}
                    cat = r["label"]
                    if r["route"] != "auto": rec["warnings"].append(f"Confidence {r['confidence']:.2f} is below the routing threshold {self.tauA}: HR review recommended. Saved under the predicted category.")
                except Exception as e:
                    rec["task_a_error"] = f"{type(e).__name__}: {e}"; rec["task_a_trace"] = traceback.format_exc()[-600:]
            # --- second-stage classifier: only when Task A found no expense category (non_expense / failed / low confidence) ---
            ta = rec.get("task_a")
            if self.C is not None and (ta is None or ta["label"] == "non_expense" or ta["route"] != "auto"):
                try:
                    c2 = self.C.predict((rdir / "ocr.txt").read_text(encoding="utf-8"))
                    rec["second_classifier"] = dict(c2, model="models/doc_type_new (synthetic-only TF-IDF+LR)", trigger="task_a_failed" if ta is None else ("non_expense" if ta["label"] == "non_expense" else "low_confidence"))
                    if c2["accepted"]:
                        if ta is None or ta["label"] == "non_expense":     # Task A has no expense category: the second-stage type replaces it everywhere (no 'non_expense' is shown)
                            cat = c2["label"]
                            rec["task_a_first_stage"] = ta
                            rec["task_a"] = {"label": cat, "parent_group": "second_stage_document_type", "confidence": c2["confidence"], "route": "hr_review", "top3": c2["top3"], "tau": c2["min_prob"], "source": "second_classifier"}
                except Exception as e: rec["second_classifier_error"] = f"{type(e).__name__}: {e}"
            elif self.C is None: rec["second_classifier_error"] = "second classifier not loaded: " + self.errors.get("second_classifier", "unknown")
            # --- Task B (independent of Task A) ---
            if self.B is None: rec["task_b_error"] = "Task B model not loaded: " + self.errors.get("task_b", "unknown")
            else:
                try:
                    W, H = Image.open(page1).size
                    doc = dict(document_id=doc_id, words=[w["text"] for w in ws1], boxes=[w["bbox"] for w in ws1], width=W, height=H, country="?", supported_fields=list(self.mB.FIELDS), fields={}, labels=[])
                    fields = self.B.run(doc, Image.open(page1).convert("RGB"))
                    from field_sanity import drop_implausible
                    fields, dropped = drop_implausible(fields)
                    rec["task_b"] = {"fields": fields, "model": "Task B V1 hybrid (frozen baseline)", "bbox_space": "page-1 image pixels"}
                    if dropped: rec["task_b"]["dropped"] = dropped; rec["warnings"].append("Total amount read as 0 was discarded (shown as missing).")
                    (rdir / "task_b.json").write_text(json.dumps(rec["task_b"], indent=1, ensure_ascii=False, default=float))
                except Exception as e:
                    rec["task_b_error"] = f"{type(e).__name__}: {e}"; rec["task_b_trace"] = traceback.format_exc()[-600:]
            # --- optional local-LLM enhancement (validated against OCR; V1 output is never overwritten) ---
            v1f = ((rec.get("task_b") or {}).get("fields")) or {}
            try:
                from llm.enhance import enhance
                from llm.config import settings as llm_settings
                lr = enhance(seg1, Image.open(page1).size, v1f, Image.open(page1).convert("RGB"), cfg=llm_settings())
                rec["llm"] = {k: v for k, v in lr.items() if k not in ("fields",)}
                if lr["status"] == "ok":
                    rec["task_b_enhanced"] = {"fields": lr["fields"], "review_fields": lr["review_fields"]}
                    (rdir / "task_b_enhanced.json").write_text(json.dumps(rec["task_b_enhanced"], indent=1, ensure_ascii=False, default=float))
                elif lr["status"] in ("unavailable", "bad_output", "privacy_blocked"): rec["warnings"].append(f"LLM assist not used ({lr['status']}): {lr.get('error', '')[:160]}. Showing the standard extraction.")
            except Exception as e:
                rec["llm"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            # --- tax-section suggestions (suggestion only; employee confirms, HR verifies) ---
            try:
                from tax.suggest import suggest
                final = (rec.get("task_b_enhanced") or {}).get("fields") or v1f
                rec["tax_suggestions"] = suggest((rdir / "ocr.txt").read_text(encoding="utf-8"), final, cat)
            except Exception as e:
                rec["tax_suggestions"] = []; rec["warnings"].append(f"Tax suggestion step failed: {type(e).__name__}")
            # --- structured document profile + per-field report (rules_only for the employee-document labels) ---
            try:
                from doc_profile import classify as profile_classify
                import field_report as FR
                final = (rec.get("task_b_enhanced") or {}).get("fields") or v1f
                ocr_txt = (rdir / "ocr.txt").read_text(encoding="utf-8")
                rec["profile"] = profile_classify(ocr_txt, final, rec.get("task_a_first_stage") or rec.get("task_a"), (rec.get("llm") or {}).get("ext_fields"), (rec.get("llm") or {}).get("llm_classification"))
                rec["field_report"] = FR.build(final, seg1, expected=[x for x in ("invoice_date", "total_amount") if rec["profile"]["document_type"] != "unknown"], conflicts=(rec.get("llm") or {}).get("conflicts") or [])
            except Exception as e:
                rec["profile"] = None; rec["warnings"].append(f"Document profile step failed: {type(e).__name__}")
            if cat is None: rec["status"] = "classification_failed"
            else: rec["status"] = "done" if not rec["task_b_error"] else "done_without_fields"
            return self._finish(rec, orig_tmp, ext, cat)

    def _prepare_pages(self, orig, kind, rdir, rec):
        pages = []
        if kind == "pdf":
            n = _pdf_pages(orig); rec["n_pages"] = n
            for i in range(1, min(n, MAX_PDF_PAGES) + 1):
                out = rdir / f"page_{i}.jpg"; render_pdf_page(orig, i, out); pages.append(out)   # pdftoppm JPEG, as in scripts/pdf_utils.py
        else:
            im = _load_image(orig); out = rdir / "page_1.png"; im.save(out); pages.append(out)   # PNG = lossless: no JPEG re-compression before OCR/Task A
        return pages

    def _finish(self, rec, orig_tmp, ext, cat):
        fname = f"{rec['doc_id']}__{rec['safe_name']}{ext}"
        dest_dir = (CLASSIFIED / cat) if cat else UNCLASSIFIED
        if cat and not re.fullmatch(r"[a-z][a-z0-9_]*", cat): raise RuntimeError("unsafe category name")
        dest_dir.mkdir(parents=True, exist_ok=True); dest = dest_dir / fname
        if dest.exists(): raise RuntimeError(f"refusing to overwrite {dest}")
        shutil.move(str(orig_tmp), str(dest))
        rec["category"] = cat; rec["saved_path"] = str(dest.relative_to(OUT)); rec["saved_filename"] = fname; rec["finished_at"] = _now()
        (OUT / rec["result_dir"] / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=float))
        try:
            from tax import quota as _Q; _Q.submit_bills_to_hr([rec])    # tax-relevant bill of an employee -> submitted claim, so it appears under HR Review > Needs approval
        except Exception: pass
        try:
            from tax import hr_folders; hr_folders.sync()          # keep <outputs>/hr_review/ (processed / pending_approval / approved / rejected ...) current
        except Exception: pass
        return rec

def list_records():
    out = []
    if RESULTS.exists():
        for d in sorted(RESULTS.iterdir()):
            f = d / "record.json"
            if f.exists():
                try: out.append(json.loads(f.read_text()))
                except Exception: pass
    return out

def load_record(doc_id):
    if not re.fullmatch(r"\d{8}T\d{6}_[0-9a-f]{8}", str(doc_id)): return None
    f = RESULTS / doc_id / "record.json"
    return json.loads(f.read_text()) if f.exists() else None
