"""College-demo smoke tests: the REAL pipeline (OCR, Task A, Task B) on demo/sample_bills, failure cases, persistence across processes, and the Streamlit app starting/restarting. Scratch output dir only.
Takes ~1 minute (model load). The browser-level upload flow is documented in docs/DEMO.md and was checked manually; it is not automated here."""
import os, sys, json, time, signal, socket, tempfile, subprocess, unittest, urllib.request
from pathlib import Path
import _path
ROOT = Path(__file__).resolve().parents[1]; SAMPLES = ROOT / "demo" / "sample_bills"; PY = str(ROOT / ".venv" / "bin" / "python")
GOOD = ["synthetic_electricity.png", "synthetic_fuel.png", "synthetic_hotel.png", "restaurant_receipt_owner_screenshot.png", "cafe_receipt_cc0.pdf"]
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
class DemoSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="demo_smoke_")); os.environ["HRBC_OUT"] = str(cls.out)
        import service as SV
        o = cls.out; cls._old = (SV.OUT, SV.RESULTS, SV.CLASSIFIED, SV.UNCLASSIFIED); SV.OUT, SV.RESULTS, SV.CLASSIFIED, SV.UNCLASSIFIED = o, o / "results", o / "classified", o / "unclassified"
        cls.SV = SV; cls.svc = SV.BillService().load(); cls.recs = {}
    @classmethod
    def tearDownClass(cls):
        SV = cls.SV; SV.OUT, SV.RESULTS, SV.CLASSIFIED, SV.UNCLASSIFIED = cls._old
    def test_01_models_loaded_without_errors(self): self.assertEqual(self.svc.errors, {})
    def test_02_samples_are_processed_classified_extracted_and_saved(self):
        for name in GOOD:
            p = SAMPLES / name; r = self.svc.process(p.read_bytes(), name, None); self.recs[name] = r
            self.assertEqual(r["status"], "done", name); self.assertIn(r["task_a"]["label"], self.svc.labels); self.assertGreater(r["ocr_chars"], 20, name)
            self.assertIsInstance(r["task_b"]["fields"], dict); f = self.SV.OUT / r["saved_path"]; self.assertTrue(f.exists()); self.assertEqual(f.read_bytes(), p.read_bytes()); self.assertEqual(f.parent.name, r["task_a"]["label"])
            self.assertTrue((self.SV.OUT / r["result_dir"] / "record.json").exists())
            self.assertIn("profile", r); self.assertTrue(r["profile"]["document_type"])               # document type / benefit shown separately from the expense category
    def test_03_expected_categories_for_the_demo_inputs(self):                                           # observed behaviour of the existing models on these 5 files, recorded as a regression guard (NOT an accuracy claim)
        self.assertEqual({k: v["task_a"]["label"] for k, v in self.recs.items()}.get("synthetic_electricity.png"), "electricity")
        self.assertEqual(self.recs["synthetic_fuel.png"]["task_a"]["label"], "fuel"); self.assertEqual(self.recs["synthetic_hotel.png"]["task_a"]["label"], "hotel"); self.assertEqual(self.recs["restaurant_receipt_owner_screenshot.png"]["task_a"]["label"], "food")
    def test_04_blank_page_is_reported_not_guessed(self):
        p = SAMPLES / "blank_page_unreadable.png"; r = self.svc.process(p.read_bytes(), p.name, None); self.assertEqual(r["status"], "ocr_empty"); self.assertIsNone(r["category"]); self.assertEqual((self.SV.OUT / r["saved_path"]).parent.name, "unclassified")
    def test_05_corrupt_and_unsupported_files_are_rejected_with_readable_messages(self):
        SV = self.SV
        for name in ("corrupt_image.png", "unsupported_notes.txt"):
            p = SAMPLES / name
            with self.assertRaises(SV.UserError) as cm:
                kind, ext = SV.validate_upload(p.read_bytes(), name); SV.preview_first_page(p.read_bytes(), kind)       # the exact calls the upload page makes before showing a preview
            self.assertNotIn("Traceback", str(cm.exception)); self.assertGreater(len(str(cm.exception)), 15)
        with self.assertRaises(SV.UserError): self.svc.process((SAMPLES / "corrupt_image.png").read_bytes(), "corrupt_image.png", None)
    def test_06_records_listed_and_no_record_for_rejected_uploads(self):
        ids = {r["doc_id"] for r in self.SV.list_records()}; self.assertEqual(ids, {v["doc_id"] for v in self.recs.values()} | {r["doc_id"] for r in self.SV.list_records() if r["status"] == "ocr_empty"})
        for r in self.SV.list_records(): self.assertTrue((self.SV.OUT / r["saved_path"]).exists())
    def test_07_persistence_and_reprocessing_after_a_process_restart(self):
        try:
            free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
            if free < 2500: self.skipTest(f"only {free} MiB GPU memory free: another process (e.g. the running demo app) holds the GPU; a second model copy would hit CUDA out-of-memory")
        except (OSError, ValueError, IndexError): pass
        code = ("import sys,json; sys.path.insert(0,'app'); import service as S; svc=S.BillService().load(); recs=S.list_records(); "
                "p=sys.argv[1]; r=svc.process(open(p,'rb').read(),'again.png',None); print(json.dumps(dict(before=len(recs), after=len(S.list_records()), status=r['status'], err=str(r.get('ocr_error'))[:200], loaded=bool(S.load_record(recs[0]['doc_id'])))))")
        o = subprocess.run([PY, "-c", code, str(SAMPLES / "synthetic_fuel.png")], cwd=ROOT, env=dict(os.environ), capture_output=True, text=True, timeout=300); self.assertEqual(o.returncode, 0, o.stderr[-500:])
        d = json.loads(o.stdout.strip().splitlines()[-1]); self.assertEqual(d["after"], d["before"] + 1); self.assertEqual(d["status"], "done"); self.assertTrue(d["loaded"])
    def test_08_streamlit_app_starts_serves_and_restarts(self):
        for attempt in (1, 2):                                                                             # second iteration = restart of the local service
            port = free_port(); proc = subprocess.Popen([str(ROOT / ".venv/bin/streamlit"), "run", "app/streamlit_app.py", "--server.address", "127.0.0.1", "--server.port", str(port), "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                                                         cwd=ROOT, env=dict(os.environ), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                ok = False
                for _ in range(40):
                    time.sleep(1)
                    try:
                        if urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2).read() == b"ok": ok = True; break
                    except Exception: pass
                self.assertTrue(ok, f"app did not become healthy (attempt {attempt})"); self.assertEqual(urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).status, 200)
            finally:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
if __name__ == "__main__": unittest.main()
