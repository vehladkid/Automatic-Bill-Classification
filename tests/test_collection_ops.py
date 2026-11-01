"""collection_ops: licence/permission gating, privacy checks, duplicate detection, split isolation, dry-run safety and progress/queue reporting (all on temporary files; nothing is written to the real collection manifest)."""
import sys, json, random, tempfile, unittest, hashlib
from pathlib import Path
import _path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts" / "dataset_discovery"))
import collection_ops as O
import collection_manifest as CM
def img(path, seed):
    rnd = random.Random(seed); im = Image.new("L", (96, 96), 255); d = ImageDraw.Draw(im)
    for _ in range(14): x, y = rnd.randint(0, 80), rnd.randint(0, 80); d.rectangle([x, y, x + rnd.randint(4, 30), y + rnd.randint(4, 30)], fill=rnd.randint(0, 200))
    Path(path).parent.mkdir(parents=True, exist_ok=True); im.save(path)
def item(file, **kw):
    b = dict(file=file, doc_type="rent_receipt", expense_category="rent", tax_evidence=["hra_evidence"], source_id="COLL-ORG-01", permission_basis="individual_written_consent", permission_ref="CR-0042", issuer_group_id="IG-9c41ab07e3", redactor_id="R1", verifier_id="V1", language="en", country="IN", ambiguity_flags=[], reviewer_notes="")
    b.update(kw); return b
def rec(i, **kw):
    r = dict(doc_id=f"COL-{i:08x}-0001", doc_type="rent_receipt", expense_category="rent", tax_evidence=["hra_evidence"], real=True, source_id="COLL-ORG-01", permission_basis="individual_written_consent", permission_ref="CR-1", issuer_group_id=f"IG-{i:010x}", split="unassigned", annotation_status="single_annotated",
             sensitive_review_status="redacted_verified", redacted_copy=f"x/{i}.png", ambiguity_flags=[], reviewer_notes="", eligibility_status="pending_verification", country="IN", language="en", created="2026-10-19"); r.update(kw); return r
class Base(unittest.TestCase):
    def setUp(self): self.t = Path(tempfile.mkdtemp()); self.st = self.t / "staging"; self.man = self.t / "m.jsonl"; self.man.write_text("")
    def run_import(self, items, apply=False, ocr=lambda p: "rent receipt", corpus_sha=None, corpus_ph=None):
        return O.import_batch(items, self.st, self.man, apply=apply, ocr=ocr, corpus_sha=corpus_sha if corpus_sha is not None else set(), corpus_ph=corpus_ph if corpus_ph is not None else [])
class Gating(Base):
    def test_valid_item_is_accepted_in_dry_run_and_nothing_is_written(self):
        img(self.st / "a.png", 1); res, new = self.run_import([item("a.png")]); self.assertEqual(res["accepted"], 1); self.assertTrue(res["mode"].startswith("DRY-RUN")); self.assertEqual(self.man.read_text(), "")
    def test_apply_appends_and_keeps_backup_and_manifest_validates(self):
        img(self.st / "a.png", 1); self.man.write_text(""); res, new = self.run_import([item("a.png")], apply=True); rows = O.load(self.man); self.assertEqual(len(rows), 1); self.assertEqual(CM.validate(rows), {}); self.assertEqual(rows[0]["split"], "unassigned"); self.assertTrue(rows[0]["redacted_sha256"])
    def test_excluded_and_unapproved_public_sources_are_blocked(self):
        img(self.st / "a.png", 1)
        for sid, word in (("in_financial", "excluded"), ("sroie", "excluded"), ("some_new_dataset", "not in the approved")):
            res, _ = self.run_import([item("a.png", permission_basis="public_licence_verified", source_id=sid)]); self.assertEqual(res["accepted"], 0); self.assertIn(word, " ".join(res["rejected_detail"][0]["reasons"]), sid)
    def test_approved_public_source_passes_the_licence_gate(self):
        img(self.st / "a.png", 1); res, _ = self.run_import([item("a.png", permission_basis="public_licence_verified", source_id="funsd", permission_ref="LIC-FUNSD")]); self.assertEqual(res["accepted"], 1)
    def test_missing_permission_ref_blocked(self):
        img(self.st / "a.png", 1); res, _ = self.run_import([item("a.png", permission_ref="")]); self.assertEqual(res["accepted"], 0); self.assertIn("permission_ref", " ".join(res["rejected_detail"][0]["reasons"]))
    def test_same_person_cannot_redact_and_verify(self):
        img(self.st / "a.png", 1); res, _ = self.run_import([item("a.png", verifier_id="R1")]); self.assertEqual(res["accepted"], 0); self.assertIn("different people", " ".join(res["rejected_detail"][0]["reasons"]))
    def test_unverified_redaction_is_imported_only_as_redaction_required_without_copy(self):
        img(self.st / "a.png", 1); res, new = self.run_import([item("a.png", verifier_id=None)]); self.assertEqual(res["accepted"], 1); self.assertEqual(new[0]["sensitive_review_status"], "redaction_required"); self.assertIsNone(new[0]["redacted_copy"])
    def test_unsafe_paths_and_missing_or_corrupt_files_blocked(self):
        (self.st).mkdir(parents=True, exist_ok=True); (self.st / "bad.png").write_bytes(b"not an image")
        for f in ("/etc/passwd", "../x.png", "missing.png", "bad.png"): res, _ = self.run_import([item(f)]); self.assertEqual(res["accepted"], 0, f)
class Privacy(Base):
    def test_personal_data_in_notes_is_rejected_and_never_echoed(self):
        img(self.st / "a.png", 1); res, _ = self.run_import([item("a.png", reviewer_notes="holder PAN ABCDE1234F phone 9876543210")]); self.assertEqual(res["accepted"], 0); out = json.dumps(res); self.assertNotIn("ABCDE1234F", out); self.assertNotIn("9876543210", out); self.assertIn("personal-data pattern", out)
    def test_ocr_of_redacted_copy_with_residual_pii_is_rejected(self):
        img(self.st / "a.png", 1); res, _ = self.run_import([item("a.png")], ocr=lambda p: "Name\nPAN: ABCDE1234F\nrent"); self.assertEqual(res["accepted"], 0); out = json.dumps(res); self.assertIn("PAN", out); self.assertNotIn("ABCDE1234F", out)
    def test_status_output_contains_no_document_text(self):
        rows = [rec(1, reviewer_notes="handwritten stamp")]; self.man.write_text("".join(json.dumps(r) + "\n" for r in rows)); rep, txt, err = O.status(self.man); self.assertNotIn("handwritten", json.dumps(rep) + txt)
class Duplicates(Base):
    def test_byte_duplicate_in_batch_and_in_collection_and_in_corpus(self):
        img(self.st / "a.png", 1); (self.st / "b.png").write_bytes((self.st / "a.png").read_bytes())
        res, _ = self.run_import([item("a.png"), item("b.png")]); self.assertEqual((res["accepted"], res["rejected"]), (1, 1))
        res, new = self.run_import([item("a.png")], apply=True); self.assertEqual(res["accepted"], 1); res, _ = self.run_import([item("a.png", issuer_group_id="IG-0000000001")]); self.assertIn("collection", " ".join(res["rejected_detail"][0]["reasons"]))
        s = hashlib.sha256((self.st / "a.png").read_bytes()).hexdigest(); self.man.write_text(""); res, _ = self.run_import([item("a.png")], corpus_sha={s}); self.assertIn("real corpus", " ".join(res["rejected_detail"][0]["reasons"]))
    def test_near_duplicate_of_corpus_blocked_and_distinct_image_passes(self):
        img(self.st / "a.png", 1); img(self.st / "c.png", 99); ph = O.phash64(self.st / "a.png")
        res, _ = self.run_import([item("a.png")], corpus_ph=[ph]); self.assertEqual(res["accepted"], 0); self.assertIn("near-duplicate", " ".join(res["rejected_detail"][0]["reasons"]))
        far = "%016x" % (int(O.phash64(self.st / "c.png"), 16) ^ 0xFFFFFFFFFFFFFFFF); res, _ = self.run_import([item("c.png")], corpus_ph=[far]); self.assertEqual(res["accepted"], 1)
    def test_validator_rejects_two_records_with_same_sha(self):
        a, b = rec(1, redacted_sha256="a" * 64), rec(2, redacted_sha256="a" * 64); self.assertIn("duplicate_redacted_sha256", CM.validate([a, b]))
class SplitIsolation(Base):
    def test_whole_issuer_group_gets_one_split_and_is_deterministic(self):
        rows = [rec(1, issuer_group_id="IG-aaaaaaaaaa", redacted_copy="x/1.png"), rec(2, issuer_group_id="IG-aaaaaaaaaa", redacted_copy="x/2.png"), rec(3, issuer_group_id="IG-bbbbbbbbbb", redacted_copy="x/3.png")]
        for r in rows: r["annotation_status"] = "double_annotated"
        ch, w = O.assign_splits(rows, apply=True); sp = {r["doc_id"]: r["split"] for r in rows}; self.assertEqual(sp[rows[0]["doc_id"]], sp[rows[1]["doc_id"]]); self.assertEqual(CM.validate(rows), {})
        rows2 = [dict(r, split="unassigned") for r in rows]; ch2, _ = O.assign_splits(rows2, apply=True); self.assertEqual([r["split"] for r in rows], [r["split"] for r in rows2])
    def test_groups_wait_for_redaction_annotation_and_double_annotation(self):
        rows = [rec(1, sensitive_review_status="redaction_required", redacted_copy=None), rec(2, annotation_status="pending")]; ch, w = O.assign_splits(rows); self.assertEqual(ch, []); self.assertEqual(w["waiting_for_verified_redaction"], 1); self.assertEqual(w["waiting_for_annotation"], 1)
        eval_groups = [rec(i, issuer_group_id=f"IG-{i:010x}") for i in range(10, 70)]; ch, w = O.assign_splits(eval_groups)
        self.assertTrue(all(t == "train" for _, t in ch)); self.assertGreater(w["waiting_for_double_annotation_before_eval_split"], 0)
    def test_specimen_groups_go_to_train_only(self):
        rows = [rec(i, issuer_group_id=f"IG-{i:010x}", ambiguity_flags=["specimen_or_template"], annotation_status="double_annotated") for i in range(100, 160)]; ch, _ = O.assign_splits(rows); self.assertTrue(all(t == "train" for _, t in ch))
    def test_new_document_of_an_assigned_group_inherits_its_split(self):
        rows = [rec(1, issuer_group_id="IG-cccccccccc", split="train", redacted_copy="x/1.png"), rec(2, issuer_group_id="IG-cccccccccc", redacted_copy="x/2.png")]; ch, _ = O.assign_splits(rows, apply=True); self.assertEqual(rows[1]["split"], "train")
    def test_group_spanning_splits_is_reported_not_repaired(self):
        rows = [rec(1, issuer_group_id="IG-dddddddddd", split="train"), rec(2, issuer_group_id="IG-dddddddddd", split="train", redacted_copy="x/2.png")]; rows[1]["split"] = "val"; rows[1]["annotation_status"] = "double_annotated"
        self.assertIn("issuer_group_spans_splits", CM.validate(rows)); ch, w = O.assign_splits(rows); self.assertEqual(w["group_already_spans_splits"], 1)
class Reporting(Base):
    def test_progress_counts_issuers_and_ready_flag(self):
        rows = [rec(i, issuer_group_id=f"IG-{i % 7:010x}", split="test" if i % 2 else "val", annotation_status="double_annotated") for i in range(1, 80)]
        for r in rows: r["issuer_group_id"] = f"IG-{(0 if r['split'] == 'test' else 100) + int(r['doc_id'][4:12], 16) % 7:010x}"
        p = {x["doc_type"]: x for x in O.progress(rows)}["rent_receipt"]; self.assertEqual(p["usable_in_splits"], 79); self.assertEqual(p["issuers_per_split"]["val"], 7); self.assertTrue(p["ready_for_per_class_metric"])
        p2 = {x["doc_type"]: x for x in O.progress(rows[:10])}["rent_receipt"]; self.assertFalse(p2["ready_for_per_class_metric"])
    def test_queues_and_blocked_reasons_are_explained(self):
        rows = [rec(1, sensitive_review_status="redaction_required", redacted_copy=None), rec(2, ambiguity_flags=["type_ambiguous"]), rec(3, annotation_status="pending")]; q = O.queues(rows)
        self.assertEqual(q["redaction_verification"], [rows[0]["doc_id"]]); self.assertEqual(q["label_review"], [rows[1]["doc_id"]]); self.assertEqual(q["annotation_needed"], [rows[2]["doc_id"]]); self.assertTrue(any("redaction" in m for _, m in O.blockers(rows[0])))
        self.assertIn("leakage", O.explain("issuer_group_spans_splits")); self.assertIn("never", O.explain("eligibility_status_invalid"))
    def test_status_exit_semantics_on_invalid_manifest(self):
        self.man.write_text(json.dumps(rec(1, doc_type="passport")) + "\n"); rep, txt, err = O.status(self.man); self.assertFalse(rep["passed"]); self.assertIn("ERROR bad_doc_type", txt)
    def test_real_collection_manifest_is_empty_and_valid(self): rep, txt, err = O.status(O.DEFAULT); self.assertEqual(rep["n_records"], 0); self.assertFalse(err)
import approval_gate as AG
class AcquisitionGate(unittest.TestCase):
    def test_excluded_sources_refused_with_reason(self):
        for sid in ("in_financial", "sroie", "hf_medical", "user_real", "receiptbench"): d = AG.decision(sid); self.assertFalse(d["approved"], sid); self.assertIn("excluded", d["reason"])
        with self.assertRaises(SystemExit): AG.require_approved("in_financial")
    def test_unknown_and_unlisted_sources_refused(self):
        self.assertIn("licence register", AG.decision("totally_new")["reason"]); self.assertFalse(AG.decision("docile")["approved"]); self.assertFalse(AG.decision("midd")["approved"])
    def test_approved_sources_return_evidence_for_provenance(self):
        for sid in ("fatura", "agami_indian_bank_statements", "funsd"): d = AG.require_approved(sid); self.assertTrue(d["approved"]); self.assertTrue(d["evidence_url"] or d["evidence_level"]); self.assertIn("approval_config", d)
        self.assertIn("non-commercial", AG.decision("funsd")["restriction"])
    def test_config_register_disagreement_blocks(self):
        tmp = Path(tempfile.mkdtemp()); reg = json.loads((ROOT / "datasets/metadata/license_register_v2.json").read_text())
        for s in reg["sources"]:
            if s["id"] == "fatura": s["status"] = "unverified"
        (tmp / "r.json").write_text(json.dumps(reg)); d = AG.decision("fatura", reg_path=tmp / "r.json"); self.assertFalse(d["approved"]); self.assertIn("disagree", d["reason"])
    def test_fetch_verified_refuses_dataset_download_without_approval(self):
        import subprocess
        for args in (["hf", "x/y", "--dest", "/tmp/nope", "--dry-run"], ["hf", "T0rl/ReceiptBench", "--dest", "/tmp/nope", "--source-id", "receiptbench", "--dry-run"]):
            p = subprocess.run([sys.executable, str(ROOT / "scripts/dataset_discovery/fetch_verified.py")] + args, capture_output=True, text=True); self.assertNotEqual(p.returncode, 0); self.assertIn("REFUSED", p.stdout + p.stderr)

if __name__ == "__main__": unittest.main()
