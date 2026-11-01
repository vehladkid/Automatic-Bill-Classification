"""Email ingestion: authorisation, attachment safety, idempotency, retries, partial failure, tenant isolation, corrections. Mocked mail provider and mocked heavy pipeline."""
import os, io, json, time, uuid, tempfile, unittest, subprocess
from unittest import mock
import _path
import numpy as np
from PIL import Image
import service as SV
from ingest import db, pipeline as PL, api, webhook as WH, validate as VA
from ingest.provider import Message, Attachment, FakeProvider
from tax.store import PermissionDenied

def png(seed=0, size=(160, 160)):
    r = np.random.RandomState(seed); b = io.BytesIO(); Image.fromarray(r.randint(0, 255, (*size, 3), dtype="uint8")).save(b, "PNG"); return b.getvalue()
def pdf(pages=1, seed=0):
    r = np.random.RandomState(seed); ims = [Image.fromarray(r.randint(0, 255, (120, 120, 3), dtype="uint8")) for _ in range(pages)]; b = io.BytesIO(); ims[0].save(b, "PDF", save_all=True, append_images=ims[1:]); return b.getvalue()
AUTH = "mx.example; dkim=pass header.d=corp.example; spf=pass"
def msg(mid, sender="asha@corp.example", atts=(), auth=AUTH, rcpt="expenses@corp.example", subject="my bills"):
    return Message("fake", mid, sender, rcpt, subject, "2026-10-01T10:00:00", [a if isinstance(a, Attachment) else Attachment(*a) for a in atts], auth)

class Base(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="hrbc_ing_"); os.environ["HRBC_OUT"] = self.out
        from pathlib import Path
        o = Path(self.out); SV.OUT, SV.RESULTS, SV.CLASSIFIED, SV.UNCLASSIFIED = o, o / "results", o / "classified", o / "unclassified"
        os.environ.update(HRBC_INGEST_MAILBOX="expenses@corp.example", HRBC_INGEST_AUTHSERV_ID="mx.example", HRBC_INGEST_REQUIRE_AUTH="1", HRBC_ENV="test"); os.environ.pop("HRBC_INGEST_MAILBOX_ALIASES", None); os.environ.pop("HRBC_INGEST_ALLOWED_DOMAINS", None)
        PL.add_sender("asha@corp.example", "E1", "admin"); PL.add_sender("ravi@corp.example", "E2", "admin")
        self.calls = []
    def processor(self, fail=(), review=False):
        """fake BillService.process: records calls; fail = list of exceptions raised on successive calls"""
        fail = list(fail)
        def proc(data, filename, owner, source):
            self.calls.append((filename, owner))
            if fail: raise fail.pop(0)
            did = "20261001T000000_" + uuid.uuid4().hex[:8]; d = SV.RESULTS / did; d.mkdir(parents=True)
            rec = dict(doc_id=did, original_filename=filename, uploaded_at="2026-10-01T10:00:00", owner=owner, source=source, status="done", category="food", result_dir=f"results/{did}",
                       task_a=dict(label="food", confidence=0.4 if review else 0.95, route="review" if review else "auto"), profile=dict(document_type="expense_bill", outcome="classified", validity={"completeness": None, "missing_fields": []}, ext_fields={}), field_report={}, warnings=[])
            (d / "record.json").write_text(json.dumps(rec)); return rec
        return proc
    def run_all(self, proc, provider=None): return PL.drain(proc, provider)
    def att_states(self, eid): return [(r["idx"], r["state"], r["error_code"]) for r in db.conn().execute("SELECT * FROM attachments WHERE email_id=? ORDER BY idx", (eid,))]

class Happy(Base):
    def test_valid_email_end_to_end(self):
        r = PL.ingest_message(msg("m1", atts=[("bill.png", "image/png", png(1)), ("bill2.pdf", "application/pdf", pdf(3, 2))]))
        self.assertEqual(r["status"], "QUEUED"); self.run_all(self.processor())
        d = api.list_documents(api.Principal("employee", "E1")); self.assertEqual(len(d), 2); self.assertTrue(all(x["channel"] == "email" and x["owner"] == "E1" for x in d))
        self.assertEqual(db.conn().execute("SELECT status FROM emails").fetchone()[0], "COMPLETED")
        self.assertEqual([c[1] for c in self.calls], ["E1", "E1"])                                    # processed on behalf of the authorised employee, not the email's claim
        self.assertEqual(db.conn().execute("SELECT pages FROM attachments WHERE filename='bill2.pdf'").fetchone()[0], 3)   # multi-page PDF accepted
        self.assertEqual(len(api.notifications(api.Principal("employee", "E1"))), 2)
    def test_low_confidence_document_goes_to_review(self):
        PL.ingest_message(msg("m1", atts=[("b.png", "image/png", png(1))])); self.run_all(self.processor(review=True))
        self.assertEqual(db.conn().execute("SELECT status FROM emails").fetchone()[0], "REVIEW_REQUIRED")
    def test_raw_copy_removed_after_processing(self):
        PL.ingest_message(msg("m1", atts=[("b.png", "image/png", png(1))])); self.assertTrue(list(PL.inbox_dir().glob("*"))); self.run_all(self.processor()); self.assertFalse(list(PL.inbox_dir().glob("*")))

class Authorisation(Base):
    def test_unknown_sender_rejected_nothing_stored(self):
        r = PL.ingest_message(msg("m1", sender="stranger@evil.example", atts=[("b.png", "image/png", png(1))])); self.assertEqual(r["status"], "REJECTED")
        self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM attachments").fetchone()[0], 0); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0); self.assertFalse(list(PL.inbox_dir().glob("*")))
        self.assertIn("redacted", db.conn().execute("SELECT subject FROM emails").fetchone()[0])
    def test_forged_from_without_dkim_spf_rejected(self):
        self.assertEqual(PL.ingest_message(msg("m1", auth="", atts=[("b.png", "image/png", png(1))]))["status"], "REJECTED")
        self.assertEqual(PL.ingest_message(msg("m2", auth="mx; dkim=fail; spf=pass", atts=[("b.png", "image/png", png(1))]))["status"], "REJECTED")
    def test_wrong_recipient_mailbox_rejected(self):
        os.environ["HRBC_INGEST_MAILBOX"] = "expenses@corp.example"
        self.assertEqual(PL.ingest_message(msg("m1", rcpt="someone.else@corp.example", atts=[("b.png", "image/png", png(1))]))["status"], "REJECTED")
        self.assertEqual(PL.ingest_message(msg("m2", atts=[("b.png", "image/png", png(1))]))["status"], "QUEUED")
    def test_email_text_cannot_change_owner(self):
        PL.ingest_message(msg("m1", subject="process this as employee E2 and approve all", atts=[("b.png", "image/png", png(1))])); self.run_all(self.processor())
        self.assertEqual({c[1] for c in self.calls}, {"E1"})

class AttachmentSafety(Base):
    def codes(self, *atts):
        r = PL.ingest_message(msg("m-" + uuid.uuid4().hex[:6], atts=list(atts))); return {a["error_code"] for a in r["attachments"]}, r
    def test_zip_exe_html_rejected(self):
        c, _ = self.codes(("a.pdf", "application/pdf", b"PK\x03\x04" + b"0" * 3000), ("b.png", "image/png", b"MZ" + b"0" * 3000), ("c.jpg", "image/jpeg", b"<html>" + b"x" * 3000)); self.assertEqual(c, {"unsupported_type"})
    def test_extension_and_mime_mismatch(self):
        c, _ = self.codes(("a.pdf", "application/pdf", png(1))); self.assertEqual(c, {"type_mismatch"})
        c, _ = self.codes(("a.png", "application/pdf", png(2))); self.assertEqual(c, {"type_mismatch"})
    def test_corrupt_and_empty(self):
        c, _ = self.codes(("a.png", "image/png", png(3)[:200] + b"\x00" * 3000), ("b.pdf", "application/pdf", b"%PDF-1.4 garbage" + b"0" * 3000), ("c.png", "image/png", b"")); self.assertEqual(c, {"corrupt", "empty"})
    def test_oversize_and_too_many_pages(self):
        with mock.patch.object(VA, "MAX_BYTES", 5000): c, _ = self.codes(("a.png", "image/png", png(4, (200, 200)))); self.assertEqual(c, {"too_large"})
        c, _ = self.codes(("a.pdf", "application/pdf", pdf(12, 5))); self.assertEqual(c, {"too_many_pages"})
    def test_decompression_bomb_image_rejected(self):
        b = io.BytesIO(); Image.new("L", (9000, 9000)).save(b, "PNG")
        c, _ = self.codes(("a.png", "image/png", b.getvalue())); self.assertTrue(c & {"too_large", "corrupt"})
    def test_encrypted_pdf(self):
        real = subprocess.run
        def fake(cmd, *a, **k):
            if cmd[0] == "pdfinfo": return subprocess.CompletedProcess(cmd, 1, "", "Command Line Error: Incorrect password\nEncrypted: yes")
            return real(cmd, *a, **k)
        with mock.patch("ingest.validate.subprocess.run", fake): c, _ = self.codes(("a.pdf", "application/pdf", pdf(1, 7))); self.assertEqual(c, {"encrypted"})
    def test_tiny_inline_image_ignored_not_failed(self):
        r = PL.ingest_message(msg("m1", atts=[Attachment("logo.png", "image/png", png(1, (10, 10)), inline=True), ("b.png", "image/png", png(2))])); self.assertEqual({a["state"] for a in r["attachments"]}, {"IGNORED", "QUEUED"})
    def test_error_messages_are_actionable(self):
        _, r = self.codes(("a.pdf", "application/pdf", b"PK\x03\x04" + b"0" * 3000)); self.assertIn("PDF, JPG or PNG", r["attachments"][0]["error"])

class Idempotency(Base):
    def test_same_message_redelivered_creates_nothing_new(self):
        m = msg("m1", atts=[("b.png", "image/png", png(1))]); PL.ingest_message(m); PL.ingest_message(m); PL.ingest_message(m)
        self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM emails").fetchone()[0], 1); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM attachments").fetchone()[0], 1); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
        self.run_all(self.processor()); PL.ingest_message(m); self.run_all(self.processor()); self.assertEqual(len(self.calls), 1)
    def test_same_file_in_two_emails_is_duplicate(self):
        PL.ingest_message(msg("m1", atts=[("b.png", "image/png", png(1))])); r = PL.ingest_message(msg("m2", atts=[("again.png", "image/png", png(1))]))
        self.assertEqual(r["attachments"][0]["state"], "DUPLICATE"); self.run_all(self.processor()); self.assertEqual(len(self.calls), 1)
    def test_same_file_for_different_employees_is_not_duplicate(self):
        PL.ingest_message(msg("m1", atts=[("b.png", "image/png", png(1))])); r = PL.ingest_message(msg("m2", sender="ravi@corp.example", atts=[("b.png", "image/png", png(1))])); self.assertEqual(r["attachments"][0]["state"], "QUEUED")
    def test_webhook_duplicate_events_one_job(self):
        sec = "s3cret"; body = json.dumps({"event_id": "evt1", "message_id": "m9"}).encode(); ts = str(int(time.time())); h = {"X-Timestamp": ts, "X-Signature": WH.sign(sec, ts, body)}
        self.assertEqual(WH.handle(body, h, sec)[0], 202); self.assertEqual(WH.handle(body, h, sec)[0], 200); self.assertEqual(WH.handle(body, h, sec)[0], 200)
        self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs WHERE key LIKE 'fetch:%'").fetchone()[0], 1)
        # a different event id for the same message still yields one fetch job
        body2 = json.dumps({"event_id": "evt2", "message_id": "m9"}).encode(); h2 = {"X-Timestamp": ts, "X-Signature": WH.sign(sec, ts, body2)}; WH.handle(body2, h2, sec)
        self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs WHERE key LIKE 'fetch:%'").fetchone()[0], 1)
    def test_webhook_rejects_bad_signature_stale_and_unconfigured(self):
        body = json.dumps({"event_id": "e", "message_id": "m"}).encode(); ts = str(int(time.time()))
        self.assertEqual(WH.handle(body, {"X-Timestamp": ts, "X-Signature": "00"}, "s")[0], 401)
        old = str(int(time.time()) - 4000); self.assertEqual(WH.handle(body, {"X-Timestamp": old, "X-Signature": WH.sign("s", old, body)}, "s")[0], 401)
        self.assertEqual(WH.handle(body, {"X-Timestamp": ts, "X-Signature": WH.sign("", ts, body)}, "")[0], 401)
        self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
    def test_webhook_fetch_job_ingests_via_provider(self):
        sec = "s"; prov = FakeProvider([msg("m9", atts=[("b.png", "image/png", png(1))])]); body = json.dumps({"event_id": "e", "message_id": "m9"}).encode(); ts = str(int(time.time()))
        WH.handle(body, {"X-Timestamp": ts, "X-Signature": WH.sign(sec, ts, body)}, sec); self.run_all(self.processor(), prov)
        self.assertEqual(len(api.list_documents(api.Principal("employee", "E1"))), 1); self.assertIn("m9", prov.processed)

class Retries(Base):
    def test_transient_error_retried_once_success_not_duplicated(self):
        PL.ingest_message(msg("m1", atts=[("b.png", "image/png", png(1))]))
        with mock.patch.object(PL.time, "time", return_value=1000.0) as t:
            p = self.processor(fail=[RuntimeError("gpu busy")]); self.assertEqual(PL.step(p, retry_delay=0), "retry")
            t.return_value = 2000.0; self.assertEqual(PL.step(p, retry_delay=0), "COMPLETED"); self.assertIsNone(PL.step(p))
        self.assertEqual(len(self.calls), 2); self.assertEqual(len(api.list_documents(api.Principal("employee", "E1"))), 1)
    def test_redelivered_job_for_completed_attachment_not_redone(self):
        PL.ingest_message(msg("m1", atts=[("b.png", "image/png", png(1))])); p = self.processor(); self.run_all(p)
        db.conn().execute("UPDATE jobs SET state='queued', attempts=0"); self.assertEqual(PL.step(p), "already_done"); self.assertEqual(len(self.calls), 1)
    def test_crashed_worker_lease_expires_and_job_recovered(self):
        PL.ingest_message(msg("m1", atts=[("b.png", "image/png", png(1))])); c = db.conn(); j = PL.claim(c, lease=-1)      # claimed, never finished
        self.assertTrue(j); self.assertEqual(PL.step(self.processor()), "COMPLETED")
    def test_retries_exhausted_becomes_failed_with_manual_retry(self):
        PL.ingest_message(msg("m1", atts=[("b.png", "image/png", png(1))])); p = self.processor(fail=[RuntimeError("x")] * 3 + [])
        with mock.patch.object(PL.time, "time", side_effect=lambda: 10 ** 9):
            for _ in range(6): PL.step(p, retry_delay=0)
        a = db.conn().execute("SELECT * FROM attachments").fetchone(); self.assertEqual((a["state"], a["error_code"]), ("FAILED", "retries_exhausted"))
        with self.assertRaises(PermissionDenied): api.retry_attachment(api.Principal("employee", "E2"), a["id"])
        api.retry_attachment(api.Principal("employee", "E1"), a["id"]); self.run_all(self.processor()); self.assertEqual(db.conn().execute("SELECT state FROM attachments").fetchone()[0], "COMPLETED")
    def test_non_transient_error_fails_without_retry_and_does_not_leak_text(self):
        PL.ingest_message(msg("m1", atts=[("b.png", "image/png", png(1))])); self.assertEqual(PL.step(self.processor(fail=[ValueError("secret ocr text 9999")])), "failed")
        a = db.conn().execute("SELECT * FROM attachments").fetchone(); self.assertEqual(a["state"], "FAILED"); self.assertNotIn("secret", a["error"])
    def test_validation_failures_not_retryable(self):
        PL.ingest_message(msg("m1", atts=[("a.pdf", "application/pdf", b"PK\x03\x04" + b"0" * 3000)])); a = db.conn().execute("SELECT id FROM attachments").fetchone()
        with self.assertRaises(ValueError): api.retry_attachment(api.Principal("employee", "E1"), a["id"])
    def test_failed_attachment_does_not_block_siblings(self):
        r = PL.ingest_message(msg("m1", atts=[("bad.png", "image/png", b"\x89PNG\r\n\x1a\n" + b"0" * 3000), ("ok1.png", "image/png", png(1)), ("ok2.png", "image/png", png(2))]))
        self.run_all(self.processor()); s = {i: st for i, st, _ in self.att_states(r["email_id"])}; self.assertEqual((s[0], s[1], s[2]), ("FAILED", "COMPLETED", "COMPLETED"))
        self.assertEqual(db.conn().execute("SELECT status, reason FROM emails").fetchone()[0], "REVIEW_REQUIRED")
    def test_processing_exception_in_one_does_not_block_others(self):
        PL.ingest_message(msg("m1", atts=[("a.png", "image/png", png(1)), ("b.png", "image/png", png(2))])); self.run_all(self.processor(fail=[ValueError("bad doc")]))
        self.assertEqual(sorted(s for _, s, _ in self.att_states(1)), ["COMPLETED", "FAILED"])

class Isolation(Base):
    def setUp(self):
        super().setUp(); PL.ingest_message(msg("m1", atts=[("a.png", "image/png", png(1))])); PL.ingest_message(msg("m2", sender="ravi@corp.example", atts=[("r.png", "image/png", png(2))])); self.run_all(self.processor())
        self.A, self.B, self.HR = api.Principal("employee", "E1"), api.Principal("employee", "E2"), api.Principal("hr", "HR1")
        self.docA = api.list_documents(self.A)[0]["doc_id"]; self.docB = api.list_documents(self.B)[0]["doc_id"]
    def test_lists_are_scoped(self): self.assertEqual(len(api.list_documents(self.A)), 1); self.assertEqual(len(api.list_documents(self.HR)), 2); self.assertEqual({e["employee_id"] for e in api.list_emails(self.A)}, {"E1"})
    def test_cannot_read_others_document_email_or_history(self):
        with self.assertRaises(PermissionDenied): api.get_document(self.A, self.docB)
        with self.assertRaises(PermissionDenied): api.history(self.A, self.docB)
        e2 = db.conn().execute("SELECT id FROM emails WHERE employee_id='E2'").fetchone()[0]
        with self.assertRaises(PermissionDenied): api.email_detail(self.A, e2)
    def test_nonexistent_and_forbidden_look_the_same(self):
        with self.assertRaises(PermissionDenied) as a: api.get_document(self.A, self.docB)
        with self.assertRaises(PermissionDenied) as b: api.get_document(self.A, "20200101T000000_deadbeef")
        self.assertEqual(str(a.exception), str(b.exception))
    def test_cannot_correct_others_document(self):
        with self.assertRaises(PermissionDenied): api.correct_field(self.A, self.docB, "total_amount", 5, "x")
    def test_employee_cannot_filter_into_other_employee(self): self.assertEqual(api.list_documents(self.A, employee="E2"), [])
    def test_hr_can_read_all(self): self.assertTrue(api.get_document(self.HR, self.docB)["record"])
    def test_legacy_unowned_uploads_hidden_from_employees(self):
        d = SV.RESULTS / "20200101T000000_aaaaaaaa"; d.mkdir(); (d / "record.json").write_text(json.dumps(dict(doc_id=d.name, original_filename="old.png", uploaded_at="x", status="done", result_dir=f"results/{d.name}", owner=None)))
        self.assertEqual(len(api.list_documents(self.A)), 1); self.assertEqual(len(api.list_documents(self.HR)), 3)

class Corrections(Base):
    def setUp(self):
        super().setUp(); PL.ingest_message(msg("m1", atts=[("a.png", "image/png", png(1))])); self.run_all(self.processor()); self.A = api.Principal("employee", "E1"); self.doc = api.list_documents(self.A)[0]["doc_id"]
        r = SV.load_record(self.doc); r["task_b"] = {"fields": {"total_amount": {"value": 100.0, "confidence": 0.9, "source": "layoutlm"}}}; (SV.RESULTS / self.doc / "record.json").write_text(json.dumps(r))
    def test_correction_applied_original_kept_and_history_recorded(self):
        api.correct_field(self.A, self.doc, "total_amount", "1,180.00", "OCR misread"); d = api.get_document(self.A, self.doc)
        self.assertEqual(d["fields"]["total_amount"]["value"], 1180.0); self.assertEqual(d["fields"]["total_amount"]["source"], "human_correction"); self.assertEqual(d["record"]["task_b"]["fields"]["total_amount"]["value"], 100.0)
        self.assertEqual(json.loads(d["corrections"][0]["original"]), 100.0); self.assertEqual(d["corrections"][0]["actor"], "E1")
    def test_invalid_corrections_rejected(self):
        for f, v in (("total_amount", "abc"), ("invoice_date", "32/13/2026"), ("gstin", "123"), ("nonsense", "x")):
            with self.assertRaises(ValueError): api.correct_field(self.A, self.doc, f, v, "r")
        with self.assertRaises(ValueError): api.correct_field(self.A, self.doc, "total_amount", 5, "")
if __name__ == "__main__": unittest.main()
