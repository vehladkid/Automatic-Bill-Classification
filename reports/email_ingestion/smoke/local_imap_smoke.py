"""LOCAL SMOKE TEST (post-hardening): the hardened ImapProvider (verified TLS against a private test CA, read-only mailbox, cursor polling, header-first authorisation) -> pipeline -> worker -> REAL BillService (real OCR, Task A, Task B models) -> API.
The IMAP server is the local stub in tests/imap_stub.py: this proves our client/pipeline wire path, NOT Gmail/Outlook behaviour. Temporary HRBC_OUT, no outbound network, synthetic or public-licence fixtures.
Usage: cd app && ../.venv/bin/python ../reports/email_ingestion/smoke/local_imap_smoke.py"""
import os, sys, json, tempfile, time, subprocess
from pathlib import Path
from email.message import EmailMessage
ROOT = Path(__file__).resolve().parents[3]; TMP = Path(tempfile.mkdtemp(prefix="imap_smoke_"))
os.environ.update(HRBC_OUT=str(TMP / "out"), HRBC_ENV="test", HRBC_INGEST_MAILBOX="expenses@corp.example", HRBC_INGEST_AUTHSERV_ID="mx.corp", HRBC_INGEST_REQUIRE_AUTH="1")
for p in ("app", "scripts", "scripts/taskB", "tests"): sys.path.insert(0, str(ROOT / p))
import imap_stub as STUB
pki = STUB.make_pki(TMP / "pki")
pdf = next(Path(ROOT / "datasets/public/my_receipts").rglob("*.pdf")).read_bytes(); jpg = next((ROOT / "datasets/synthetic_doctype/v1/images/salary_slip").glob("*.jpg")).read_bytes()
AUTH = "mx.corp; dkim=pass header.d=corp.example; spf=pass smtp.mailfrom=corp.example"
def mail(mid, sender="asha@corp.example", atts=(), to="expenses@corp.example", auth=AUTH, body="Please find the bill.", html=None):
    m = EmailMessage(); m["Message-ID"] = mid; m["From"] = sender; m["To"] = to; m["Subject"] = "bill " + mid; m["Date"] = "Mon, 19 Oct 2026 10:00:00 +0000"; m["Authentication-Results"] = auth; m.set_content(body)
    if html: m.add_alternative(html, subtype="html")
    for fn, data, mt, st in atts: m.add_attachment(data, maintype=mt, subtype=st, filename=fn)
    return m
inner = mail("<inner@x>", sender="vendor@shop.example", atts=[("inner.pdf", pdf, "application", "pdf")]); fwd = mail("<m3@x>", body="Fwd: see attached"); fwd.add_attachment(inner)
MBOX = [mail("<m1@x>", atts=[("bill.pdf", pdf, "application", "pdf"), ("slip.jpg", jpg, "image", "jpeg")]), mail("<m2@x>", sender="stranger@evil.example", atts=[("bill.pdf", pdf, "application", "pdf")]), fwd,
        mail("<m4@x>", atts=[("a.zip", b"PK\x03\x04" + b"0" * 2000, "application", "zip")]), mail("<m5@x>", body="Invoice total Rs 1,250.00 inside body", html="<html><body><h1>Tax Invoice</h1><p>Total Rs 1,250.00 paid GSTIN 29ABCDE1234F1Z5 order id 9</p></body></html>"),
        mail("<m1@x>", atts=[("bill.pdf", pdf, "application", "pdf")]), mail("<m7@x>", atts=[("bill2.pdf", pdf, "application", "pdf")]),
        mail("<m8@x>", sender="asha@corp.example", auth="mx.corp; dkim=pass header.d=evil.example; spf=pass smtp.mailfrom=evil.example", atts=[("bill.pdf", pdf, "application", "pdf")])]    # forged From with an attacker-domain pass
srv = STUB.StubServer(pki["good"], MBOX); port = srv.start()
import service as SV
from ingest import pipeline as PL, db, api, tls, perms
from ingest.provider import ImapProvider
R = {}; PL.add_sender("asha@corp.example", "E1", "audit"); prov = ImapProvider("localhost", "user", "pw", "INBOX", port, ssl_context=tls.context(pki["ca"]))
t0 = time.time(); svc = SV.BillService().load(); R["models_loaded"] = {"ocr": svc.ocr_engine is not None, "task_a": svc.A is not None, "task_b": svc.B is not None, "errors": svc.errors}
st = PL.poll_once(prov, initial="all", now=1000); R["poll1_stats"] = st
c = db.conn(); R["emails_after_poll1"] = [(r["message_id"], r["status"], r["reason"]) for r in c.execute("SELECT * FROM emails ORDER BY id")]
R["attachments_after_ingest"] = [(r["filename"], r["state"], r["error_code"]) for r in c.execute("SELECT * FROM attachments ORDER BY id")]
R["jobs_processed"] = PL.drain(svc.process, prov)
R["attachments_after_worker"] = [(r["filename"], r["state"], r["error"]) for r in c.execute("SELECT * FROM attachments ORDER BY id")]; R["emails_after_worker"] = [(r["message_id"], r["status"]) for r in c.execute("SELECT * FROM emails ORDER BY id")]
R["poll2_stats(should ingest nothing)"] = PL.poll_once(prov, initial="all", now=1100); R["cursor"] = db.get_cursor("imap", prov.mailbox_key())
docs = api.list_documents(api.Principal("employee", "E1")); R["employee_E1_documents"] = [dict(status=d["status"], category=d["category"], document_type=d["document_type"], channel=d["channel"], has_email=bool(d["email"])) for d in docs]; R["employee_E2_documents"] = len(api.list_documents(api.Principal("employee", "E2")))
if docs:
    D = api.get_document(api.Principal("employee", "E1"), docs[0]["doc_id"]); R["sample_fields_present"] = sorted(k for k, v in D["fields"].items() if v.get("value") is not None)[:20]; R["sample_ocr_source"] = D["record"].get("ocr_source_page1")
R["raw_inbox_files_left"] = len(list((TMP / "out/ingest/inbox").glob("*"))); R["imap_commands_sent"] = sorted(set(srv.log)); R["mailbox_modified"] = bool(srv.stored); R["permission_violations_under_HRBC_OUT"] = len(perms.audit(TMP / "out"))
# poison message: pdfinfo missing for one new message; a later valid one must still complete
real = subprocess.run
def boom(cmd, *a, **k):
    if cmd and cmd[0] == "pdfinfo": raise FileNotFoundError("pdfinfo")
    return real(cmd, *a, **k)
srv.add(mail("<m9@x>", atts=[("new.pdf", pdf + b"\n%x" , "application", "pdf")])); srv.add(mail("<m10@x>", atts=[("slip2.jpg", jpg + b"\x00", "image", "jpeg")]))
import ingest.validate as VV; VV.subprocess.run = boom
try: R["poison_poll_stats"] = PL.poll_once(prov, initial="all", now=2000)
finally: VV.subprocess.run = real
R["poison_failure_rows"] = [dict(key=r["key"], state=r["state"], attempts=r["attempts"], error_code=r["error_code"]) for r in c.execute("SELECT * FROM inbox_failures")]
R["later_valid_message_ingested"] = bool(c.execute("SELECT 1 FROM emails WHERE message_id='<m10@x>'").fetchone()); R["failed_message_not_recorded_as_email"] = not c.execute("SELECT 1 FROM emails WHERE message_id='<m9@x>'").fetchone()
R["poll_after_backoff_recovers"] = PL.poll_once(prov, initial="all", now=2000 + 10_000); R["health_after_recovery"] = {k: v for k, v in PL.health().items() if k in ("jobs_dead", "messages_dead", "messages_retrying", "problems")}
R["elapsed_s"] = round(time.time() - t0, 1); out = Path(__file__).with_name("local_imap_smoke_result.json"); out.write_text(json.dumps(R, indent=1, default=str)); print(json.dumps(R, indent=1, default=str))
