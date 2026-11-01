"""AUDIT SMOKE TEST (local only): a minimal IMAP4rev1 server on 127.0.0.1 with a self-signed certificate drives the REAL ImapProvider / parse_rfc822 / ingest pipeline / BillService (real OCR, Task A, Task B models). Not a live provider: it proves the wire path of our client code and the end-to-end flow, not Gmail/Outlook behaviour.
All state goes to a temporary HRBC_OUT; no network leaves the machine; fixtures are synthetic or public-licence documents. Usage: cd app && ../.venv/bin/python ../reports/email_ingestion/smoke/local_imap_smoke.py"""
import os, sys, ssl, json, socket, threading, tempfile, subprocess, time, logging, glob, re
from pathlib import Path
from email.message import EmailMessage
ROOT = Path(__file__).resolve().parents[3]; TMP = Path(tempfile.mkdtemp(prefix="imap_smoke_")); os.environ["HRBC_OUT"] = str(TMP / "out"); os.environ["HRBC_INGEST_MAILBOX"] = "expenses@corp.example"
for p in ("app", "scripts", "scripts/taskB"): sys.path.insert(0, str(ROOT / p))
logging.basicConfig(level=logging.WARNING)
# ---- self-signed cert
subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(TMP / "k.pem"), "-out", str(TMP / "c.pem"), "-days", "1", "-subj", "/CN=localhost"], capture_output=True, check=True)
# ---- fixtures
pdf = next(Path(ROOT / "datasets/public/my_receipts").rglob("*.pdf")).read_bytes()
png = next((ROOT / "datasets/synthetic_doctype/v1/images/salary_slip").glob("*.jpg")).read_bytes()
def mail(mid, sender="asha@corp.example", atts=(), to="expenses@corp.example", auth="mx.corp; dkim=pass; spf=pass", body="Please find the bill.", html=None):
    m = EmailMessage(); m["Message-ID"] = mid; m["From"] = sender; m["To"] = to; m["Subject"] = "bill " + mid; m["Date"] = "Mon, 19 Oct 2026 10:00:00 +0000"; m["Authentication-Results"] = auth
    m.set_content(body)
    if html: m.add_alternative(html, subtype="html")
    for fn, data, mt, st in atts: m.add_attachment(data, maintype=mt, subtype=st, filename=fn)
    return m
fwd_inner = mail("<inner@x>", sender="vendor@shop.example", atts=[("inner.pdf", pdf, "application", "pdf")])
fwd = mail("<m3@x>", body="Fwd: see attached"); fwd.add_attachment(fwd_inner)           # forwarded as message/rfc822
MBOX = [  # uid -> message
    mail("<m1@x>", atts=[("bill.pdf", pdf, "application", "pdf"), ("slip.jpg", png, "image", "jpeg")]),   # 2 attachments
    mail("<m2@x>", sender="stranger@evil.example", atts=[("bill.pdf", pdf, "application", "pdf")]),        # unknown sender
    fwd,                                                                                                    # forwarded message containing a PDF
    mail("<m4@x>", atts=[("a.zip", b"PK\x03\x04" + b"0" * 2000, "application", "zip")]),                    # unsafe file
    mail("<m5@x>", body="Invoice total Rs 1,250.00 inside body", html="<html><body><h1>Tax Invoice</h1><p>Total Rs 1,250.00 GSTIN 29ABCDE1234F1Z5</p></body></html>"),  # bill only in body
    mail("<m1@x>", atts=[("bill.pdf", pdf, "application", "pdf")]),                                         # same Message-ID re-sent (uid 6)
    mail("<m7@x>", atts=[("bill2.pdf", pdf, "application", "pdf")]),                                        # same file under a new message
]
SEEN = set(); LOG = []; ALLOW_LOGIN = {"ok": True}
class Handler(threading.Thread):
    def __init__(self, sock): super().__init__(daemon=True); self.s = sock
    def run(self):
        f = self.s.makefile("rwb", buffering=0); w = lambda b: f.write(b if isinstance(b, bytes) else b.encode()); w("* OK ready\r\n")
        while True:
            line = f.readline()
            if not line: return
            tag, _, rest = line.decode().strip().partition(" "); cmd = rest.split(" ")[0].upper(); LOG.append(cmd + (" " + rest.split(" ")[1].upper() if cmd == "UID" else ""))
            if cmd == "CAPABILITY": w("* CAPABILITY IMAP4rev1\r\n"); w(f"{tag} OK done\r\n")
            elif cmd == "LOGIN": w(f"{tag} OK logged in\r\n" if ALLOW_LOGIN["ok"] else f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials\r\n")
            elif cmd == "SELECT": w(f"* {len(MBOX)} EXISTS\r\n"); w(f"{tag} OK [READ-WRITE] selected\r\n")
            elif cmd == "UID":
                parts = rest.split(" ")
                if parts[1].upper() == "SEARCH": w("* SEARCH " + " ".join(str(i + 1) for i in range(len(MBOX)) if (i + 1) not in SEEN) + "\r\n"); w(f"{tag} OK done\r\n")
                elif parts[1].upper() == "FETCH":
                    uid = int(parts[2]); raw = MBOX[uid - 1].as_bytes(); w(f"* {uid} FETCH (UID {uid} BODY[] {{{len(raw)}}}\r\n"); w(raw); w(")\r\n"); w(f"{tag} OK done\r\n")
                elif parts[1].upper() == "STORE": SEEN.add(int(parts[2])); w(f"* {parts[2]} FETCH (FLAGS (\\Seen))\r\n"); w(f"{tag} OK done\r\n")
            elif cmd == "LOGOUT": w("* BYE\r\n"); w(f"{tag} OK bye\r\n"); self.s.close(); return
            else: w(f"{tag} OK\r\n")
def serve():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(TMP / "c.pem", TMP / "k.pem"); srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(5); PORT.append(srv.getsockname()[1])
    def loop():
        while True:
            c, _ = srv.accept()
            try: Handler(ctx.wrap_socket(c, server_side=True)).start()
            except Exception: pass
    threading.Thread(target=loop, daemon=True).start()
PORT = []; serve()
import service as SV
from ingest import pipeline as PL, db, api
from ingest.provider import ImapProvider
R = {}
PL.add_sender("asha@corp.example", "E1", "audit"); prov = ImapProvider("127.0.0.1", "u", "p", "INBOX", PORT[0])
R["tls_self_signed_cert_accepted_without_verification"] = True      # reaching here later proves it; set False below if connect fails
t0 = time.time(); svc = SV.BillService().load(); R["models_loaded"] = {"ocr": svc.ocr_engine is not None, "task_a": svc.A is not None, "task_b": svc.B is not None, "errors": svc.errors}
n = PL.poll_once(prov); R["poll1_messages_ingested"] = n; R["server_seen_after_poll1"] = sorted(SEEN)
c = db.conn(); R["emails_after_poll1"] = [(r["id"], r["status"], r["reason"], r["n_attachments"]) for r in c.execute("SELECT * FROM emails ORDER BY id")]
R["attachments_after_ingest"] = [(r["email_id"], r["idx"], r["filename"], r["state"], r["error_code"]) for r in c.execute("SELECT * FROM attachments ORDER BY id")]
k = PL.drain(svc.process, prov); R["jobs_processed"] = k
R["attachments_after_worker"] = [(r["email_id"], r["filename"], r["state"], r["error"]) for r in c.execute("SELECT * FROM attachments ORDER BY id")]
R["emails_after_worker"] = [(r["id"], r["status"]) for r in c.execute("SELECT * FROM emails ORDER BY id")]
n2 = PL.poll_once(prov); R["poll2_messages_ingested(should be 0)"] = n2; R["rejected_message_marked_seen_on_server"] = 2 in SEEN
docs = api.list_documents(api.Principal("employee", "E1")); R["employee_E1_documents"] = [dict(status=d["status"], category=d["category"], document_type=d["document_type"], channel=d["channel"], has_email=bool(d["email"])) for d in docs]
R["employee_E2_documents"] = len(api.list_documents(api.Principal("employee", "E2")))
if docs:
    D = api.get_document(api.Principal("employee", "E1"), docs[0]["doc_id"]); R["sample_document_fields_present"] = sorted(k for k, v in D["fields"].items() if v.get("value") is not None)[:20]; R["sample_task_a"] = {k: (D["record"].get("task_a") or {}).get(k) for k in ("category", "route")}; R["sample_ocr_source"] = D["record"].get("ocr_source_page1")
R["notifications_E1"] = len(api.notifications(api.Principal("employee", "E1")))
R["raw_inbox_files_left"] = len(list((TMP / "out/ingest/inbox").glob("*"))); R["imap_commands_seen"] = sorted(set(LOG))
# expired / rejected credentials
ALLOW_LOGIN["ok"] = False
try: prov.fetch_new(); R["bad_credentials"] = "no error?!"
except Exception as e: R["bad_credentials"] = f"{type(e).__name__} (raised to caller; run.py loop logs the class name only and sleeps)"
ALLOW_LOGIN["ok"] = True
# poison message: pdfinfo missing -> exception type escapes ingest_message?
SEEN.clear(); import ingest.validate as VV
orig = subprocess.run
def boom(cmd, *a, **k):
    if cmd and cmd[0] == "pdfinfo": raise FileNotFoundError("pdfinfo")
    return orig(cmd, *a, **k)
subprocess.run = boom
db_before = len(c.execute("SELECT 1 FROM emails").fetchall())
MBOX.append(mail("<m8@x>", atts=[("new.pdf", b"%PDF-1.4\n" + os.urandom(3000), "application", "pdf")])); MBOX.append(mail("<m9@x>", atts=[("slip2.jpg", png + b"\x00", "image", "jpeg")]))
try: PL.poll_once(prov); R["poison_message_blocks_later_mail"] = False
except Exception as e: R["poison_message_blocks_later_mail"] = f"poll_once raised {type(e).__name__}: later messages in the same poll were not ingested; message stays UNSEEN and fails identically every cycle"
subprocess.run = orig
R["emails_total_after_poison"] = len(db.conn().execute("SELECT 1 FROM emails").fetchall()) - db_before
R["elapsed_s"] = round(time.time() - t0, 1)
out = Path(__file__).with_name("local_imap_smoke_result.json"); out.write_text(json.dumps(R, indent=1, default=str)); print(json.dumps(R, indent=1, default=str))
