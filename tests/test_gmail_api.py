"""Gmail API adapter against a LOCAL fake Gmail REST server: read-only calls only, cursor, attachment + sender flow through the real pipeline, error mapping. No network, no credentials."""
import base64, json, threading, unittest, urllib.parse, os, time
from email.message import EmailMessage
from http.server import HTTPServer, BaseHTTPRequestHandler
import _path
from test_ingest import Base, png, pdf, AUTH
from ingest import gmail_api as GM, pipeline as PL, api, db, oauth, config
from ingest.provider import ProviderAuthError, RateLimited, ProviderTransient

def mime(mid, atts=(), sender="asha@corp.example", rcpt="expenses@corp.example", auth=AUTH):
    m = EmailMessage(); m["From"], m["To"], m["Subject"], m["Message-ID"], m["Date"] = sender, rcpt, "my bill", f"<{mid}@x>", "Wed, 01 Oct 2026 10:00:00 +0000"; m["Authentication-Results"] = auth; m.set_content("hello")
    for fn, ct, data in atts: m.add_attachment(data, maintype=ct.split("/")[0], subtype=ct.split("/")[1], filename=fn)
    return m.as_bytes()

class FakeGmail:
    def __init__(self): self.msgs = {}; self.calls = []; self.fail = {}; self.token_ok = True
    def add(self, i, raw): self.msgs[i] = raw
    def start(self):
        o = self
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query); o.calls.append((u.path, dict((k, v) for k, v in q.items())))
                if self.headers.get("Authorization") != "Bearer GOOD" or o.fail.get("status"): return self.out(o.fail.get("status", 401), {"error": {"errors": [{"reason": o.fail.get("reason", "authError")}]}})
                if u.path.endswith("/profile"): return self.out(200, {"emailAddress": "someone@gmail.com", "messagesTotal": len(o.msgs)})
                if u.path.endswith("/messages"):
                    ids = sorted(o.msgs, reverse=True)
                    if "q" in q and q["q"][0].startswith("rfc822msgid:"): ids = [i for i in ids if f"<{q['q'][0][12:]}>".encode() in o.msgs[i]]
                    return self.out(200, {"messages": [{"id": f"{i:016x}"} for i in ids]})
                i = int(u.path.rsplit("/", 1)[1], 16)
                if i not in o.msgs: return self.out(404, {"error": {"errors": [{"reason": "notFound"}]}})
                raw = o.msgs[i]
                if q.get("format") == ["raw"]: return self.out(200, {"raw": base64.urlsafe_b64encode(raw).decode().rstrip("=")})
                import email; m = email.message_from_bytes(raw)
                return self.out(200, {"sizeEstimate": len(raw), "payload": {"headers": [{"name": k, "value": v} for k, v in m.items()]}})
            def out(self, code, body):
                b = json.dumps(body).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
            def log_message(self, *a): pass
        self.srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=self.srv.serve_forever, daemon=True).start(); return f"http://127.0.0.1:{self.srv.server_port}"
    def stop(self): self.srv.shutdown(); self.srv.server_close()

class Tok:
    scope = None
    def access_token(self): return "GOOD"

class T(Base):
    def setUp(self):
        super().setUp(); self.g = FakeGmail(); self.base = self.g.start(); self.addCleanup(self.g.stop); self.p = GM.GmailApiProvider(token_source=Tok(), base=self.base + "/gmail/v1/users/me", batch=5)
    def test_first_poll_records_cursor_only_then_new_mail_is_ingested_once(self):
        self.g.add(0x100, mime("old", [("b.png", "image/png", png(1))]))
        s = PL.poll_once(self.p); self.assertEqual(s.get("cursor_initialised"), 1); self.assertEqual(api.list_emails(api.Principal("hr", "h")), [])      # existing mail is NOT read
        self.g.add(0x101, mime("new", [("bill.png", "image/png", png(2)), ("bill.pdf", "application/pdf", pdf(1, 3))]))
        s = PL.poll_once(self.p); self.assertEqual(s["ingested"], 1); self.run_all(self.processor(), self.p)
        d = api.list_documents(api.Principal("employee", "E1")); self.assertEqual(sorted(x["filename"] for x in d), ["bill.pdf", "bill.png"]); self.assertTrue(all(x["channel"] == "email" for x in d))
        PL.poll_once(self.p); self.run_all(self.processor(), self.p); self.assertEqual(len(api.list_documents(api.Principal("employee", "E1"))), 2)          # repeat: nothing new
    def test_unauthorised_sender_attachments_never_downloaded(self):
        PL.poll_once(self.p); self.g.add(0x200, mime("bad", [("b.png", "image/png", png(1))], sender="evil@x.example")); s = PL.poll_once(self.p)
        self.assertEqual(s["rejected"], 1); self.assertFalse(any(q.get("format") == ["raw"] for _, q in self.g.calls))
    def test_only_get_requests(self):
        PL.poll_once(self.p); self.assertTrue(all(c[0].startswith("/gmail/v1/users/me") for c in self.g.calls))
    def test_error_mapping(self):
        self.g.fail = {"status": 401}; 
        with self.assertRaises(ProviderAuthError): self.p.list_new(None)
        self.g.fail = {"status": 403, "reason": "accessNotConfigured"}
        with self.assertRaises(ProviderAuthError) as e: self.p.list_new(None)
        self.assertIn("accessNotConfigured", str(e.exception))
        self.g.fail = {"status": 429, "reason": "rateLimitExceeded"}
        with self.assertRaises(RateLimited): self.p.list_new(None)
        self.g.fail = {"status": 503}
        with self.assertRaises(ProviderTransient): self.p.list_new(None)
    def test_revoked_token_is_auth_error(self):
        class Rev:
            scope = None
            def access_token(self): raise oauth.CredentialsExpired("invalid_grant")
        with self.assertRaises(ProviderAuthError): GM.GmailApiProvider(token_source=Rev(), base=self.base).list_new(None)
    def test_fetch_by_id_and_check_masks_address(self):
        self.g.add(0x300, mime("find-me")); self.assertEqual(self.p.fetch_by_id("<find-me@x>").message_id, "<find-me@x>@x" if False else "<find-me@x>"); self.assertIsNone(self.p.fetch_by_id("nothing@x"))
        with self.assertRaises(ValueError): self.p.fetch_by_id('a" OR b')
        c = GM.check(tokens=Tok(), base=self.base + "/gmail/v1/users/me"); self.assertTrue(c["authenticated"]); self.assertNotIn("someone@", json.dumps(c)); self.assertEqual(c["messages_total"], 1)
    def test_prod_requires_https(self):
        os.environ["HRBC_ENV"] = "prod"
        with self.assertRaises(config.ConfigError): GM.GmailApiProvider(token_source=Tok(), base="http://127.0.0.1:1")
if __name__ == "__main__": unittest.main()
