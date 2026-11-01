"""Production-hardening regression tests for email ingestion (one class per audit finding / phase). Local IMAP stub + throw-away PKI + FakeProvider; the heavy models are replaced by a stub processor except where stated.
Real-provider behaviour is NOT covered here: see reports/email_ingestion/PRODUCTION_READINESS_TESTS.md."""
import os, io, json, time, stat, uuid, logging, tempfile, threading, unittest, subprocess, http.server, socketserver, ssl
from email.message import EmailMessage
from pathlib import Path
from unittest import mock
import _path
import numpy as np
from PIL import Image
import imap_stub as STUB
import service as SV
from ingest import db, pipeline as PL, api, webhook as WH, validate as VA, config, tls, oauth, perms, policy, run as RUN
from ingest import provider as PV
from ingest.provider import Message, Attachment, FakeProvider, ImapProvider, parse_rfc822
from tax.store import PermissionDenied

def png(seed=0): r = np.random.RandomState(seed); b = io.BytesIO(); Image.fromarray(r.randint(0, 255, (160, 160, 3), dtype="uint8")).save(b, "PNG"); return b.getvalue()
def pdf(seed=0): r = np.random.RandomState(seed); b = io.BytesIO(); Image.fromarray(r.randint(0, 255, (120, 120, 3), dtype="uint8")).save(b, "PDF"); return b.getvalue()
AUTH = "mx.example; dkim=pass header.d=corp.example; spf=pass smtp.mailfrom=corp.example"
def msg(mid, sender="asha@corp.example", atts=(), auth=AUTH, rcpt="expenses@corp.example", **kw):
    return Message("fake", mid, sender, rcpt, "my bills", "2026-10-01T10:00:00", [a if isinstance(a, Attachment) else Attachment(*a) for a in atts], auth, **kw)
def mime(mid, sender="asha@corp.example", to="expenses@corp.example", atts=(), auth=AUTH, body="Please see attached.", html=None, extra=None):
    m = EmailMessage(); m["Message-ID"] = mid; m["From"] = sender; m["To"] = to; m["Subject"] = "bill"; m["Date"] = "Mon, 19 Oct 2026 10:00:00 +0000"
    for a in ([auth] if isinstance(auth, str) else auth): m["Authentication-Results"] = a
    for k, v in (extra or {}).items(): m[k] = v
    m.set_content(body)
    if html: m.add_alternative(html, subtype="html")
    for fn, data, mt, st in atts: m.add_attachment(data, maintype=mt, subtype=st, filename=fn)
    return m
PDFA = lambda seed=1, name="b.pdf": (name, "application/pdf", pdf(seed)); PNGA = lambda seed=1, name="b.png": (name, "image/png", png(seed))

class Base(unittest.TestCase):
    def setUp(self):
        env = dict(os.environ); self.addCleanup(lambda: (os.environ.clear(), os.environ.update(env)))
        self.out = tempfile.mkdtemp(prefix="hrbc_hard_"); os.environ.update(HRBC_OUT=self.out, HRBC_ENV="test", HRBC_INGEST_MAILBOX="expenses@corp.example", HRBC_INGEST_AUTHSERV_ID="mx.example", HRBC_INGEST_REQUIRE_AUTH="1")
        for k in ("HRBC_INGEST_MAILBOX_ALIASES", "HRBC_INGEST_ALLOWED_DOMAINS", "HRBC_IMAP_MARK_SEEN", "HRBC_IMAP_CA_FILE", "HRBC_IMAP_INITIAL"): os.environ.pop(k, None)
        o = Path(self.out); old = (SV.OUT, SV.RESULTS, SV.CLASSIFIED, SV.UNCLASSIFIED); SV.OUT, SV.RESULTS, SV.CLASSIFIED, SV.UNCLASSIFIED = o, o / "results", o / "classified", o / "unclassified"
        self.addCleanup(lambda: (setattr(SV, "OUT", old[0]), setattr(SV, "RESULTS", old[1]), setattr(SV, "CLASSIFIED", old[2]), setattr(SV, "UNCLASSIFIED", old[3])))
        PL.add_sender("asha@corp.example", "E1", "admin"); PL.add_sender("ravi@corp.example", "E2", "admin"); self.calls = []; self.sources = []
    def processor(self, fail=()):
        fail = list(fail)
        def proc(data, filename, owner, source):
            self.calls.append((filename, owner)); self.sources.append(source)
            if fail: raise fail.pop(0)
            did = source["doc_id"]; d = SV.RESULTS / did; d.mkdir(parents=True, exist_ok=True)
            rec = dict(doc_id=did, original_filename=filename, uploaded_at="2026-10-01T10:00:00", owner=owner, source=source, status="done", category="food", result_dir=f"results/{did}", task_a=dict(label="food", confidence=0.95, route="auto"),
                       profile=dict(document_type="expense_bill", outcome="classified", validity={"completeness": None, "missing_fields": []}), field_report={}, finished_at="x")
            (d / "record.json").write_text(json.dumps(rec)); return rec
        return proc
    def emails(self): return [dict(r) for r in db.conn().execute("SELECT * FROM emails ORDER BY id")]
    def poll(self, prov, now=None, initial="all"): return PL.poll_once(prov, now=now, initial=initial)

# =============================================================== 2.1 verified TLS
class VerifiedTls(Base):
    @classmethod
    def setUpClass(cls): cls.pki = STUB.make_pki()
    def serve(self, cert, **kw):
        s = STUB.StubServer(cert, [mime("<a@x>")], **kw); port = s.start(); self.addCleanup(s.stop); return s, port
    def provider(self, port, ca=None, **kw):
        return ImapProvider("localhost", "user", "pw", "INBOX", port, ssl_context=tls.context(ca), **kw)
    def test_trusted_ca_and_matching_hostname_accepted_in_test_mode(self):
        s, port = self.serve(self.pki["good"]); p = self.provider(port, self.pki["ca"])
        with p.session(): refs, uv, _ = p.list_new(None, "all"); self.assertEqual([r.uid for r in refs], [1])
    def test_self_signed_rejected_by_default(self):
        s, port = self.serve(self.pki["selfsigned"])
        with self.assertRaises(PV.TlsVerificationError):
            with self.provider(port).session(): pass
    def test_valid_cert_from_untrusted_ca_rejected(self):
        s, port = self.serve(self.pki["good"])
        with self.assertRaises(PV.TlsVerificationError):
            with self.provider(port).session(): pass
    def test_hostname_mismatch_rejected_even_with_trusted_ca(self):
        s, port = self.serve(self.pki["mismatch"])
        with self.assertRaises(PV.TlsVerificationError):
            with self.provider(port, self.pki["ca"]).session(): pass
    def test_expired_certificate_rejected_even_with_trusted_ca(self):
        s, port = self.serve(self.pki["expired"])
        with self.assertRaises(PV.TlsVerificationError):
            with self.provider(port, self.pki["ca"]).session(): pass
    def test_default_provider_context_verifies_certificate_and_hostname(self):
        os.environ.update(HRBC_IMAP_HOST="localhost", HRBC_IMAP_USER="user", HRBC_IMAP_PASSWORD="pw"); p = ImapProvider(port=1)
        self.assertEqual(p.ctx.verify_mode, ssl.CERT_REQUIRED); self.assertTrue(p.ctx.check_hostname)
        s, port = self.serve(self.pki["selfsigned"])
        with self.assertRaises(PV.TlsVerificationError):
            with ImapProvider(port=port).session(): pass             # audit finding: this used to connect without verification
    def test_extra_trust_not_allowed_in_prod_and_no_way_to_disable_verification(self):
        os.environ["HRBC_ENV"] = "prod"
        with self.assertRaises(tls.TlsConfigError): tls.context(self.pki["ca"])
        os.environ.update(HRBC_IMAP_HOST="h", HRBC_IMAP_USER="u", HRBC_IMAP_PASSWORD="p", HRBC_IMAP_CA_FILE=self.pki["ca"], HRBC_INGEST_MAILBOX="e@c.example", HRBC_INGEST_AUTHSERV_ID="mx"); self.assertTrue(any("CA_FILE" in x for x in config.validate()))
        self.assertFalse(hasattr(tls, "insecure_context"))
    def test_secrets_never_logged_on_failure(self):
        s, port = self.serve(self.pki["good"], password="s3cr3t-pw"); h = logging.Handler(); recs = []; h.emit = lambda r: recs.append(r.getMessage()); logging.getLogger().addHandler(h); self.addCleanup(lambda: logging.getLogger().removeHandler(h))
        with self.assertRaises(PV.ProviderAuthError):
            with ImapProvider("localhost", "user", "wrong-pw-xyz", "INBOX", port, ssl_context=tls.context(self.pki["ca"])).session(): pass
        PL.poll_once(FakeProvider([]), now=0)
        self.assertFalse(any("wrong-pw-xyz" in r or "s3cr3t-pw" in r for r in recs))
    def test_mailbox_opened_read_only_and_nothing_modified_by_default(self):
        s, port = self.serve(self.pki["good"]); p = self.provider(port, self.pki["ca"])
        PL.add_sender("asha@corp.example", "E1", "a"); self.poll(p)
        self.assertIn("EXAMINE", s.log); self.assertNotIn("SELECT", s.log); self.assertFalse({"STORE", "EXPUNGE", "COPY", "DELETE", "APPEND"} & {x.split(" ")[-1] for x in s.log}); self.assertEqual(s.stored, [])

# =============================================================== 5 auth: password + XOAUTH2 against local stubs
class TokenEndpoint:
    def __init__(self): self.requests = []; self.mode = "ok"; self.expires = 3600; self.rotate = None; self.n = 0
    def start(self):
        te = self
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"])).decode(); te.requests.append(body)
                code, data, hdr = 200, {}, {}
                if te.mode == "invalid_grant": code, data = 400, {"error": "invalid_grant"}
                elif te.mode == "throttle": code, data, hdr = 429, {"error": "rate_limit_exceeded"}, {"Retry-After": "120"}
                elif te.mode == "down": code, data = 503, {}
                else:
                    te.n += 1; data = {"access_token": f"tok{te.n}", "expires_in": te.expires, "token_type": "Bearer"}
                    if te.rotate: data["refresh_token"] = te.rotate
                self.send_response(code); [self.send_header(k, v) for k, v in hdr.items()]; self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(json.dumps(data).encode())
        self.srv = socketserver.TCPServer(("127.0.0.1", 0), H); self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/token"; threading.Thread(target=self.srv.serve_forever, daemon=True).start(); return self.url
    def stop(self): self.srv.shutdown(); self.srv.server_close()
class OAuthAndAuth(Base):
    @classmethod
    def setUpClass(cls): cls.pki = STUB.make_pki()
    def setUp(self):
        super().setUp(); self.te = TokenEndpoint(); self.url = self.te.start(); self.addCleanup(self.te.stop); self.now = [1000.0]
        self.src = lambda **kw: oauth.RefreshTokenSource(self.url, "cid", "csecret", "refresh-1", clock=lambda: self.now[0], **kw)
    def test_refresh_flow_request_shape_caching_and_expiry(self):
        s = self.src(); self.assertEqual(s.access_token(), "tok1"); self.assertEqual(s.access_token(), "tok1"); self.assertEqual(len(self.te.requests), 1)
        body = self.te.requests[0]; self.assertIn("grant_type=refresh_token", body); self.assertIn("refresh_token=refresh-1", body); self.assertIn("client_id=cid", body)
        self.now[0] += 3600; self.assertEqual(s.access_token(), "tok2"); self.assertEqual(len(self.te.requests), 2)
    def test_rotated_refresh_token_is_kept_and_persisted_privately(self):
        self.te.rotate = "refresh-2"; store = Path(self.out) / "rt"; s = self.src(persist_path=str(store)); s.access_token(); self.now[0] += 4000; s.access_token()
        self.assertIn("refresh_token=refresh-2", self.te.requests[1]); self.assertEqual(store.read_text(), "refresh-2"); self.assertEqual(stat.S_IMODE(store.stat().st_mode), 0o600)
    def test_revoked_credentials_are_permanent_error(self):
        self.te.mode = "invalid_grant"
        with self.assertRaises(oauth.CredentialsExpired): self.src().access_token()
    def test_throttle_and_outage_are_transient_with_retry_after(self):
        self.te.mode = "throttle"
        with self.assertRaises(oauth.TransientAuthError) as cm: self.src().access_token()
        self.assertEqual(cm.exception.retry_after, 120.0); self.te.mode = "down"
        with self.assertRaises(oauth.TransientAuthError): self.src().access_token()
    def test_https_required_in_prod(self):
        os.environ["HRBC_ENV"] = "prod"
        with self.assertRaises(config.ConfigError): oauth.RefreshTokenSource("http://127.0.0.1:1/t", "a", "b", "c")
    def test_xoauth2_imap_login_end_to_end_against_stub(self):
        srv = STUB.StubServer(self.pki["good"], [mime("<a@x>")], tokens={"tok1"}); port = srv.start(); self.addCleanup(srv.stop)
        p = ImapProvider("localhost", "user", None, "INBOX", port, token_source=self.src(), ssl_context=tls.context(self.pki["ca"]))
        self.assertEqual(p.auth, "xoauth2")
        with p.session(): self.assertEqual(len(p.list_new(None, "all")[0]), 1)
        self.assertIn("AUTHENTICATE", srv.log); self.assertNotIn("LOGIN", srv.log)
    def test_xoauth2_rejected_token_is_auth_error_and_poll_backs_off(self):
        srv = STUB.StubServer(self.pki["good"], [], tokens={"other"}); port = srv.start(); self.addCleanup(srv.stop)
        p = ImapProvider("localhost", "user", None, "INBOX", port, token_source=self.src(), ssl_context=tls.context(self.pki["ca"]))
        with self.assertRaises(PV.ProviderAuthError):
            with p.session(): pass
        st = PL.poll_once(p, now=1000); self.assertEqual(st["poll_error"], 1); ws = dict(db.conn().execute("SELECT * FROM worker_state").fetchone()); self.assertEqual(ws["last_poll_error"], "credentials_rejected"); self.assertGreater(ws["next_poll_after"], 1000)
    def test_revoked_refresh_token_surfaces_as_credentials_rejected(self):
        self.te.mode = "invalid_grant"; srv = STUB.StubServer(self.pki["good"], [], tokens={"tok1"}); port = srv.start(); self.addCleanup(srv.stop)
        p = ImapProvider("localhost", "user", None, "INBOX", port, token_source=self.src(), ssl_context=tls.context(self.pki["ca"]))
        PL.poll_once(p, now=5); self.assertEqual(db.conn().execute("SELECT last_poll_error FROM worker_state").fetchone()[0], "credentials_rejected")
    def test_rate_limited_login_honours_backoff_and_skips_polls_until_due(self):
        srv = STUB.StubServer(self.pki["good"], [mime("<a@x>")]); srv.login_error = "throttle"; port = srv.start(); self.addCleanup(srv.stop)
        p = ImapProvider("localhost", "user", "pw", "INBOX", port, ssl_context=tls.context(self.pki["ca"]))
        PL.poll_once(p, now=100); ws = dict(db.conn().execute("SELECT * FROM worker_state").fetchone()); self.assertEqual(ws["last_poll_error"], "rate_limited"); due = ws["next_poll_after"]; self.assertGreater(due, 100)
        conns = srv.connections; self.assertEqual(PL.poll_once(p, now=due - 1).get("skipped_backoff"), 1); self.assertEqual(srv.connections, conns)         # no connection while backing off
        srv.login_error = None; PL.poll_once(p, now=due + 1, initial="all"); self.assertEqual(len(self.emails()), 1); self.assertEqual(dict(db.conn().execute("SELECT * FROM worker_state").fetchone())["consecutive_poll_failures"], 0)
    def test_token_endpoint_url_and_secrets_never_in_problem_text(self):
        os.environ.update(HRBC_IMAP_HOST="h", HRBC_IMAP_USER="u", HRBC_IMAP_AUTH="xoauth2", HRBC_OAUTH_TOKEN_URL=self.url, HRBC_OAUTH_CLIENT_ID="cid-secret-value", HRBC_OAUTH_CLIENT_SECRET="csecret-value", HRBC_OAUTH_REFRESH_TOKEN="rt-value", HRBC_TEST_MAILBOX_USER="u", HRBC_TEST_FOLDER="INBOX")
        self.assertEqual(config.validate(), []); os.environ.pop("HRBC_OAUTH_CLIENT_SECRET"); p = config.validate(); self.assertTrue(p); self.assertFalse(any("csecret-value" in x or "rt-value" in x for x in p))

# =============================================================== 2.2 sender / recipient authorisation
class Authorisation(Base):
    def auth(self, m): return policy.authorise(m)
    def test_unset_expense_mailbox_fails_closed(self):
        os.environ.pop("HRBC_INGEST_MAILBOX"); self.assertEqual(self.auth(msg("<1>")), (None, "mailbox_not_configured")); self.assertTrue(any("HRBC_INGEST_MAILBOX" in p for p in config.validate(imap=False)))
        r = PL.ingest_message(msg("<1>", atts=[PNGA()])); self.assertEqual(r["status"], "REJECTED"); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM attachments").fetchone()[0], 0)
    def test_wrong_recipient_rejected_aliases_and_cc_accepted(self):
        self.assertEqual(self.auth(msg("<1>", rcpt="other@corp.example"))[1], "not_addressed_to_expense_mailbox")
        os.environ["HRBC_INGEST_MAILBOX_ALIASES"] = "bills@corp.example, claims@corp.example"; self.assertEqual(self.auth(msg("<2>", rcpt="bills@corp.example"))[0], "E1")
        self.assertEqual(self.auth(msg("<3>", rcpt="someone@corp.example", recipients=["someone@corp.example", "claims@corp.example"]))[0], "E1")        # Cc / Delivered-To
        self.assertEqual(self.auth(msg("<4>", rcpt="someone@corp.example", recipients=["someone@corp.example"]))[0], None)
    def test_allow_list_is_the_authorisation_authentication_alone_is_not(self):
        self.assertEqual(self.auth(msg("<1>", sender="stranger@corp.example", auth="mx.example; dkim=pass header.d=corp.example; dmarc=pass header.from=corp.example"))[1], "sender_not_authorised")
    def test_forged_from_with_pass_for_attackers_domain_rejected(self):
        forged = "mx.example; dkim=pass header.d=evil.example; spf=pass smtp.mailfrom=bounce@evil.example"; self.assertEqual(self.auth(msg("<1>", auth=forged))[1], "sender_not_authenticated")
    def test_aligned_evidence_variants_accepted(self):
        for a in ("mx.example; dkim=pass header.d=corp.example", "mx.example; spf=pass smtp.mailfrom=mail.corp.example", "mx.example; dmarc=pass header.from=corp.example", "MX.Example; dkim=fail header.d=x.example; dkim=pass header.d=corp.example"):
            self.assertEqual(self.auth(msg("<x>", auth=a))[0], "E1", a)
    def test_dmarc_fail_rejects_even_with_other_aligned_pass(self):
        self.assertEqual(self.auth(msg("<1>", auth="mx.example; dkim=pass header.d=corp.example; dmarc=fail header.from=corp.example"))[1], "sender_authentication_failed")
    def test_authentication_results_from_a_foreign_server_are_ignored(self):
        self.assertEqual(self.auth(msg("<1>", auth="attacker.example; dkim=pass header.d=corp.example; spf=pass smtp.mailfrom=corp.example"))[1], "sender_not_authenticated")
        self.assertEqual(self.auth(msg("<2>", auth="", auth_headers=["attacker.example; dkim=pass header.d=corp.example", "mx.example; dkim=none"]))[1], "sender_not_authenticated")
    def test_org_domain_alignment_does_not_cross_public_suffixes(self):
        self.assertTrue(policy.aligned("mail.corp.example", "corp.example")); self.assertFalse(policy.aligned("evil.co.uk", "corp.co.uk")); self.assertFalse(policy.aligned("co.uk", "corp.co.uk")); self.assertFalse(policy.aligned("com", "corp.com"))
    def test_multiple_from_addresses_rejected(self): self.assertEqual(self.auth(msg("<1>", from_count=2))[1], "ambiguous_from")
    def test_domain_allow_list_optional_restriction(self):
        os.environ["HRBC_INGEST_ALLOWED_DOMAINS"] = "other.example"; self.assertEqual(self.auth(msg("<1>"))[1], "sender_domain_not_allowed")
    def test_employee_aliases_map_to_one_employee(self):
        PL.add_sender("a.sharma@corp.example", "E1", "admin"); m = msg("<1>", sender="a.sharma@corp.example", auth="mx.example; dkim=pass header.d=corp.example"); self.assertEqual(self.auth(m)[0], "E1")
    def test_forwarded_mail_identity_is_the_outer_sender_only(self):
        inner_by_employee = mime("<in1>", sender="asha@corp.example", atts=[("v.pdf", pdf(3), "application", "pdf")]); outer = mime("<o1>", sender="stranger@evil.example", auth="mx.example; dkim=pass header.d=evil.example"); outer.add_attachment(inner_by_employee)
        m = parse_rfc822(outer.as_bytes()); self.assertEqual(PL.ingest_message(m)["status"], "REJECTED")                                  # inner employee header is not trusted
        inner_by_vendor = mime("<in2>", sender="vendor@shop.example", atts=[("v.pdf", pdf(4), "application", "pdf")]); outer2 = mime("<o2>"); outer2.add_attachment(inner_by_vendor)
        m2 = parse_rfc822(outer2.as_bytes()); self.assertEqual(self.auth(m2)[0], "E1"); self.assertTrue(any(a.filename == "v.pdf" for a in m2.attachments))       # employee forwarding a vendor bill: accepted, owned by the employee
    def test_unauthorised_messages_are_rejected_before_any_body_is_fetched(self):
        fp = FakeProvider([msg("<1>", sender="stranger@evil.example", atts=[PNGA()]), msg("<2>", atts=[PNGA(2)])]); st = self.poll(fp)
        self.assertEqual(st["rejected"], 1); self.assertEqual(fp.fetch_calls, [2])                                                           # fetch_message was never called for uid 1
        self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM attachments WHERE email_id=(SELECT id FROM emails WHERE message_id='<1>')").fetchone()[0], 0)
    def test_require_auth_cannot_be_disabled_in_prod(self):
        os.environ.update(HRBC_ENV="prod", HRBC_INGEST_REQUIRE_AUTH="0"); self.assertTrue(any("REQUIRE_AUTH=0" in p for p in config.validate(imap=False))); os.environ["HRBC_ENV"] = "dev"; self.assertFalse(any("REQUIRE_AUTH" in p for p in config.validate(imap=False)))
        os.environ.update(HRBC_ENV="prod", HRBC_INGEST_REQUIRE_AUTH="1"); os.environ.pop("HRBC_INGEST_AUTHSERV_ID"); self.assertTrue(any("AUTHSERV" in p for p in config.validate(imap=False)))

# =============================================================== 3.1 / 3.2 resilience, cursor, idempotency
def no_pdfinfo(real=subprocess.run):
    def fake(cmd, *a, **k):
        if cmd and cmd[0] == "pdfinfo": raise FileNotFoundError("pdfinfo")
        return real(cmd, *a, **k)
    return fake
class Resilience(Base):
    def test_pdfinfo_missing_does_not_block_later_valid_messages(self):          # the audit's poison-message defect
        fp = FakeProvider([msg("<m1>", atts=[PDFA(1)]), msg("<m2>", atts=[PNGA(2)]), msg("<m3>", atts=[PNGA(3)])])
        with mock.patch.object(VA.subprocess, "run", side_effect=no_pdfinfo()): st = self.poll(fp, now=1000)
        self.assertEqual(st["failed"], 1); self.assertEqual(st["ingested"], 2); self.assertEqual([e["message_id"] for e in self.emails()], ["<m2>", "<m3>"])      # the failed one is NOT recorded as an email, let alone COMPLETED
        f = dict(db.conn().execute("SELECT * FROM inbox_failures").fetchone()); self.assertEqual((f["state"], f["attempts"], f["error_code"], f["message_id"]), ("retry", 1, "validation_tool_unavailable", "<m1>")); self.assertGreater(f["next_run"], 1000)
        self.assertEqual(db.conn().execute("SELECT last_uid FROM poll_state").fetchone()[0], 3)                                                                    # cursor moved past the failure
        self.assertEqual(len(list((Path(self.out) / "ingest" / "inbox").glob("*"))), 2)                                                                            # rolled back: only the two ingested messages have stored files (no orphan from the failed one)
        st2 = self.poll(fp, now=1000 + 10_000); self.assertEqual(st2["recovered"], 1); self.assertEqual(db.conn().execute("SELECT state FROM inbox_failures").fetchone()[0], "resolved"); self.assertEqual(len(self.emails()), 3)
    def test_retries_are_bounded_visible_and_end_in_dead_letter(self):
        fp = FakeProvider([msg("<m1>", atts=[PDFA(1)])]); t = 1000; seen = []
        with mock.patch.object(VA.subprocess, "run", side_effect=no_pdfinfo()):
            for i in range(PL.MAX_MSG_ATTEMPTS + 3):
                self.poll(fp, now=t); r = db.conn().execute("SELECT state,attempts,next_run FROM inbox_failures").fetchone(); seen.append((r["state"], r["attempts"], r["next_run"])); t = r["next_run"] + 1
        self.assertEqual(seen[PL.MAX_MSG_ATTEMPTS - 1][:2], ("dead", PL.MAX_MSG_ATTEMPTS)); self.assertTrue(all(s[1] <= PL.MAX_MSG_ATTEMPTS for s in seen)); self.assertEqual(len(fp.fetch_calls), PL.MAX_MSG_ATTEMPTS)    # no attempts after the dead-letter
        gaps = [seen[i][2] for i in range(PL.MAX_MSG_ATTEMPTS - 1)]; self.assertTrue(all(gaps[i + 1] - gaps[i] > 0 for i in range(len(gaps) - 1)))                                         # backoff grows
        h = PL.health(); self.assertEqual(h["messages_dead"], 1); self.assertIn("dead_letters_present", h["problems"]); self.assertEqual(self.emails(), [])
        self.assertEqual(len(api.list_failures(api.Principal("hr", "H1"))), 1); self.assertEqual(api.list_failures(api.Principal("hr", "H1"))[0]["error_code"], "validation_tool_unavailable")
        with self.assertRaises(PermissionDenied): api.list_failures(api.Principal("employee", "E1"))
    def test_dead_letter_recovery_after_the_cause_is_fixed(self):
        fp = FakeProvider([msg("<m1>", atts=[PDFA(1)])]); t = 1000
        with mock.patch.object(VA.subprocess, "run", side_effect=no_pdfinfo()):
            for i in range(PL.MAX_MSG_ATTEMPTS): self.poll(fp, now=t); t = db.conn().execute("SELECT next_run FROM inbox_failures").fetchone()[0] + 1
        self.assertEqual(db.conn().execute("SELECT state FROM inbox_failures").fetchone()[0], "dead"); self.poll(fp, now=t + 10_000); self.assertEqual(self.emails(), [])
        with self.assertRaises(PermissionDenied): api.requeue_failures(api.Principal("employee", "E1"))
        self.assertEqual(api.requeue_failures(api.Principal("hr", "H1")), 1); self.poll(fp, now=t + 20_000); self.assertEqual(len(self.emails()), 1); self.assertEqual(db.conn().execute("SELECT state FROM inbox_failures").fetchone()[0], "resolved")
    def test_unexpected_provider_error_on_one_message_is_isolated(self):
        fp = FakeProvider([msg("<m1>", atts=[PNGA(1)]), msg("<m2>", atts=[PNGA(2)])]); fp.fail_fetch[1] = [PV.ProviderTransient("boom")]; st = self.poll(fp, now=1)
        self.assertEqual((st["failed"], st["ingested"]), (1, 1)); self.assertEqual([e["message_id"] for e in self.emails()], ["<m2>"])
    def test_first_poll_starts_at_the_current_end_unless_initial_all(self):
        fp = FakeProvider([msg("<old>", atts=[PNGA(1)])]); st = self.poll(fp, initial="new_only"); self.assertEqual(st["cursor_initialised"], 1); self.assertEqual(self.emails(), [])
        fp.add(msg("<new>", atts=[PNGA(2)])); self.poll(fp, initial="new_only"); self.assertEqual([e["message_id"] for e in self.emails()], ["<new>"])
    def test_repeated_polls_and_arrivals_during_a_poll_create_no_duplicates(self):
        fp = FakeProvider([msg("<m1>", atts=[PNGA(1)])]); self.poll(fp); self.poll(fp); fp.add(msg("<m2>", atts=[PNGA(2)])); self.poll(fp); self.poll(fp)
        self.assertEqual(len(self.emails()), 2); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 2)
    def test_crash_between_ingest_and_cursor_update_is_idempotent(self):
        fp = FakeProvider([msg("<m1>", atts=[PNGA(1)])]); real = db.set_cursor; boom = {"n": 0}
        def flaky(*a, **k):
            boom["n"] += 1
            if boom["n"] == 1: raise sqlite3_error()
            return real(*a, **k)
        import sqlite3; sqlite3_error = lambda: sqlite3.OperationalError("simulated crash")
        with mock.patch.object(db, "set_cursor", flaky):
            with self.assertRaises(sqlite3.OperationalError): self.poll(fp)
        self.poll(fp); self.assertEqual(len(self.emails()), 1); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM attachments").fetchone()[0], 1); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
    def test_uidvalidity_change_rereads_without_duplicates(self):
        fp = FakeProvider([msg("<m1>", atts=[PNGA(1)])]); self.poll(fp); fp.uidvalidity = "2"; st = self.poll(fp); self.assertEqual(st["uidvalidity_changed"], 1); self.assertEqual(st["duplicate"], 1); self.assertEqual(len(self.emails()), 1)
    def test_worker_crash_mid_processing_resumes_with_the_same_document_id(self):
        PL.ingest_message(msg("<m1>", atts=[PNGA(1)])); seen = []
        def crashing(data, filename, owner, source): seen.append(source["doc_id"]); raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): PL.step(crashing)
        db.conn().execute("UPDATE jobs SET lease_until=0"); proc = self.processor(); self.assertEqual(PL.step(proc), "COMPLETED"); self.assertEqual(seen[0], self.sources[0]["doc_id"]); self.assertRegex(seen[0], r"^\d{8}T\d{6}_[0-9a-f]{8}$")
        self.assertEqual(len(list(SV.RESULTS.glob("*"))), 1); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM attachments WHERE doc_id IS NOT NULL").fetchone()[0], 1)
    def test_service_returns_the_finished_record_for_a_reused_document_id_and_cleans_partial_leftovers(self):
        did = "20261001T100000_abcdef12"; d = SV.RESULTS / did; d.mkdir(parents=True); (d / "record.json").write_text(json.dumps({"doc_id": did, "finished_at": "t", "marker": 1}))
        with mock.patch.object(SV.BillService, "load", lambda self: self): r = SV.BillService().process(png(1), "a.png", "E1", {"doc_id": did})
        self.assertEqual(r["marker"], 1); self.assertEqual(len(list(SV.RESULTS.glob("*"))), 1)
        did2 = "20261001T100000_abcdef13"; d2 = SV.RESULTS / did2; d2.mkdir(); (d2 / "partial.txt").write_text("x"); stale = SV.UNCLASSIFIED / f"{did2}__a.png"; stale.parent.mkdir(parents=True, exist_ok=True); stale.write_bytes(b"x")
        with mock.patch.object(SV.BillService, "load", lambda self: self), mock.patch.object(SV.BillService, "_prepare_pages", side_effect=SV.UserError("stop")):
            with self.assertRaises(SV.UserError): SV.BillService().process(png(1), "a.png", "E1", {"doc_id": did2})
        self.assertFalse(stale.exists()); self.assertFalse(d2.exists())
        with self.assertRaises(SV.UserError): SV.BillService().process(png(1), "a.png", "E1", {"doc_id": "../../etc"})
    def test_processing_failure_after_retries_stays_visible_and_is_never_completed(self):
        PL.ingest_message(msg("<m1>", atts=[PNGA(1)])); proc = self.processor(fail=[RuntimeError("x")] * 5)
        for _ in range(8): db.conn().execute("UPDATE jobs SET next_run=0"); PL.step(proc, retry_delay=0)
        self.assertEqual(self.emails()[0]["status"], "FAILED"); self.assertEqual(db.conn().execute("SELECT error_code FROM attachments").fetchone()[0], "retries_exhausted")

# =============================================================== 3.3 webhook + fetch-by-id
class Webhook(Base):
    @classmethod
    def setUpClass(cls): cls.pki = STUB.make_pki()
    def deliver(self, eid, mid, secret="s"):
        body = json.dumps({"event_id": eid, "message_id": mid}).encode(); ts = str(int(time.time())); return WH.handle(body, {"X-Timestamp": ts, "X-Signature": WH.sign(secret, ts, body)}, secret)
    def test_repeated_deliveries_produce_one_job_and_one_email(self):
        fp = FakeProvider([msg("<w1>", atts=[PNGA(1)])]); self.assertEqual(self.deliver("e1", "<w1>")[0], 202); self.assertEqual(self.deliver("e1", "<w1>")[0], 200); self.assertEqual(self.deliver("e2", "<w1>")[0], 202)
        self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs WHERE key LIKE 'fetch:%'").fetchone()[0], 1); PL.drain(self.processor(), fp); self.assertEqual(len(self.emails()), 1); self.assertEqual(len(self.calls), 1)
    def test_job_is_done_only_after_durable_ingest_and_missing_message_ends_dead_lettered(self):
        fp = FakeProvider([]); self.deliver("e1", "<missing>"); outs = []
        for _ in range(6):
            db.conn().execute("UPDATE jobs SET next_run=0"); r = PL.step(self.processor(), fp, retry_delay=0); outs.append(r)
            if r == "fetch_retry": self.assertEqual(db.conn().execute("SELECT state FROM jobs").fetchone()[0], "queued")      # never 'done' while the message is missing
            if r is None: break
        self.assertIn("fetch_dead", outs); f = dict(db.conn().execute("SELECT * FROM inbox_failures").fetchone()); self.assertEqual((f["state"], f["error_code"]), ("dead", "message_not_found")); self.assertEqual(self.emails(), [])
        fp.add(msg("<missing>", atts=[PNGA(1)])); self.assertEqual(PL.requeue_failures("hr1"), 1); PL.drain(self.processor(), fp); self.assertEqual(len(self.emails()), 1)
    def test_transient_provider_failure_retries_then_succeeds(self):
        class Flaky(FakeProvider):
            n = 0
            def fetch_by_id(self, mid):
                Flaky.n += 1
                if Flaky.n < 3: raise PV.ProviderTransient("net")
                return super().fetch_by_id(mid)
        fp = Flaky([msg("<w1>", atts=[PNGA(1)])]); self.deliver("e1", "<w1>")
        for _ in range(4): db.conn().execute("UPDATE jobs SET next_run=0"); PL.step(self.processor(), fp, retry_delay=0)
        PL.drain(self.processor(), fp); self.assertEqual(len(self.emails()), 1); self.assertEqual(db.conn().execute("SELECT state FROM jobs WHERE key LIKE 'fetch:%'").fetchone()[0], "done")
    def test_permanent_provider_failure_is_dead_lettered_not_silently_dropped(self):
        class Bad(FakeProvider):
            def fetch_by_id(self, mid): raise PV.ProviderAuthError("no")
        self.deliver("e1", "<w1>"); fp = Bad([])
        for _ in range(6): db.conn().execute("UPDATE jobs SET next_run=0"); PL.step(self.processor(), fp, retry_delay=0)
        self.assertEqual(db.conn().execute("SELECT error_code,state FROM inbox_failures").fetchone()[:], ("credentials_rejected", "dead")); self.assertEqual(PL.health()["jobs_dead"], 1)
    def test_webhook_job_cannot_complete_without_a_provider(self):
        self.deliver("e1", "<w1>")
        for _ in range(5): db.conn().execute("UPDATE jobs SET next_run=0"); PL.step(self.processor(), None, retry_delay=0)
        self.assertEqual(db.conn().execute("SELECT state FROM jobs").fetchone()[0], "dead")
    def test_imap_fetch_by_id_against_local_server(self):
        os.environ["HRBC_IMAP_MARK_SEEN"] = "0"; s = STUB.StubServer(self.pki["good"], [mime("<a@x>", atts=[("b.png", png(1), "image", "png")]), mime("<b@x>")]); port = s.start(); self.addCleanup(s.stop)
        p = ImapProvider("localhost", "user", "pw", "INBOX", port, ssl_context=tls.context(self.pki["ca"]))
        with p.session():
            m = p.fetch_by_id("<a@x>"); self.assertEqual(m.message_id, "<a@x>"); self.assertEqual(len(m.attachments), 1); self.assertIsNone(p.fetch_by_id("<zzz@x>"))
            for bad in ('<a"b@x>', "<a\r\nb@x>", ""):
                with self.assertRaises(ValueError): p.fetch_by_id(bad)
    def test_webhook_flow_with_imap_provider_end_to_end(self):
        s = STUB.StubServer(self.pki["good"], [mime("<a@x>", atts=[("b.png", png(1), "image", "png")])]); port = s.start(); self.addCleanup(s.stop)
        p = ImapProvider("localhost", "user", "pw", "INBOX", port, ssl_context=tls.context(self.pki["ca"])); self.deliver("e1", "<a@x>"); PL.drain(self.processor(), p)
        self.assertEqual(self.emails()[0]["status"], "COMPLETED"); self.assertEqual(len(self.calls), 1)

# =============================================================== IMAP cursor flow with the stub
class ImapCursor(Base):
    @classmethod
    def setUpClass(cls): cls.pki = STUB.make_pki()
    def test_end_to_end_poll_cursor_persisted_poison_isolated_no_mailbox_changes(self):
        s = STUB.StubServer(self.pki["good"], [mime("<m1@x>", atts=[("p.pdf", pdf(1), "application", "pdf")]), mime("<m2@x>", atts=[("p.png", png(2), "image", "png")]), mime("<m3@x>", sender="stranger@evil.example", atts=[("p.png", png(3), "image", "png")])]); port = s.start(); self.addCleanup(s.stop)
        p = ImapProvider("localhost", "user", "pw", "INBOX", port, ssl_context=tls.context(self.pki["ca"]))
        with mock.patch.object(VA.subprocess, "run", side_effect=no_pdfinfo()): st = self.poll(p, now=10)
        self.assertEqual((st["ingested"], st["failed"], st["rejected"]), (1, 1, 1)); self.assertEqual(db.get_cursor("imap", p.mailbox_key())["last_uid"], 3)
        s.add(mime("<m4@x>", atts=[("q.png", png(4), "image", "png")])); st = self.poll(p, now=20); self.assertEqual(st["ingested"], 1)                  # only the new message; the failed one waits for its backoff
        self.assertEqual(sorted(e["message_id"] for e in self.emails() if e["status"] != "REJECTED"), ["<m2@x>", "<m4@x>"]); self.assertEqual(s.stored, [])
        st = self.poll(p, now=10_000); self.assertEqual(st["recovered"], 1); self.assertEqual(len([e for e in self.emails() if e["status"] != "REJECTED"]), 3)
    def test_mark_seen_is_explicit_opt_in(self):
        s = STUB.StubServer(self.pki["good"], [mime("<m1@x>", atts=[("p.png", png(1), "image", "png")])]); port = s.start(); self.addCleanup(s.stop)
        p = ImapProvider("localhost", "user", "pw", "INBOX", port, ssl_context=tls.context(self.pki["ca"]), mark_seen=True); self.poll(p); self.assertIn("SELECT", s.log); self.assertEqual(s.stored, ["1"])

# =============================================================== 4 bill content outside attachments, unsafe formats
class BodyAndFormats(Base):
    INVOICE_HTML = "<html><body><h1>Tax Invoice</h1><p>Total Rs 1,250.00 paid. GSTIN 29ABCDE1234F1Z5. Order id 884.</p></body></html>"
    def ingest_raw(self, m): return PL.ingest_message(parse_rfc822(m.as_bytes()))
    def test_html_body_only_bill_is_flagged_for_review_never_completed(self):
        r = self.ingest_raw(mime("<b1>", body="see html", html=self.INVOICE_HTML)); e = self.emails()[0]; self.assertEqual((r["status"], e["status"], e["reason"]), ("REVIEW_REQUIRED", "REVIEW_REQUIRED", "possible_bill_in_body")); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
    def test_plain_text_body_bill_and_forwarded_body_bill_flagged(self):
        self.assertEqual(self.ingest_raw(mime("<b2>", body="Receipt: total Rs 450.00 paid, order id 12, payment received"))["status"], "REVIEW_REQUIRED")
        outer = mime("<b3>", body="fwd"); outer.add_attachment(mime("<in>", sender="shop@x.example", body="Tax invoice total Rs 99.00 amount paid order no 7")); self.assertEqual(self.ingest_raw(outer)["status"], "REVIEW_REQUIRED")
    def test_message_without_any_bill_is_no_document_not_completed(self):
        r = self.ingest_raw(mime("<b4>", body="Lunch tomorrow?")); self.assertEqual((r["status"], self.emails()[0]["reason"]), ("NO_DOCUMENT", "no_supported_attachment")); self.assertNotEqual(self.emails()[0]["status"], "COMPLETED")
    def test_inline_logo_only_with_bill_body_is_flagged(self):
        m = msg("<b5>", atts=[Attachment("logo.png", "image/png", png(9)[:1200], True)], body_signal=dict(candidate=True, cues=["invoice"], amount_mentions=1)); self.assertEqual(PL.ingest_message(m)["status"], "REVIEW_REQUIRED")
    def test_body_text_is_not_persisted_only_signal_labels(self):
        self.ingest_raw(mime("<b6>", html=self.INVOICE_HTML)); bs = json.loads(self.emails()[0]["body_signal"]); self.assertEqual(set(bs), {"candidate", "cues", "amount_mentions", "has_html", "chars"}); self.assertNotIn("29ABCDE1234F1Z5", json.dumps(dict(self.emails()[0])))
    def test_attachment_beats_body_and_multiple_attachments_are_independent(self):
        r = self.ingest_raw(mime("<b7>", html=self.INVOICE_HTML, atts=[("a.png", png(1), "image", "png"), ("bad.zip", b"PK\x03\x04" + b"0" * 2000, "application", "zip"), ("c.png", png(2), "image", "png")]))
        self.assertEqual([(a["idx"], a["state"]) for a in r["attachments"]], [(0, "QUEUED"), (1, "FAILED"), (2, "QUEUED")]); PL.drain(self.processor()); self.assertEqual(self.emails()[0]["status"], "REVIEW_REQUIRED"); self.assertEqual(len(self.calls), 2)
    def test_unsafe_corrupt_empty_and_archive_attachments_rejected_with_codes(self):
        bomb = b"PK\x03\x04" + b"\x00" * 5000; atts = [("a.zip", "application/zip", bomb), ("b.exe", "application/octet-stream", b"MZ" + b"0" * 3000), ("c.pdf", "application/pdf", b"%PDF-1.4 garbage" + b"0" * 2000), ("d.png", "image/png", b""), ("e.png", "image/png", b"\x89PNG\r\n\x1a\n" + b"0" * 3000)]
        r = PL.ingest_message(msg("<u1>", atts=atts)); codes = [a["error_code"] for a in r["attachments"]]; self.assertEqual(codes, ["unsupported_type", "unsupported_type", "corrupt", "empty", "corrupt"]); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
    def test_attachment_count_and_total_size_limits(self):
        atts = [PNGA(i, f"{i}.png") for i in range(VA.MAX_ATTACHMENTS + 2)]; r = PL.ingest_message(msg("<l1>", atts=atts)); self.assertEqual([a["error_code"] for a in r["attachments"]].count("too_many_attachments"), 2)

# =============================================================== 2.3 permissions + tenancy
class Permissions(Base):
    def mode(self, p): return stat.S_IMODE(Path(p).stat().st_mode)
    def test_databases_inbox_and_directories_are_private(self):
        PL.ingest_message(msg("<p1>", atts=[PNGA(1)])); from tax import store; store.conn().close(); o = Path(self.out)
        for rel in ("hr/ingest.db", "hr/hr.db"): self.assertEqual(self.mode(o / rel), 0o600, rel)
        for f in (o / "hr").glob("ingest.db-*"): self.assertEqual(self.mode(f), 0o600)
        for rel in ("", "hr", "ingest", "ingest/inbox"): self.assertEqual(self.mode(o / rel), 0o700, rel)
        for f in (o / "ingest" / "inbox").glob("*"): self.assertEqual(self.mode(f), 0o600)
        self.assertEqual(perms.audit(o), [])
    def test_audit_detects_world_readable_data_and_fix_repairs_it(self):
        o = Path(self.out); (o / "results" / "d1").mkdir(parents=True); f = o / "results" / "d1" / "record.json"; f.write_text("{}"); os.chmod(f, 0o644); os.chmod(f.parent, 0o755)
        bad = perms.audit(o); self.assertTrue(any(b[0].endswith("record.json") for b in bad)); self.assertEqual(RUN.main(["check-perms"]), 1); self.assertEqual(RUN.main(["check-perms", "--fix"]), 0); self.assertEqual(perms.audit(o), [])
    def test_service_sets_a_private_umask_so_results_are_not_world_readable(self):
        old = os.umask(0o022)
        try: SV.BillService(); self.assertEqual(os.umask(0o077), 0o077)
        finally: os.umask(old)
    def test_secret_files_must_be_private(self):
        f = Path(self.out) / "pw"; f.write_text("x"); os.chmod(f, 0o644); os.environ["HRBC_IMAP_PASSWORD_FILE"] = str(f)
        with self.assertRaises(config.ConfigError): config.secret("HRBC_IMAP_PASSWORD")
        os.chmod(f, 0o600); self.assertEqual(config.secret("HRBC_IMAP_PASSWORD"), "x")
    def test_stored_documents_are_only_visible_to_their_owner_and_hr(self):
        PL.ingest_message(msg("<p2>", atts=[PNGA(1)])); PL.drain(self.processor()); E1, E2, HR = api.Principal("employee", "E1"), api.Principal("employee", "E2"), api.Principal("hr", "H1")
        d = api.list_documents(E1)[0]["doc_id"]; self.assertEqual(api.list_documents(E2), []); self.assertTrue(api.get_document(HR, d)); self.assertTrue(api.get_document(E1, d))
        with self.assertRaises(PermissionDenied): api.get_document(E2, d)
        with self.assertRaises(PermissionDenied): api.email_detail(E2, 1)
        self.assertEqual(api.list_emails(E2), [])

# =============================================================== 7 configuration, worker, health
class Operation(Base):
    def setUp(self):
        super().setUp(); self.args = type("A", (), dict(imap=False, interval=1, once=True))()
    def test_worker_refuses_to_start_on_invalid_configuration(self):
        os.environ.pop("HRBC_INGEST_MAILBOX"); self.assertEqual(RUN.run_worker(self.args, svc=object()), 2)
    def test_check_config_exit_codes_and_no_values_printed(self):
        os.environ.update(HRBC_IMAP_HOST="h", HRBC_IMAP_USER="u", HRBC_IMAP_PASSWORD="topsecret-value", HRBC_TEST_MAILBOX_USER="other", HRBC_TEST_FOLDER="INBOX")
        import contextlib; buf = io.StringIO()
        with contextlib.redirect_stdout(buf): rc = RUN.main(["check-config"])
        self.assertEqual(rc, 2); self.assertNotIn("topsecret-value", buf.getvalue()); self.assertIn("HRBC_TEST_MAILBOX_USER", buf.getvalue())
    def test_test_mailbox_guard_and_prod_isolation(self):
        os.environ.update(HRBC_IMAP_HOST="h", HRBC_IMAP_USER="boss@corp.example", HRBC_IMAP_PASSWORD="p", HRBC_IMAP_FOLDER="INBOX"); self.assertTrue(any("HRBC_TEST_MAILBOX_USER" in p for p in config.validate()))
        os.environ.update(HRBC_TEST_MAILBOX_USER="boss@corp.example", HRBC_TEST_FOLDER="INBOX"); self.assertEqual(config.validate(), [])      # explicitly declared as THE test mailbox
        os.environ["HRBC_IMAP_MARK_SEEN"] = "1"; self.assertTrue(any("MARK_SEEN" in p for p in config.validate()))
        os.environ.pop("HRBC_IMAP_MARK_SEEN"); os.environ["HRBC_ENV"] = "prod"; self.assertTrue(any("HRBC_TEST_" in p for p in config.validate()))
        os.environ["HRBC_ENV"] = "test"; os.environ.pop("HRBC_OUT"); self.assertTrue(any("HRBC_OUT" in p for p in config.validate()))
    def test_config_validation_covers_xoauth2_and_https(self):
        os.environ.update(HRBC_ENV="prod", HRBC_IMAP_HOST="h", HRBC_IMAP_USER="u", HRBC_IMAP_AUTH="xoauth2", HRBC_OAUTH_TOKEN_URL="http://idp.example/token"); os.environ.pop("HRBC_OUT", None)
        p = config.validate(); self.assertTrue(any("https" in x for x in p)); self.assertTrue(any("CLIENT_ID" in x for x in p)); os.environ["HRBC_IMAP_AUTH"] = "plain"; self.assertTrue(any("HRBC_IMAP_AUTH" in x for x in config.validate()))
    def test_single_worker_lock(self):
        fd = RUN.acquire_lock(); self.assertIsNotNone(fd); self.assertEqual(RUN.run_worker(self.args, svc=object()), 3); os.close(fd)
    def test_worker_processes_jobs_and_shuts_down_gracefully(self):
        PL.ingest_message(msg("<w1>", atts=[PNGA(1)])); svc = type("S", (), {"process": staticmethod(self.processor())})(); stop = {"v": False}
        def flag(): return stop["v"]
        args = type("A", (), dict(imap=False, interval=1, once=False))(); calls = []
        orig = PL.drain
        def drain_then_stop(*a, **k): n = orig(*a, **k); stop["v"] = True; calls.append(n); return n
        with mock.patch.object(PL, "drain", drain_then_stop): rc = RUN.run_worker(args, svc=svc, stop_flag=flag)
        self.assertEqual(rc, 0); self.assertEqual(calls, [1]); ws = dict(db.conn().execute("SELECT * FROM worker_state").fetchone()); self.assertTrue(ws["stopped_at"]); self.assertEqual(self.emails()[0]["status"], "COMPLETED")
        self.assertIsNotNone(RUN.acquire_lock())                                                                       # lock released on exit
    def test_worker_loop_survives_provider_errors_and_keeps_heartbeat(self):
        class Boom(FakeProvider):
            def list_new(self, c, i="x"): raise PV.ProviderTransient("net")
        svc = type("S", (), {"process": staticmethod(self.processor())})(); stop = {"n": 0}
        def flag(): stop["n"] += 1; return stop["n"] > 3
        args = type("A", (), dict(imap=True, interval=1, once=False))()
        with mock.patch.object(config, "validate", lambda imap=True: []):
            self.assertEqual(RUN.run_worker(args, svc=svc, prov=Boom([]), stop_flag=flag), 0)
        h = PL.health(); self.assertEqual(h["last_poll_error"], "provider_transient"); self.assertIn("poll_error:provider_transient", h["problems"])
    def test_health_statuses(self):
        h = PL.health(); self.assertEqual(h["status"], "down"); self.assertIn("worker_not_running", h["problems"])
        c = db.conn(); PL.worker_state(c); c.execute("UPDATE worker_state SET last_loop_at=?, last_poll_ok_at=? WHERE id=1", (time.time(), time.time())); self.assertEqual(PL.health()["status"], "ok")
        c.execute("UPDATE worker_state SET last_loop_at=? WHERE id=1", (time.time() - 3600,)); self.assertEqual(PL.health()["status"], "down")
        c.execute("UPDATE worker_state SET last_loop_at=? WHERE id=1", (time.time(),)); PL.requeue_failures("x"); c.execute("INSERT INTO inbox_failures(key,state,error_code) VALUES('k','dead','x')"); self.assertEqual(PL.health()["status"], "degraded")
        out = json.dumps(PL.health()); self.assertNotIn("@", out)
    def test_logs_contain_codes_only(self):
        recs = []; h = logging.Handler(); h.emit = lambda r: recs.append(r.getMessage()); lg = logging.getLogger("ingest"); lg.addHandler(h); lg.setLevel(logging.INFO); self.addCleanup(lambda: lg.removeHandler(h))
        PL.ingest_message(msg("<g1>", sender="asha@corp.example", atts=[PNGA(1)], rcpt="expenses@corp.example")); PL.ingest_message(msg("<g2>", sender="stranger@evil.example"))
        text = " ".join(recs); self.assertNotIn("stranger@evil.example", text); self.assertNotIn("asha@corp.example", text); self.assertNotIn("my bills", text)

if __name__ == "__main__": unittest.main()
