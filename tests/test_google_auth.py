"""Google consent flow against a LOCAL fake authorization/token server. No network, no real credentials."""
import unittest, json, os, stat, tempfile, threading, urllib.parse, urllib.request, io, contextlib, _path
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from ingest import google_auth as G, oauth, config

class Fake:
    """token endpoint: returns what `reply` says"""
    def __init__(self, reply, status=200): self.reply, self.status, self.seen = reply, status, []
    def start(self):
        outer = self
        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                outer.seen.append(urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())); b = json.dumps(outer.reply).encode()
                self.send_response(outer.status); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
            def log_message(self, *a): pass
        self.srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=self.srv.serve_forever, daemon=True).start(); return f"http://127.0.0.1:{self.srv.server_port}"
    def stop(self): self.srv.shutdown(); self.srv.server_close()

class T(unittest.TestCase):
    def setUp(self):
        os.environ["HRBC_ENV"] = "test"; self.d = Path(tempfile.mkdtemp()); self.addCleanup(os.environ.pop, "HRBC_ENV", None)
    def client(self, tok_url, mode=0o600):
        p = self.d / "client.json"; p.write_text(json.dumps({"installed": dict(client_id="cid", client_secret="SECRET-VALUE", auth_uri="https://accounts.example/auth", token_uri=tok_url)})); os.chmod(p, mode); return p
    def run_flow(self, reply, mutate=None, status=200, scope_ok=True):
        fk = Fake(reply, status); url = fk.start(); self.addCleanup(fk.stop); cf = self.client(url + "/token"); lines = []
        def browser(u):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(u).query); self.q = q; red = q["redirect_uri"][0]
            params = dict(code="AUTHCODE", state=q["state"][0]); params = mutate(params) if mutate else params
            threading.Thread(target=lambda: urllib.request.urlopen(red + "/?" + urllib.parse.urlencode(params)).read(), daemon=True).start()
        tf = self.d / "sub" / "refresh"
        try: r = G.authorize(cf, tf, login_hint="me@gmail.com", timeout=10, out=lines.append, opener=browser)
        except Exception as e: r = e
        return r, tf, fk, "\n".join(lines)
    def test_success_stores_private_token_and_never_prints_secrets(self):
        r, tf, fk, out = self.run_flow({"access_token": "AT", "refresh_token": "RT-SECRET", "scope": G.SCOPE, "expires_in": 3600})
        self.assertEqual(r, str(tf)); self.assertEqual(tf.read_text(), "RT-SECRET"); self.assertEqual(stat.S_IMODE(tf.stat().st_mode), 0o600); self.assertEqual(stat.S_IMODE(tf.parent.stat().st_mode), 0o700)
        for s in ("RT-SECRET", "SECRET-VALUE", "AUTHCODE"): self.assertNotIn(s, out)
        q = self.q; self.assertEqual(q["scope"], [G.SCOPE]); self.assertEqual(q["code_challenge_method"], ["S256"]); self.assertEqual(q["access_type"], ["offline"]); self.assertEqual(q["login_hint"], ["me@gmail.com"])
        sent = fk.seen[0]; self.assertEqual(sent["grant_type"], ["authorization_code"]); self.assertTrue(sent["code_verifier"][0]); self.assertEqual(sent["code"], ["AUTHCODE"])
    def test_state_mismatch_rejected_nothing_stored_nothing_sent(self):
        r, tf, fk, _ = self.run_flow({}, mutate=lambda p: dict(p, state="forged")); self.assertIsInstance(r, G.AuthError); self.assertIn("State mismatch", str(r)); self.assertFalse(tf.exists()); self.assertEqual(fk.seen, [])
    def test_user_denied(self):
        r, tf, fk, _ = self.run_flow({}, mutate=lambda p: {"error": "access_denied", "state": p["state"]}); self.assertIsInstance(r, G.AuthError); self.assertIn("access_denied", str(r)); self.assertFalse(tf.exists())
    def test_missing_scope_or_refresh_token_not_stored(self):
        r, tf, _, _ = self.run_flow({"access_token": "A", "refresh_token": "R", "scope": G.IMAP_SCOPE}); self.assertIn("granted scopes", str(r)); self.assertFalse(tf.exists())
        r, tf, _, _ = self.run_flow({"access_token": "A", "scope": G.SCOPE}); self.assertIn("no refresh token", str(r)); self.assertFalse(tf.exists())
    def test_token_endpoint_error_is_readable_and_secret_free(self):
        r, tf, _, _ = self.run_flow({"error": "invalid_grant"}, status=400); self.assertIsInstance(r, G.AuthError); self.assertIn("invalid_grant", str(r)); self.assertNotIn("SECRET", str(r)); self.assertFalse(tf.exists())
    def test_client_file_must_be_private_and_valid(self):
        with self.assertRaises(config.ConfigError): G.load_client(self.client("http://127.0.0.1:1/t", 0o644))
        (self.d / "bad.json").write_text("{}"); os.chmod(self.d / "bad.json", 0o600)
        with self.assertRaises(config.ConfigError): G.load_client(self.d / "bad.json")
        with self.assertRaises(config.ConfigError): G.load_client(self.d / "nope.json")
    def test_prod_requires_https_endpoints(self):
        os.environ["HRBC_ENV"] = "prod"
        with self.assertRaises(config.ConfigError): G.load_client(self.client("http://127.0.0.1:1/t"))
    def test_refresh_source_and_config_use_client_file(self):
        fk = Fake({"access_token": "ACC", "expires_in": 3600}); url = fk.start(); self.addCleanup(fk.stop); cf = self.client(url + "/token"); rt = self.d / "rt"; rt.write_text("RT"); os.chmod(rt, 0o600)
        env = dict(HRBC_OAUTH_CLIENT_FILE=str(cf), HRBC_OAUTH_REFRESH_TOKEN_FILE=str(rt)); os.environ.update(env); self.addCleanup(lambda: [os.environ.pop(k, None) for k in env])
        self.assertEqual(oauth.RefreshTokenSource.from_env().access_token(), "ACC"); body = fk.seen[0]; self.assertEqual(body["grant_type"], ["refresh_token"]); self.assertEqual(body["refresh_token"], ["RT"])
        os.environ.pop("HRBC_OAUTH_REFRESH_TOKEN_FILE"); os.environ.update(HRBC_INGEST_MAILBOX="a@b.co", HRBC_INGEST_AUTHSERV_ID="mx", HRBC_OUT=str(self.d), HRBC_IMAP_HOST="h", HRBC_IMAP_USER="u", HRBC_IMAP_AUTH="xoauth2", HRBC_TEST_MAILBOX_USER="u", HRBC_TEST_FOLDER="INBOX")
        self.addCleanup(lambda: [os.environ.pop(k, None) for k in ("HRBC_INGEST_MAILBOX", "HRBC_INGEST_AUTHSERV_ID", "HRBC_OUT", "HRBC_IMAP_HOST", "HRBC_IMAP_USER", "HRBC_IMAP_AUTH", "HRBC_TEST_MAILBOX_USER", "HRBC_TEST_FOLDER")])
        p = config.validate(); self.assertTrue(any("google-auth" in x for x in p)); self.assertFalse(any("SECRET" in x for x in p))
        os.environ["HRBC_OAUTH_REFRESH_TOKEN_FILE"] = str(rt); self.assertEqual(config.validate(), [])
    def test_revoked_token_is_credentials_expired(self):
        fk = Fake({"error": "invalid_grant"}, 400); url = fk.start(); self.addCleanup(fk.stop)
        with self.assertRaises(oauth.CredentialsExpired): oauth.RefreshTokenSource(url + "/t", "c", "s", "r").access_token()
if __name__ == "__main__": unittest.main()
