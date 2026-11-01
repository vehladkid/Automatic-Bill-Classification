"""One-time interactive Google consent for the mailbox worker (OAuth 2.0 authorization-code flow for installed apps with PKCE and a loopback redirect, RFC 7636 / RFC 8252).
Default transport: Gmail API (app/ingest/gmail_api.py) with the least-privilege read-only scope gmail.readonly. The IMAP transport needs the much broader https://mail.google.com/ (--imap-scope); gmail.readonly is not accepted by IMAP.
Secrets (client secret, authorization code, tokens) are never printed or logged; the refresh token is written to a 0600 file outside the repository."""
import base64, hashlib, json, os, re, secrets, threading, time, urllib.error, urllib.parse, urllib.request, webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from . import config
SCOPE = "https://www.googleapis.com/auth/gmail.readonly"            # Gmail API (default transport of this project)
IMAP_SCOPE = "https://mail.google.com/"                             # only if HRBC_IMAP_AUTH=xoauth2 against imap.gmail.com is used instead
class AuthError(Exception):
    """message is safe to show (no secrets)"""

def load_client(path):
    """-> dict(client_id, client_secret, auth_uri, token_uri) from the downloaded Google client JSON ('installed' or 'web'). Requires private permissions. Never echoes contents."""
    p = Path(path).expanduser()
    if not p.is_file(): raise config.ConfigError("OAuth client file not found")
    if p.stat().st_mode & 0o077: raise config.ConfigError("OAuth client file must not be readable by group/other (chmod 600)")
    try: d = json.loads(p.read_text())
    except Exception: raise config.ConfigError("OAuth client file is not valid JSON")
    c = d.get("installed") or d.get("web") or {}
    miss = [k for k in ("client_id", "client_secret", "auth_uri", "token_uri") if not c.get(k)]
    if miss: raise config.ConfigError("OAuth client file is missing: " + ", ".join(miss))
    for k in ("auth_uri", "token_uri"): _https(c[k])
    return {k: c[k] for k in ("client_id", "client_secret", "auth_uri", "token_uri")}

def _https(url):
    if not (url.startswith("https://") or (config.env_name() != "prod" and re.match(r"http://(127\.0\.0\.1|localhost)[:/]", url))): raise config.ConfigError("OAuth endpoints must be https")

def pkce():
    v = secrets.token_urlsafe(64); return v, base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()

def auth_url(client, redirect, state, challenge, scope=SCOPE, login_hint=None):
    q = dict(client_id=client["client_id"], redirect_uri=redirect, response_type="code", scope=scope, state=state, code_challenge=challenge, code_challenge_method="S256", access_type="offline", prompt="consent")
    if login_hint: q["login_hint"] = login_hint
    return client["auth_uri"] + ("&" if "?" in client["auth_uri"] else "?") + urllib.parse.urlencode(q)

def _wait_for_redirect(srv, timeout):
    got = {}
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            got.update({k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()})
            ok = "code" in got and "error" not in got
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(("<h3>Authorization received. You can close this tab and return to the terminal.</h3>" if ok else "<h3>Authorization was not completed. Return to the terminal.</h3>").encode())
        def log_message(self, *a): pass
    srv.RequestHandlerClass = H; srv.timeout = timeout; t0 = time.time()
    while not got and time.time() - t0 < timeout: srv.handle_request()
    return got

def exchange(client, code, redirect, verifier, timeout=20):
    body = urllib.parse.urlencode(dict(grant_type="authorization_code", code=code, client_id=client["client_id"], client_secret=client["client_secret"], redirect_uri=redirect, code_verifier=verifier)).encode()
    req = urllib.request.Request(client["token_uri"], data=body, headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try: code_ = json.loads(e.read().decode()).get("error", "")
        except Exception: code_ = ""
        raise AuthError(f"Google rejected the authorization code ({code_ or 'http_' + str(e.code)}). Start again.")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError): raise AuthError("Could not reach the Google token endpoint (network problem).")

def write_private(path, text):
    p = Path(path).expanduser(); p.parent.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(p.parent, 0o700)
    tmp = p.with_name(p.name + ".tmp"); fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try: os.write(fd, text.encode())
    finally: os.close(fd)
    os.replace(tmp, p); os.chmod(p, 0o600)

def _run_flow(client_file, scope, login_hint, open_browser, timeout, out, opener):
    """consent flow shared by authorize() and authorize_tokens(): PKCE, loopback redirect, state check, scope check, refresh token required. Returns the token response; writes nothing."""
    client = load_client(client_file); verifier, challenge = pkce(); state = secrets.token_urlsafe(24)
    srv = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler); redirect = f"http://127.0.0.1:{srv.server_port}"
    url = auth_url(client, redirect, state, challenge, scope, login_hint)
    out("Open this address in a browser, sign in to the DEDICATED test Gmail account and approve access:\n" + url)
    if open_browser:
        try: opener(url)
        except Exception: pass
    out(f"Waiting up to {timeout}s on {redirect} ...")
    try: got = _wait_for_redirect(srv, timeout)
    finally: srv.server_close()
    if not got: raise AuthError("Timed out waiting for the browser authorization.")
    if got.get("error"): raise AuthError(f"Authorization was not granted ({re.sub(r'[^a-z_]', '', got['error'])[:40]}).")
    if got.get("state") != state: raise AuthError("State mismatch: the response did not come from this authorization request. Nothing was stored.")
    tok = exchange(client, got.get("code", ""), redirect, verifier)
    granted = set(str(tok.get("scope", "")).split())
    if scope not in granted: raise AuthError(f"The granted scopes do not include {scope}. Tick the Gmail permission on the consent screen (and add it under Data Access / Scopes in Google Cloud). Nothing was stored.")
    if not tok.get("refresh_token"): raise AuthError("Google returned no refresh token. Remove this app under myaccount.google.com/permissions and run the authorization again.")
    return tok

def authorize_tokens(client_file, login_hint=None, scope=SCOPE, open_browser=True, timeout=300, out=print, opener=webbrowser.open):
    """same flow, for connecting an ADDITIONAL mailbox: returns the token response (the caller stores the refresh token per account; nothing is written or overwritten here)."""
    return _run_flow(client_file, scope, login_hint, open_browser, timeout, out, opener)

def authorize(client_file, token_file, login_hint=None, scope=SCOPE, open_browser=True, timeout=300, out=print, opener=webbrowser.open):
    tok = _run_flow(client_file, scope, login_hint, open_browser, timeout, out, opener)
    write_private(token_file, tok["refresh_token"]); out(f"Refresh token stored (mode 600) at {Path(token_file).expanduser()}")
    return str(Path(token_file).expanduser())
