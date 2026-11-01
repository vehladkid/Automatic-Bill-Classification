"""OAuth2 refresh-token flow (RFC 6749 section 6) and the XOAUTH2 SASL string for IMAP. Implemented and unit-tested against a LOCAL token endpoint only; NOT verified against Google or Microsoft.
Provider facts you must supply (and check against current provider docs): token endpoint URL, client id/secret, a refresh token obtained once through the provider's interactive consent (Google: `python -m ingest.run google-auth`, see google_auth.py), and the IMAP scope
(Google: https://mail.google.com/ ; Microsoft: https://outlook.office.com/IMAP.AccessAsUser.All offline_access). Secrets are never logged or put in exception text."""
import json, time, urllib.request, urllib.parse, urllib.error, os, re
from pathlib import Path
from . import config
class CredentialsExpired(Exception):
    """refresh token revoked/expired, client rejected: permanent until an operator re-authorises"""
class TransientAuthError(Exception):
    def __init__(self, code, retry_after=None): super().__init__(code); self.code = code; self.retry_after = retry_after
PERMANENT = {"invalid_grant", "invalid_client", "unauthorized_client", "invalid_scope", "access_denied"}
class RefreshTokenSource:
    def __init__(self, token_url, client_id, client_secret, refresh_token, scope=None, timeout=15, clock=time.time, persist_path=None, skew=60):
        if not token_url.startswith("https://") and not (config.env_name() != "prod" and re.match(r"http://(127\.0\.0\.1|localhost)[:/]", token_url)): raise config.ConfigError("token URL must be https")
        self.url, self.cid, self.secret, self._refresh, self.scope, self.timeout, self.clock, self.persist, self.skew = token_url, client_id, client_secret, refresh_token, scope, timeout, clock, persist_path, skew
        self._tok = None; self._exp = 0.0; self.refreshes = 0
    @classmethod
    def from_env(cls):
        cf = os.environ.get("HRBC_OAUTH_CLIENT_FILE")          # downloaded Google client JSON (private file outside the repo): supplies client id/secret and token endpoint
        if cf:
            from . import google_auth; c = google_auth.load_client(cf)
            return cls(os.environ.get("HRBC_OAUTH_TOKEN_URL") or c["token_uri"], c["client_id"], c["client_secret"], config.secret("HRBC_OAUTH_REFRESH_TOKEN"), os.environ.get("HRBC_OAUTH_SCOPE"), persist_path=os.environ.get("HRBC_OAUTH_REFRESH_TOKEN_STORE"))
        return cls(os.environ["HRBC_OAUTH_TOKEN_URL"], config.secret("HRBC_OAUTH_CLIENT_ID"), config.secret("HRBC_OAUTH_CLIENT_SECRET"), config.secret("HRBC_OAUTH_REFRESH_TOKEN"), os.environ.get("HRBC_OAUTH_SCOPE"), persist_path=os.environ.get("HRBC_OAUTH_REFRESH_TOKEN_STORE"))
    def access_token(self):
        if self._tok and self.clock() < self._exp - self.skew: return self._tok
        return self._refresh_now()
    def _refresh_now(self):
        body = {"grant_type": "refresh_token", "refresh_token": self._refresh, "client_id": self.cid, "client_secret": self.secret}
        if self.scope: body["scope"] = self.scope
        req = urllib.request.Request(self.url, data=urllib.parse.urlencode(body).encode(), headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r: data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try: code = json.loads(e.read().decode()).get("error", "")
            except Exception: code = ""
            if code in PERMANENT or (e.code in (400, 401) and not code): raise CredentialsExpired(code or f"http_{e.code}")
            ra = e.headers.get("Retry-After") if e.headers else None
            raise TransientAuthError(code or f"http_{e.code}", float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError): raise TransientAuthError("token_endpoint_unreachable")
        tok = data.get("access_token")
        if not tok: raise TransientAuthError("no_access_token_in_response")
        self._tok = tok; self._exp = self.clock() + float(data.get("expires_in", 3600)); self.refreshes += 1
        if data.get("refresh_token") and data["refresh_token"] != self._refresh:     # rotating refresh tokens (e.g. Microsoft): keep the new one, persist it privately
            self._refresh = data["refresh_token"]
            if self.persist:
                p = Path(self.persist); p.parent.mkdir(parents=True, exist_ok=True); fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600); os.write(fd, self._refresh.encode()); os.close(fd)
        return self._tok
def xoauth2_string(user, token): return f"user={user}\x01auth=Bearer {token}\x01\x01".encode()
