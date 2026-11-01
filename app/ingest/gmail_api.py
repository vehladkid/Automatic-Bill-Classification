"""Gmail API adapter (REST, read-only) for the existing ingestion pipeline. Scope: https://www.googleapis.com/auth/gmail.readonly ONLY (no send/modify/delete code exists here).
Same provider contract as ImapProvider (list_new / fetch_headers / fetch_message / fetch_by_id); raw messages go through the SAME parser (provider.parse_rfc822) as IMAP, so sender checks,
attachment validation, de-duplication and the processing service are unchanged. Cursor: Gmail message ids are hex numbers that increase with arrival; the cursor is the highest id seen (uidvalidity fixed 'gmail').
Limitation: a message that Gmail inserts with a LOWER id than the cursor (e.g. imported/back-dated mail) is not seen. Only INBOX is read, at most 3 pages of 100 ids per poll.
Nothing here logs tokens, message bodies or attachment contents. Verified against a local fake Gmail server in tests; see docs/EMAIL_LIVE_DEMO.md for the live status."""
import base64, json, re, socket, ssl, urllib.error, urllib.parse, urllib.request
from . import config, oauth
from .provider import (MailProvider, Ref, Message, parse_headers, parse_rfc822, ProviderError, ProviderAuthError, ProviderTransient, RateLimited, MessageNotFound, TlsVerificationError, MAX_MESSAGE_BYTES)
SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
API = "https://gmail.googleapis.com/gmail/v1/users/me"
HEADERS = ["From", "To", "Cc", "Delivered-To", "X-Original-To", "Envelope-To", "Subject", "Date", "Message-ID", "Authentication-Results"]

class GmailApiProvider(MailProvider):
    name = "gmail"
    def __init__(self, token_source=None, base=None, timeout=30, batch=None, pages=3, key=None):
        base = base or API
        if not (base.startswith("https://") or (config.env_name() != "prod" and re.match(r"http://(127\.0\.0\.1|localhost)[:/]", base))): raise config.ConfigError("Gmail API base URL must be https")
        self.base, self.timeout, self.pages = base.rstrip("/"), timeout, pages; self.batch = batch or config.imap_settings()["batch"]
        self.tokens = token_source or oauth.RefreshTokenSource.from_env(); self._key = key
    def mailbox_key(self): return self._key or "gmail:me"      # per-account cursor key for connected accounts; the legacy mailbox keeps 'gmail:me'
    # transport -----------------------------------------------------------
    def _get(self, path, params=None):
        try: tok = self.tokens.access_token()
        except oauth.CredentialsExpired: raise ProviderAuthError("oauth credentials expired or revoked")
        except oauth.TransientAuthError as e: raise (RateLimited(e.retry_after) if e.retry_after else ProviderTransient("token_endpoint"))
        url = self.base + path + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r: return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try: reason = (json.loads(e.read().decode()).get("error") or {}).get("errors", [{}])[0].get("reason", "")
            except Exception: reason = ""
            if e.code == 404: raise MessageNotFound("not found")
            if e.code == 429 or reason in ("rateLimitExceeded", "userRateLimitExceeded"):
                ra = e.headers.get("Retry-After") if e.headers else None; raise RateLimited(float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
            if e.code in (401, 403): raise ProviderAuthError("gmail api rejected the credentials" + (f" ({reason})" if reason in ("accessNotConfigured", "insufficientPermissions", "authError") else ""))
            raise ProviderTransient(f"http_{e.code}")
        except ssl.SSLCertVerificationError: raise TlsVerificationError("certificate not trusted or hostname mismatch")
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, ValueError):
            raise ProviderTransient("connection_lost")
    # reading -------------------------------------------------------------
    def _ids(self, q=None, pages=None):
        out, tok = [], None
        for _ in range(pages or self.pages):
            p = {"labelIds": "INBOX", "maxResults": 100}
            if q: p["q"] = q
            if tok: p["pageToken"] = tok
            d = self._get("/messages", p); out += [int(m["id"], 16) for m in d.get("messages", [])]; tok = d.get("nextPageToken")
            if not tok: break
        return out
    def list_new(self, cursor, initial="new_only"):
        ids = self._ids(); top = max(ids, default=0)
        if cursor is None or str(cursor["uidvalidity"]) != "gmail":
            if initial != "all": return [], "gmail", top
            last = 0
        else: last = cursor["last_uid"]
        return [Ref(i, "gmail") for i in sorted(x for x in ids if x > last)[: self.batch]], "gmail", None
    def fetch_headers(self, ref):
        d = self._get(f"/messages/{ref.uid:016x}", {"format": "metadata", "metadataHeaders": HEADERS})
        raw = "".join(f"{h['name']}: {h['value']}\r\n" for h in d.get("payload", {}).get("headers", []) if h.get("name") in HEADERS) + "\r\n"
        return parse_headers(raw.encode("utf-8", "replace"), "gmail", ref.uid, int(d.get("sizeEstimate") or 0))
    def fetch_message(self, ref):
        d = self._get(f"/messages/{ref.uid:016x}", {"format": "raw"}); raw = base64.urlsafe_b64decode(d["raw"] + "=" * (-len(d["raw"]) % 4))
        if len(raw) > MAX_MESSAGE_BYTES: raise ProviderError("message_too_large")
        return parse_rfc822(raw, "gmail", ref.uid)
    def fetch_by_id(self, message_id):
        mid = str(message_id).strip().strip("<>")
        if not mid or re.search(r"[\x00-\x1f\"\\\s]", mid): raise ValueError("invalid message id")
        ids = self._ids(q=f"rfc822msgid:{mid}", pages=1)
        return self.fetch_message(Ref(max(ids), "gmail")) if ids else None

def check(tokens=None, base=None):
    """read-only status check: refresh the access token, read the profile counters, list at most 5 INBOX ids. Returns a dict WITHOUT message content; the address is masked."""
    p = GmailApiProvider(token_source=tokens, base=base); prof = p._get("/profile"); ids = p._ids(pages=1)[:5]
    a = prof.get("emailAddress", ""); u, _, dom = a.partition("@")
    return dict(authenticated=True, account=(u[:2] + "***@" + dom) if a else None, messages_total=prof.get("messagesTotal"), inbox_ids_listed=len(ids), api_scope_needed=SCOPE)

def granted_scope(tokens):
    """scopes Google actually granted to the stored authorization (tokeninfo, POST body so the token never appears in a URL). Returns a list or None when it cannot be determined."""
    try:
        req = urllib.request.Request("https://oauth2.googleapis.com/tokeninfo", data=urllib.parse.urlencode({"access_token": tokens.access_token()}).encode(), method="POST")
        with urllib.request.urlopen(req, timeout=15) as r: return str(json.loads(r.read().decode()).get("scope", "")).split()
    except Exception: return None
