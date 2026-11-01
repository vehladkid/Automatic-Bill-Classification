"""Mail provider abstraction + the IMAP adapter.
Provider contract used by the pipeline (cursor flow):  session() -> context manager; list_new(cursor, initial) -> (refs, uidvalidity, init_uid)  (init_uid not None = first run in new_only mode: set the cursor there, ingest nothing existing); fetch_headers(ref) -> Message without attachments (+size);
fetch_message(ref) -> full Message; fetch_by_id(message_id) -> Message | None; mark_processed(msg) (optional, off by default: the mailbox is opened READ-ONLY unless HRBC_IMAP_MARK_SEEN=1).
Errors are classified so the pipeline can react: ProviderAuthError (credentials rejected), TlsVerificationError (certificate/hostname not trusted), RateLimited (backoff, optional Retry-After), ProviderTransient (network/server).
Nothing here logs credentials, tokens, message bodies or attachment contents. NOT verified against a live mailbox (no test mailbox was available); verified against a local IMAP server in tests/test_ingest_hardening.py and reports/email_ingestion/smoke."""
import os, re, ssl, socket, imaplib, email, html, contextlib
from email import policy
from email.utils import parseaddr, getaddresses, parsedate_to_datetime
from html.parser import HTMLParser
from dataclasses import dataclass, field
from . import config, tls, oauth

class ProviderError(Exception): code = "provider_error"
class ProviderTransient(ProviderError): code = "provider_transient"
class ProviderAuthError(ProviderError): code = "credentials_rejected"
class TlsVerificationError(ProviderError): code = "tls_verification_failed"
class RateLimited(ProviderError):
    code = "rate_limited"
    def __init__(self, retry_after=None): super().__init__("rate limited"); self.retry_after = retry_after
class MessageNotFound(ProviderError): code = "message_not_found"

@dataclass
class Attachment:
    filename: str; content_type: str; data: bytes; inline: bool = False
@dataclass
class Message:
    provider: str; message_id: str; sender: str; recipient: str; subject: str; received_at: str
    attachments: list = field(default_factory=list); auth_results: str = ""
    body_ignored: bool = True                       # body text is never executed or trusted; only a bill-likelihood signal is derived from it
    recipients: list = field(default_factory=list); from_count: int = 1; auth_headers: list = field(default_factory=list)
    body_signal: dict = None; size: int = 0; uid: int = None; has_attachments_hint: bool = None

@dataclass
class Ref:
    uid: int; uidvalidity: str = ""

class MailProvider:
    name = "base"
    def session(self): return contextlib.nullcontext(self)
    def list_new(self, cursor, initial="new_only"): raise NotImplementedError
    def fetch_headers(self, ref): raise NotImplementedError
    def fetch_message(self, ref): raise NotImplementedError
    def fetch_by_id(self, message_id): raise NotImplementedError
    def mailbox_key(self): return "default"
    def mark_processed(self, message): pass

# ---------------------------------------------------------------- parsing
BILL_TERMS = ["tax invoice", "invoice", "receipt", "amount due", "amount paid", "total", "order id", "order no", "booking id", "gstin", "bill", "payment", "paid", "fare", "subtotal", "statement"]
AMOUNT = re.compile(r"(?:₹|rs\.?|inr)\s?[\d,]+(?:\.\d{1,2})?", re.I)
class _Text(HTMLParser):
    def __init__(self): super().__init__(); self.t = []; self.skip = 0
    def handle_starttag(self, tag, a):
        if tag in ("script", "style"): self.skip += 1
    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip: self.skip -= 1
    def handle_data(self, d):
        if not self.skip: self.t.append(d)
def body_signal(m):
    """bill-likelihood of the BODY text (outer message and any forwarded message/rfc822 parts). Only counts and cue labels are kept: the text itself is discarded."""
    texts = []; has_html = False
    for part in m.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment" or part.get_filename(): continue
        ct = part.get_content_type()
        if ct not in ("text/plain", "text/html"): continue
        try: raw = part.get_content()
        except Exception: continue
        if ct == "text/html": has_html = True; p = _Text(); p.feed(raw); raw = " ".join(p.t)
        texts.append(raw)
    t = html.unescape(" ".join(texts)).lower(); cues = sorted({x for x in BILL_TERMS if x in t}); amounts = len(AMOUNT.findall(t))
    cand = (amounts >= 1 and len(cues) >= 2) or len(cues) >= 4
    return dict(candidate=bool(cand), cues=cues, amount_mentions=min(amounts, 20), has_html=has_html, chars=len(t))
def _addrs(m, *names): return sorted({a.lower() for n in names for _, a in getaddresses(m.get_all(n, [])) if a})
def _envelope(m, provider, uid):
    try: ts = parsedate_to_datetime(m["Date"]).isoformat() if m["Date"] else ""
    except Exception: ts = ""
    mid = (m["Message-ID"] or "").strip() or f"{provider}-uid-{uid}"
    froms = _addrs(m, "From"); auth = [str(h) for h in m.get_all("Authentication-Results", [])]
    rc = _addrs(m, "To", "Cc", "Delivered-To", "X-Original-To", "Envelope-To")
    return dict(provider=provider, message_id=mid, sender=(froms[0] if froms else ""), recipient=(_addrs(m, "To") or [""])[0], subject=str(m["Subject"] or "")[:300], received_at=ts, auth_results="; ".join(auth),
                recipients=rc, from_count=len(froms), auth_headers=auth)
def parse_headers(raw, provider="imap", uid=None, size=0):
    m = email.message_from_bytes(raw, policy=policy.default); return Message(attachments=[], size=size, uid=uid, **_envelope(m, provider, uid))
def parse_rfc822(raw, provider="imap", uid=None):
    m = email.message_from_bytes(raw, policy=policy.default); atts = []
    for part in m.walk():
        if part.is_multipart(): continue
        disp = (part.get_content_disposition() or ""); fn = part.get_filename()
        if disp not in ("attachment", "inline") and not fn: continue
        data = part.get_payload(decode=True) or b""
        atts.append(Attachment(filename=fn or "unnamed", content_type=part.get_content_type(), data=data, inline=(disp == "inline" and bool(part.get("Content-ID")))))
    return Message(attachments=atts, size=len(raw), uid=uid, body_signal=body_signal(m), **_envelope(m, provider, uid))

# ---------------------------------------------------------------- fake provider (tests / demos)
class FakeProvider(MailProvider):
    """In-memory provider implementing BOTH the legacy API (fetch_new) and the cursor contract, with failure injection: fail_fetch={uid: [exceptions]}, fail_headers likewise."""
    name = "fake"
    def __init__(self, messages=None, uidvalidity="1"):
        self.messages = list(messages or []); self.processed = []; self.uidvalidity = uidvalidity; self.fail_fetch = {}; self.fail_headers = {}; self.fail_list = []; self.fetch_calls = []
        for i, m in enumerate(self.messages, 1): m.uid = m.uid or i
    def add(self, m): m.uid = len(self.messages) + 1; self.messages.append(m); return m
    def fetch_new(self): return [m for m in self.messages if m.message_id not in self.processed]
    def mailbox_key(self): return "fake"
    def list_new(self, cursor, initial="new_only"):
        if self.fail_list: raise self.fail_list.pop(0)
        top = max([m.uid for m in self.messages] or [0])
        if cursor is None and initial == "new_only": return [], self.uidvalidity, top
        last = cursor["last_uid"] if cursor and str(cursor["uidvalidity"]) == self.uidvalidity else 0
        return [Ref(m.uid, self.uidvalidity) for m in self.messages if m.uid > last], self.uidvalidity, None
    def _get(self, ref): return next(m for m in self.messages if m.uid == ref.uid)
    def fetch_headers(self, ref):
        q = self.fail_headers.get(ref.uid)
        if q: raise q.pop(0)
        m = self._get(ref); return Message(m.provider, m.message_id, m.sender, m.recipient, m.subject, m.received_at, [], m.auth_results, recipients=m.recipients or [m.recipient], from_count=m.from_count, auth_headers=m.auth_headers or ([m.auth_results] if m.auth_results else []), uid=m.uid, size=sum(len(a.data or b"") for a in m.attachments))
    def fetch_message(self, ref):
        self.fetch_calls.append(ref.uid); q = self.fail_fetch.get(ref.uid)
        if q: raise q.pop(0)
        return self._get(ref)
    def fetch_by_id(self, message_id): return next((m for m in self.messages if m.message_id == message_id), None)
    def mark_processed(self, message): self.processed.append(message.message_id)

# ---------------------------------------------------------------- IMAP adapter
MAX_MESSAGE_BYTES = int(os.environ.get("HRBC_INGEST_MAX_MESSAGE_BYTES", 70 * 1024 * 1024))
def _classify_imap_error(e):
    t = str(e).upper()
    if any(k in t for k in ("AUTHENTICATIONFAILED", "INVALID CREDENTIALS", "AUTHENTICATION FAILED", "LOGIN FAILED", "[AUTHORIZATIONFAILED]")): return ProviderAuthError("credentials rejected")
    if any(k in t for k in ("THROTTLE", "TOO MANY", "RATE", "OVERQUOTA", "BANDWIDTH")): return RateLimited(None)
    return ProviderTransient("imap_error")
class ImapProvider(MailProvider):
    """IMAP over verified TLS. Auth: password (app password) or XOAUTH2 (RefreshTokenSource). Mailbox is opened READ-ONLY (no flag changes) unless mark_seen is enabled explicitly."""
    name = "imap"
    def __init__(self, host=None, user=None, password=None, folder=None, port=None, token_source=None, ssl_context=None, mark_seen=None, timeout=30, batch=None):
        s = config.imap_settings()
        self.host, self.user = host or s["host"], user or s["user"]; self.folder = folder or s["folder"]; self.port = port or s["port"]; self.mark_seen = s["mark_seen"] if mark_seen is None else mark_seen
        self.batch = batch or s["batch"]; self.timeout = timeout; self.auth = "xoauth2" if token_source else ("password" if (password or config.secret("HRBC_IMAP_PASSWORD")) else s["auth"])
        self._password = password or (config.secret("HRBC_IMAP_PASSWORD") if self.auth == "password" else None); self.tokens = token_source
        if self.auth == "xoauth2" and self.tokens is None: self.tokens = oauth.RefreshTokenSource.from_env()
        if not (self.host and self.user and (self._password or self.tokens)): raise RuntimeError("IMAP not configured: HRBC_IMAP_HOST, HRBC_IMAP_USER and a password or OAuth settings are required (environment only; never commit them).")
        self.ctx = ssl_context or tls.context(s["ca_file"]); self._c = None; self._uv = "0"
    def mailbox_key(self): return f"{self.host}:{self.port}/{self.user}/{self.folder}"
    # connection ---------------------------------------------------------
    def _connect(self):
        try: c = imaplib.IMAP4_SSL(self.host, self.port, ssl_context=self.ctx, timeout=self.timeout)
        except ssl.SSLCertVerificationError: raise TlsVerificationError("certificate not trusted or hostname mismatch")
        except ssl.SSLError: raise TlsVerificationError("tls_handshake_failed")
        except (socket.timeout, TimeoutError, OSError): raise ProviderTransient("connect_failed")
        try:
            if self.auth == "xoauth2":
                try: tok = self.tokens.access_token()
                except oauth.CredentialsExpired: raise ProviderAuthError("oauth credentials expired or revoked")
                except oauth.TransientAuthError as e: raise (RateLimited(e.retry_after) if e.code in ("rate_limit_exceeded", "http_429") or e.retry_after else ProviderTransient("token_endpoint"))
                state = {"n": 0}
                def cb(_):
                    state["n"] += 1; return oauth.xoauth2_string(self.user, tok) if state["n"] == 1 else b""
                c.authenticate("XOAUTH2", cb)
            else: c.login(self.user, self._password)
            typ, _ = c.select(self.folder, readonly=not self.mark_seen)
            if typ != "OK": raise ProviderTransient("select_failed")
            t2, d2 = c.response("UIDVALIDITY"); self._uv = d2[0].decode() if d2 and d2[0] else "0"        # imaplib pops untagged responses when read: capture once here
        except imaplib.IMAP4.error as e:
            self._quiet_logout(c); raise _classify_imap_error(e)
        except (socket.timeout, TimeoutError, OSError): self._quiet_logout(c); raise ProviderTransient("connection_lost")
        except ProviderError: self._quiet_logout(c); raise
        return c
    @staticmethod
    def _quiet_logout(c):
        try: c.logout()
        except Exception: pass
    @contextlib.contextmanager
    def session(self):
        self._c = self._connect()
        try: yield self
        finally: self._quiet_logout(self._c); self._c = None
    def _conn(self):
        if self._c is None: raise ProviderTransient("no_session")
        return self._c
    def _guard(self, fn):
        try: return fn()
        except imaplib.IMAP4.error as e: raise _classify_imap_error(e)
        except (socket.timeout, TimeoutError, OSError, ssl.SSLError): raise ProviderTransient("connection_lost")
    # reading ------------------------------------------------------------
    def list_new(self, cursor, initial="new_only"):
        c = self._conn(); uv = None
        def go():
            nonlocal uv
            uv = self._uv
            if cursor is not None and str(cursor["uidvalidity"]) == uv: last = cursor["last_uid"]
            else:
                last = 0 if (cursor is not None or initial == "all") else None      # UIDVALIDITY changed or first run with initial=all: start from the beginning (Message-ID dedupe protects)
            typ, d = c.uid("search", None, "ALL" if last in (0, None) else f"UID {last + 1}:*")
            uids = sorted(int(x) for x in (d[0].split() if typ == "OK" and d and d[0] else []))
            if last is None:                                                         # first run, new_only: record the current top and ingest nothing that already exists
                return [], uv, (uids[-1] if uids else 0)
            return [Ref(u, uv) for u in uids if u > last][: self.batch], uv, None
        return self._guard(go)
    def _fetch(self, uid, what):
        c = self._conn()
        def go():
            typ, data = c.uid("fetch", str(uid), what)
            if typ != "OK" or not data or data[0] is None: raise MessageNotFound("uid vanished")
            return data
        return self._guard(go)
    def fetch_headers(self, ref):
        data = self._fetch(ref.uid, "(RFC822.SIZE BODY.PEEK[HEADER])"); raw = next((x[1] for x in data if isinstance(x, tuple)), b""); meta = next((x[0] for x in data if isinstance(x, tuple)), b"").decode(errors="ignore")
        m = re.search(r"RFC822\.SIZE (\d+)", meta + " ".join(x.decode(errors="ignore") for x in data if isinstance(x, bytes))); size = int(m.group(1)) if m else 0
        return parse_headers(raw, "imap", ref.uid, size)
    def fetch_message(self, ref):
        data = self._fetch(ref.uid, "(BODY.PEEK[])"); raw = next(x[1] for x in data if isinstance(x, tuple))
        if len(raw) > MAX_MESSAGE_BYTES: raise ProviderError("message_too_large")
        return parse_rfc822(raw, "imap", ref.uid)
    def fetch_by_id(self, message_id):
        mid = str(message_id).strip()
        if not mid or re.search(r"[\x00-\x1f\"\\]", mid): raise ValueError("invalid message id")
        c = self._conn()
        def go():
            typ, d = c.uid("search", None, "HEADER", "Message-ID", f'"{mid}"')
            return sorted(int(x) for x in (d[0].split() if typ == "OK" and d and d[0] else []))
        uids = self._guard(go)
        if not uids: return None
        return self.fetch_message(Ref(uids[-1]))
    def mark_processed(self, message):
        if not self.mark_seen or message.uid is None: return
        c = self._conn(); self._guard(lambda: c.uid("store", str(message.uid), "+FLAGS", "(\\Seen)"))
