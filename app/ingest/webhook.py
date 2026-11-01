"""Signed webhook receiver (stdlib http.server): verifies an HMAC-SHA256 signature, rejects stale timestamps, de-duplicates by event id, and ONLY enqueues a 'fetch' job.
No OCR/LLM runs in the request. Secret from HRBC_WEBHOOK_SECRET. Provider-specific payload parsing belongs in `parse_event`; the default accepts {"event_id","message_id"}.
Signature scheme (adapt per provider): hex(HMAC_SHA256(secret, f"{timestamp}.{raw_body}")) in header X-Signature, unix seconds in X-Timestamp."""
import hmac, hashlib, json, time, os
from http.server import BaseHTTPRequestHandler, HTTPServer
from . import db
MAX_BODY = 64 * 1024; TOLERANCE = 300

def sign(secret, ts, body): return hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()

def verify(secret, ts, sig, body, now=None):
    if not secret: return False, "webhook secret not configured"
    try: t = int(ts)
    except (TypeError, ValueError): return False, "bad timestamp"
    if abs((now or time.time()) - t) > TOLERANCE: return False, "stale timestamp"
    return (hmac.compare_digest(sign(secret, ts, body), str(sig or "")), "bad signature")

def handle(body, headers, secret, provider="webhook", now=None):
    """-> (http_status, message). Idempotent: a retried event id returns 200 without creating another job."""
    ok, why = verify(secret, headers.get("X-Timestamp"), headers.get("X-Signature"), body, now)
    if not ok: return 401, why
    try: ev = json.loads(body); eid, mid = str(ev["event_id"]), str(ev["message_id"])
    except Exception: return 400, "payload must contain event_id and message_id"
    c = db.conn()
    try:
        c.execute("BEGIN IMMEDIATE")
        if c.execute("INSERT OR IGNORE INTO webhook_events VALUES(?,?,?)", (provider, eid, db.now())).rowcount == 0: c.execute("COMMIT"); return 200, "duplicate event ignored"
        c.execute("INSERT OR IGNORE INTO jobs(key,attachment_id,state,created_at,updated_at) VALUES(?,?,?,?,?)", (f"fetch:{provider}:{mid}", None, "queued", db.now(), db.now()))
        db.audit(c, "webhook", "event_accepted", detail={"event_id": eid}); c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK"); return 500, "could not record event"
    return 202, "accepted"

def serve(host="127.0.0.1", port=8765):
    secret = os.environ.get("HRBC_WEBHOOK_SECRET", "")
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass                        # never log bodies
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n > MAX_BODY: self.send_response(413); self.end_headers(); return
            code, msg = handle(self.rfile.read(n), self.headers, secret); self.send_response(code); self.end_headers(); self.wfile.write(msg.encode())
    HTTPServer((host, port), H).serve_forever()
