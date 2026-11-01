"""Idempotent ingestion + background processing (hardened).
 poll_once(): cursor-based polling (UIDVALIDITY+UID in poll_state). Per message: fetch HEADERS -> authorise (sender allow-list, mailbox, aligned auth evidence) -> only then fetch the full message -> ingest_message. Every message is isolated: a failure is recorded in
   inbox_failures (safe code, attempts, next retry, bounded; terminal state 'dead' = dead-letter, recoverable with requeue_failures) and the cursor moves on; later messages are never blocked.
 ingest_message(): atomic (one transaction, rolled back on any error) -> validate each attachment independently -> hash/dedupe -> store 0600 -> one job per attachment (UNIQUE key). An email with no document is NEVER 'COMPLETED' (NO_DOCUMENT, or REVIEW_REQUIRED when the body looks like a bill).
 step(): worker claims ONE job with a lease, runs the heavy pipeline outside any request with a deterministic document id (a crash-and-retry cannot create a second document).
States (attachment): QUEUED -> PROCESSING -> COMPLETED | REVIEW_REQUIRED | FAILED ; plus DUPLICATE, IGNORED.  Email: VALIDATING -> QUEUED -> PROCESSING -> REVIEW_REQUIRED | COMPLETED | FAILED | REJECTED | NO_DOCUMENT."""
import os, json, hashlib, time, logging, collections, contextlib, sqlite3
from datetime import datetime
from pathlib import Path
from . import db, policy, config, perms, accounts
from .validate import validate, AttachmentError, ValidationUnavailable, MAX_ATTACHMENTS, MAX_TOTAL, MIN_BYTES
from .provider import ProviderError, ProviderAuthError, TlsVerificationError, RateLimited, ProviderTransient, MessageNotFound, Ref
log = logging.getLogger("ingest")        # ids and codes only: never OCR text, subjects of rejected mail, bodies, tokens, or addresses (sender is logged as a short hash)
RETRYABLE_EXC = (OSError, RuntimeError, TimeoutError, ConnectionError)
MAX_MSG_ATTEMPTS = int(os.environ.get("HRBC_INGEST_MAX_ATTEMPTS", 5))
class IngestFailure(Exception):
    def __init__(self, code): super().__init__(code); self.code = code

def _h(s): return hashlib.sha256(str(s).encode()).hexdigest()[:10]
def backoff(attempts, base=30, cap=3600): return min(cap, base * (2 ** max(0, attempts - 1)))
def inbox_dir(): return perms.private_dir(db.out_dir() / "ingest" / "inbox")
def code_of(e): return getattr(e, "code", None) or type(e).__name__

def add_sender(email_addr, employee_id, admin):
    """explicit authorisation: one row per address; several addresses (aliases) may map to one employee"""
    c = db.conn(); c.execute("INSERT OR REPLACE INTO senders VALUES(?,?,1,?,?)", (email_addr.strip().lower(), employee_id, admin, db.now())); db.audit(c, admin, "add_sender", detail={"employee_id": employee_id})

def authorise(msg, require_auth=None, mailbox=None, account=None):
    """-> (employee_id | None, reason); see policy.py. require_auth/mailbox are kept for compatibility and ignored: the policy comes from the validated configuration (fail closed).
    account: the connected mailbox the message was read from; its address (operator-registered) is an accepted recipient. The employee is still decided by the sender allow-list."""
    extra = [account["address"]] if account and not account.get("is_legacy") and account.get("address") else []
    return policy.authorise(msg, extra)

def rollup(c, email_id):
    cur = c.execute("SELECT status FROM emails WHERE id=?", (email_id,)).fetchone()["status"]
    sts = [r["state"] for r in c.execute("SELECT state FROM attachments WHERE email_id=?", (email_id,))]
    live = [s for s in sts if s not in ("IGNORED",)]
    if not live: return cur                                     # no document: the status set at ingest (NO_DOCUMENT / REVIEW_REQUIRED) stands; never promoted to COMPLETED
    elif "PROCESSING" in live: st = "PROCESSING"
    elif "QUEUED" in live: st = "QUEUED"
    elif all(s in ("COMPLETED", "DUPLICATE") for s in live): st = "COMPLETED"
    elif all(s == "FAILED" for s in live): st = "FAILED"
    else: st = "REVIEW_REQUIRED"
    reason = None
    if st == "REVIEW_REQUIRED": reason = "failed_attachments" if "FAILED" in live and "REVIEW_REQUIRED" not in live else "documents_need_review"
    c.execute("UPDATE emails SET status=?, reason=COALESCE(?,reason), updated_at=? WHERE id=?", (st, reason, db.now(), email_id)); return st

def record_rejection(msg, why, account=None):
    c = db.conn()
    c.execute("INSERT OR IGNORE INTO emails(provider,message_id,sender,subject,received_at,employee_id,status,reason,created_at,updated_at,account_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
              (msg.provider, msg.message_id, msg.sender, "[redacted: unauthorised sender]", msg.received_at, None, "REJECTED", why, db.now(), db.now(), account["id"] if account else None))
    log.warning("rejected message from sender hash %s: %s", _h(msg.sender), why)

def ingest_message(msg, account=None):
    """-> dict(email_id, status, attachments=[...]). Safe to call repeatedly with the same message. Raises IngestFailure(code) (after a full rollback) for environment/storage problems so the caller can retry with backoff.
    account: mail_accounts row the message came from (None = unattributed, as before). De-duplication stays GLOBAL on (provider, message_id): the same RFC Message-ID seen in two connected inboxes is stored once (document dedupe is per employee + content hash)."""
    c = db.conn(); aid = account["id"] if account else None; emp, why = authorise(msg, account=account)
    r = c.execute("SELECT id,status FROM emails WHERE provider=? AND message_id=?", (msg.provider, msg.message_id)).fetchone()
    if r: return dict(email_id=r["id"], status=r["status"], duplicate_message=True, attachments=[])
    if emp is None: record_rejection(msg, why, account); return dict(email_id=None, status="REJECTED", reason=why, attachments=[])
    created = []; bs = getattr(msg, "body_signal", None)
    try:
        c.execute("BEGIN IMMEDIATE")
        cur = c.execute("INSERT OR IGNORE INTO emails(provider,message_id,sender,subject,received_at,employee_id,status,n_attachments,body_signal,created_at,updated_at,account_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (msg.provider, msg.message_id, msg.sender, msg.subject, msg.received_at, emp, "VALIDATING", len(msg.attachments), json.dumps(bs) if bs else None, db.now(), db.now(), aid))
        if cur.rowcount == 0: c.execute("COMMIT"); return ingest_message(msg, account)                # lost a race with an identical delivery
        eid = cur.lastrowid; out = []; total = 0
        for i, a in enumerate(msg.attachments):
            sha = hashlib.sha256(a.data or b"").hexdigest(); rec = dict(idx=i, filename=(a.filename or "unnamed")[:120], mime=a.content_type, size=len(a.data or b""), sha256=sha)
            state, code, err, dup, pages, path = "QUEUED", None, None, None, None, None
            total += rec["size"]
            if a.inline and rec["size"] < MIN_BYTES * 10: state, code, err = "IGNORED", "inline_image", "Inline image (logo/signature) ignored."
            elif i >= MAX_ATTACHMENTS: state, code, err = "FAILED", "too_many_attachments", f"Only the first {MAX_ATTACHMENTS} attachments are processed."
            elif total > MAX_TOTAL: state, code, err = "FAILED", "email_too_large", "Total attachment size for this email exceeds the limit."
            else:
                try:
                    kind, pages = validate(a.data, a.filename, a.content_type)
                    d = c.execute("SELECT id, doc_id FROM attachments WHERE employee_id=? AND sha256=? AND state IN ('COMPLETED','REVIEW_REQUIRED','QUEUED','PROCESSING') ORDER BY id LIMIT 1", (emp, sha)).fetchone()
                    if d: state, dup, code, err = "DUPLICATE", d["id"], "duplicate_content", f"Same file already submitted (attachment {d['id']}); not processed again."
                    else:
                        ext = {"jpeg": ".jpg", "png": ".png", "pdf": ".pdf"}[kind]; p = inbox_dir() / f"{eid}_{i}_{sha[:12]}{ext}"
                        if not p.exists():
                            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                            with os.fdopen(fd, "wb") as f: f.write(a.data)
                            created.append(p)
                        path = str(p.relative_to(db.out_dir()))
                except AttachmentError as e: state, code, err = "FAILED", e.code, e.msg
                except ValidationUnavailable: raise IngestFailure("validation_tool_unavailable")        # environment problem: roll back, retry later; the attachment is NOT declared bad
            ac = c.execute("INSERT INTO attachments(email_id,idx,filename,mime,size,sha256,employee_id,state,error_code,error,duplicate_of,pages,stored_path,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (eid, i, rec["filename"], rec["mime"], rec["size"], sha, emp, state, code, err, dup, pages, path, db.now(), db.now()))
            if state == "QUEUED": c.execute("INSERT OR IGNORE INTO jobs(key,attachment_id,state,created_at,updated_at) VALUES(?,?,?,?,?)", (f"att:{ac.lastrowid}", ac.lastrowid, "queued", db.now(), db.now()))
            out.append(dict(rec, id=ac.lastrowid, state=state, error_code=code, error=err))
        if not [o for o in out if o["state"] != "IGNORED"]:       # nothing to process: say so explicitly instead of 'COMPLETED'
            if bs and bs.get("candidate"): c.execute("UPDATE emails SET status='REVIEW_REQUIRED', reason='possible_bill_in_body', updated_at=? WHERE id=?", (db.now(), eid))
            else: c.execute("UPDATE emails SET status='NO_DOCUMENT', reason='no_supported_attachment', updated_at=? WHERE id=?", (db.now(), eid))
            st = c.execute("SELECT status FROM emails WHERE id=?", (eid,)).fetchone()["status"]
        else: st = rollup(c, eid)
        db.audit(c, "ingest", "message_ingested", eid, None, {"attachments": len(out), "status": st}); c.execute("COMMIT")
    except IngestFailure:
        _rollback(c, created); raise
    except Exception as e:
        _rollback(c, created)
        raise IngestFailure("database_busy" if isinstance(e, sqlite3.OperationalError) else "storage_error" if isinstance(e, OSError) else "ingest_error") from None
    log.info("ingested email %s: %d attachments -> %s", eid, len(out), st)
    return dict(email_id=eid, status=st, attachments=out)

def _rollback(c, created):
    try: c.execute("ROLLBACK")
    except Exception: pass
    for p in created:
        try: p.unlink()
        except OSError: pass

def review_reasons(rec):
    """Why a processed document needs a human. Empty list -> COMPLETED."""
    r = []
    if rec.get("status") != "done": r.append(f"pipeline status '{rec.get('status')}'")
    ta = rec.get("task_a") or {}
    if ta and ta.get("route") != "auto": r.append("category confidence below routing threshold")
    pf = rec.get("profile") or {}
    if pf.get("outcome") == "needs_review": r.append("ambiguous document type")
    if (pf.get("validity") or {}).get("completeness") == "incomplete": r.append("missing: " + ", ".join(pf["validity"]["missing_fields"]))
    fr = rec.get("field_report") or {}
    for f, x in fr.items():
        if x["presence"] in ("conflict", "illegible"): r.append(f"{f} {x['presence']}")
        elif x["presence"] == "absent": r.append(f"{f} not found")
        elif x["validation"]["status"] == "failed": r.append(f"{f} failed validation: {','.join(x['validation']['failed'])}")
    return sorted(set(r))

def _notify(c, emp, text, doc_id=None): c.execute("INSERT INTO notifications(employee_id,ts,text,doc_id) VALUES(?,?,?,?)", (emp, db.now(), text, doc_id))

# ------------------------------------------------------------------ failures (per message) and dead letters
def record_failure(c, key, provider, ref, message_id, code, t, account_id=None):
    r = c.execute("SELECT attempts FROM inbox_failures WHERE key=?", (key,)).fetchone(); n = (r["attempts"] if r else 0) + 1; dead = n >= MAX_MSG_ATTEMPTS
    c.execute("""INSERT INTO inbox_failures(key,provider,uidvalidity,uid,message_id,state,attempts,max_attempts,next_run,error_code,first_seen,updated_at,account_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                 ON CONFLICT(key) DO UPDATE SET state=excluded.state, attempts=excluded.attempts, next_run=excluded.next_run, error_code=excluded.error_code, message_id=COALESCE(excluded.message_id, message_id), updated_at=excluded.updated_at, resolved_at=NULL""",
              (key, provider, getattr(ref, "uidvalidity", None), getattr(ref, "uid", None), message_id, "dead" if dead else "retry", n, MAX_MSG_ATTEMPTS, t + backoff(n), code, db.now(), db.now(), account_id))
    db.audit(c, "poller", "message_dead_letter" if dead else "message_failed", detail={"key": key, "code": code, "attempts": n})
    log.error("message %s failed (%s) attempt %d/%d%s", key, code, n, MAX_MSG_ATTEMPTS, " -> dead-letter" if dead else "")
    return "dead" if dead else "retry"

def requeue_failures(admin, key=None):
    """controlled recovery of dead-lettered items (messages and webhook fetch jobs): back to retry with a fresh attempt budget. Returns the number of distinct items."""
    c = db.conn(); c.execute("BEGIN IMMEDIATE"); a = (db.now(),) + ((key,) if key else ())
    keys = {r["key"] for r in c.execute("SELECT key FROM inbox_failures WHERE state='dead'" + (" AND key=?" if key else ""), a[1:])}
    c.execute("UPDATE inbox_failures SET state='retry', attempts=0, next_run=0, updated_at=? WHERE state='dead'" + (" AND key=?" if key else ""), a)
    keys |= {r["key"] for r in c.execute("SELECT key FROM jobs WHERE state='dead' AND attachment_id IS NULL" + (" AND key=?" if key else ""), a[1:])}
    c.execute("UPDATE jobs SET state='queued', attempts=0, next_run=0, lease_until=0, last_error=NULL, updated_at=? WHERE state='dead' AND attachment_id IS NULL" + (" AND key=?" if key else ""), a)
    db.audit(c, admin, "requeue_dead_letters", detail={"count": len(keys), "key": key}); c.execute("COMMIT"); return len(keys)

def failure_key(provider, ref, account=None):
    """identifier of one failed message. The legacy/unattributed mailbox keeps its historical format; every other account is namespaced, because provider ids (Gmail hex ids, IMAP UIDs) are only unique PER MAILBOX."""
    if account is None or account.get("is_legacy"): return f"{provider.name}:{ref.uidvalidity}:{ref.uid}"
    return f"{provider.name}@{account['public_id']}:{ref.uidvalidity}:{ref.uid}"

def _handle_ref(c, provider, ref, t, stats, retry=False, account=None):
    key = failure_key(provider, ref, account); mid = None
    try:
        hdr = provider.fetch_headers(ref); mid = hdr.message_id
        if getattr(hdr, "size", 0) and hdr.size > int(os.environ.get("HRBC_INGEST_MAX_MESSAGE_BYTES", 70 * 1024 * 1024)): record_rejection(hdr, "message_too_large", account); stats["rejected"] += 1
        else:
            emp, why = authorise(hdr, account=account)                  # BEFORE the body/attachments are downloaded
            if emp is None: record_rejection(hdr, why, account); stats["rejected"] += 1; provider.mark_processed(hdr)
            else:
                msg = provider.fetch_message(ref); res = ingest_message(msg, account); provider.mark_processed(msg)
                stats["duplicate" if res.get("duplicate_message") else "ingested"] += 1
        if retry: c.execute("UPDATE inbox_failures SET state='resolved', resolved_at=?, updated_at=? WHERE key=?", (db.now(), db.now(), key)); stats["recovered"] += 1
        return True
    except (ProviderAuthError, TlsVerificationError, RateLimited): raise       # affects every message: abort the poll, back off at poll level
    except Exception as e:
        st = record_failure(c, key, provider.name, ref, mid, code_of(e), t, account["id"] if account else None); stats["dead" if st == "dead" else "failed"] += 1; return False

def worker_state(c):
    c.execute("INSERT OR IGNORE INTO worker_state(id) VALUES(1)"); return c.execute("SELECT * FROM worker_state WHERE id=1").fetchone()

def poll_once(provider, processor=None, now=None, initial=None, account=None):
    """-> dict of counters. Never raises for provider/message problems (they are recorded and backed off); programming errors in the loop still surface.
    account: the mail_accounts row this provider serves. Cursor, failure records and backoff are then THIS account's; one account's failure never touches another's. The legacy/unattributed mailbox keeps using worker_state (health, existing behaviour)."""
    t = time.time() if now is None else now; stats = collections.Counter(); c = db.conn(); ws = worker_state(c)
    acc = accounts.get(account["id"], c) if account else None                       # fresh row: status/backoff may have changed since the caller read it
    own = acc is not None and not acc["is_legacy"]                                  # state kept on the account row (False: worker_state, as before)
    if acc is not None and acc["status"] != "active": stats["skipped_inactive"] = 1; return dict(stats)
    if t < ((acc["next_poll_after"] if own else ws["next_poll_after"]) or 0): stats["skipped_backoff"] = 1; return dict(stats)
    initial = initial or config.imap_settings()["initial"]
    if not hasattr(provider, "list_new"):                                           # legacy providers (fetch_new): same isolation guarantees
        for msg in provider.fetch_new():
            try: ingest_message(msg, acc); provider.mark_processed(msg); stats["ingested"] += 1
            except Exception as e: stats["failed"] += 1; log.error("legacy message failed (%s)", code_of(e))
        return dict(stats)
    key = provider.mailbox_key(); aid = acc["id"] if acc else None
    try:
        with provider.session():
            for row in c.execute("SELECT * FROM inbox_failures WHERE state='retry' AND provider=? AND next_run<=? AND account_id IS ? ORDER BY first_seen LIMIT 50", (provider.name, t, aid)).fetchall():
                _handle_ref(c, provider, Ref(row["uid"], row["uidvalidity"] or ""), t, stats, retry=True, account=acc)
            cur = db.get_cursor(provider.name, key); refs, uv, init_uid = provider.list_new(cur, initial)
            if init_uid is not None: db.set_cursor(c, provider.name, key, uv, init_uid); stats["cursor_initialised"] = 1
            if cur is not None and str(cur["uidvalidity"]) != str(uv): stats["uidvalidity_changed"] = 1; log.warning("UIDVALIDITY changed: re-reading the folder (Message-ID dedupe prevents duplicates)")
            for ref in refs:
                _handle_ref(c, provider, ref, t, stats, account=acc); db.set_cursor(c, provider.name, key, uv, ref.uid)      # progress is durable per message, after ingest or after the failure was recorded
        if not own: c.execute("UPDATE worker_state SET last_poll_ok_at=?, last_poll_error=NULL, consecutive_poll_failures=0, next_poll_after=0 WHERE id=1", (t,))
        if acc is not None: accounts.record_sync_ok(c, acc["id"])
    except ProviderError as e:
        n = ((acc["consecutive_failures"] if own else ws["consecutive_poll_failures"]) or 0) + 1; delay = e.retry_after if isinstance(e, RateLimited) and e.retry_after else backoff(n, 60, 3600)
        if not own: c.execute("UPDATE worker_state SET last_poll_error=?, consecutive_poll_failures=?, next_poll_after=? WHERE id=1", (e.code, n, t + delay))
        if acc is not None: accounts.record_error(c, acc["id"], e.code, n, t + delay, "needs_reauth" if own and isinstance(e, ProviderAuthError) else None)     # revoked/expired credentials: stop polling THIS account until re-authorised
        stats["poll_error"] = 1; log.error("poll failed: %s (consecutive %d, next attempt in %ds)", e.code, n, int(delay))
    return dict(stats)

def poll_accounts(factory=None, cache=None, now=None, initial=None):
    """poll every active connected (non-legacy) account independently, one provider each. factory(account) -> provider (default accounts.provider_for); cache keeps providers between loops.
    A failure of any account (bad credential file, revoked token, outage, unexpected exception) is recorded on THAT account only. -> {public_id: stats}"""
    factory = factory or accounts.provider_for; cache = {} if cache is None else cache; res = {}; t = time.time() if now is None else now
    for a in accounts.runnable():
        sig = (a["id"], a["updated_at"])
        try:
            if sig not in cache:
                for k in [k for k in cache if k[0] == a["id"]]: cache.pop(k)
                cache[sig] = factory(a)
            res[a["public_id"]] = poll_once(cache[sig], account=a, now=t, initial=initial)
        except accounts.AccountError as e:
            c = db.conn(); n = (a["consecutive_failures"] or 0) + 1; accounts.record_error(c, a["id"], e.args[0] if e.args else "account_error", n, t + backoff(n, 60, 3600), "needs_reauth" if e.args and e.args[0] in ("credential_missing", "credential_file_permissions") else None)
            res[a["public_id"]] = {"account_error": 1}; log.error("account %s unusable: %s", a["public_id"], e.args[0] if e.args else "account_error")
        except Exception as e:
            c = db.conn(); n = (a["consecutive_failures"] or 0) + 1; accounts.record_error(c, a["id"], code_of(e), n, t + backoff(n, 60, 3600))
            res[a["public_id"]] = {"account_error": 1}; log.error("account %s failed: %s", a["public_id"], type(e).__name__)           # class name only
    return res

# ------------------------------------------------------------------ jobs
def claim(c, lease=900, backoff_s=None):
    t = time.time(); c.execute("BEGIN IMMEDIATE")
    j = c.execute("SELECT * FROM jobs WHERE (state='queued' AND next_run<=?) OR (state='processing' AND lease_until<?) ORDER BY id LIMIT 1", (t, t)).fetchone()
    if not j: c.execute("COMMIT"); return None
    if j["attempts"] >= j["max_attempts"]:
        c.execute("UPDATE jobs SET state='dead', updated_at=? WHERE id=?", (db.now(), j["id"]))
        if j["attachment_id"]: c.execute("UPDATE attachments SET state='FAILED', error_code='retries_exhausted', error='Processing failed repeatedly; see the error history. A new submission or manual retry is required.', updated_at=? WHERE id=?", (db.now(), j["attachment_id"])); rollup(c, c.execute("SELECT email_id FROM attachments WHERE id=?", (j["attachment_id"],)).fetchone()["email_id"])
        else: _dead_fetch(c, j, j["last_error"] or "retries_exhausted")
        c.execute("COMMIT"); return "dead"
    c.execute("UPDATE jobs SET state='processing', attempts=attempts+1, lease_until=?, updated_at=? WHERE id=?", (t + lease, db.now(), j["id"])); c.execute("COMMIT"); return j

def _dead_fetch(c, j, code):
    c.execute("INSERT INTO inbox_failures(key,provider,message_id,state,attempts,max_attempts,error_code,first_seen,updated_at) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET state='dead', error_code=excluded.error_code, attempts=excluded.attempts, updated_at=excluded.updated_at",
              (j["key"], j["key"].split(":", 2)[1], j["key"].split(":", 2)[2], "dead", j["attempts"], j["max_attempts"], code, db.now(), db.now()))
    db.audit(c, "worker", "fetch_job_dead_letter", detail={"key": j["key"], "code": code})

def _doc_id(a):
    """deterministic per attachment (stable across crashes/retries); matches the app's id format YYYYMMDDTHHMMSS_xxxxxxxx"""
    try: ts = datetime.fromisoformat(a["created_at"]).strftime("%Y%m%dT%H%M%S")
    except Exception: ts = "20000101T000000"
    return f"{ts}_{hashlib.sha256(f'{a['id']}:{a['sha256']}'.encode()).hexdigest()[:8]}"

def step(processor, provider=None, retry_delay=30):
    """Run at most one job. processor(data, filename, owner, source) -> record dict (BillService.process). Returns None if idle, else a short outcome string."""
    c = db.conn(); j = claim(c)
    if j is None: return None
    if j == "dead": return "dead"
    if j["key"].startswith("fetch:"):
        mid = j["key"].split(":", 2)[2]
        try:
            if provider is None: raise ProviderTransient("no_provider_configured")
            with provider.session():
                msg = provider.fetch_by_id(mid)
                if msg is None: raise MessageNotFound("message not available from provider")
                ingest_message(msg); provider.mark_processed(msg)
            c.execute("UPDATE jobs SET state='done', last_error=NULL, updated_at=? WHERE id=?", (db.now(), j["id"])); c.execute("UPDATE inbox_failures SET state='resolved', resolved_at=?, updated_at=? WHERE key=?", (db.now(), db.now(), j["key"])); return "fetched"            # completed only after the message was durably ingested
        except Exception as e:
            code = code_of(e); c.execute("BEGIN IMMEDIATE")
            if j["attempts"] + 1 >= j["max_attempts"]:                  # claim() already counted this attempt in the database
                c.execute("UPDATE jobs SET state='dead', last_error=?, updated_at=? WHERE id=?", (code, db.now(), j["id"])); _dead_fetch(c, dict(j, attempts=j["attempts"] + 1, last_error=code), code); c.execute("COMMIT"); return "fetch_dead"
            wait = e.retry_after if isinstance(e, RateLimited) and e.retry_after else (backoff(j["attempts"] + 1, retry_delay) if retry_delay else 0)
            c.execute("UPDATE jobs SET state='queued', next_run=?, last_error=?, updated_at=? WHERE id=?", (time.time() + wait, code, db.now(), j["id"])); c.execute("COMMIT"); return "fetch_retry"
    a = c.execute("SELECT * FROM attachments WHERE id=?", (j["attachment_id"],)).fetchone()
    if a["state"] in ("COMPLETED", "REVIEW_REQUIRED", "DUPLICATE") and a["doc_id"]:      # work already done (retried / re-delivered job): do not redo it
        c.execute("UPDATE jobs SET state='done', updated_at=? WHERE id=?", (db.now(), j["id"])); return "already_done"
    c.execute("UPDATE attachments SET state='PROCESSING', updated_at=? WHERE id=?", (db.now(), a["id"])); rollup(c, a["email_id"])
    em = c.execute("SELECT * FROM emails WHERE id=?", (a["email_id"],)).fetchone(); path = db.out_dir() / a["stored_path"]
    try:
        data = path.read_bytes()
        rec = processor(data, a["filename"], a["employee_id"], {"channel": "email", "email_id": a["email_id"], "attachment_id": a["id"], "provider": em["provider"], "message_id": em["message_id"], "received_at": em["received_at"], "account_id": em["account_id"], "doc_id": _doc_id(a)})
        why = review_reasons(rec); st = "REVIEW_REQUIRED" if why else "COMPLETED"
        c.execute("BEGIN IMMEDIATE")
        c.execute("UPDATE attachments SET state=?, doc_id=?, error=?, error_code=NULL, updated_at=? WHERE id=?", (st, rec["doc_id"], "; ".join(why)[:500] or None, db.now(), a["id"]))
        c.execute("UPDATE jobs SET state='done', last_error=NULL, updated_at=? WHERE id=?", (db.now(), j["id"])); es = rollup(c, a["email_id"])
        _notify(c, a["employee_id"], f"'{a['filename']}' processed: {(rec.get('category') or 'unclassified')}" + (" - needs your review" if why else ""), rec["doc_id"]); db.audit(c, "worker", "processed", a["email_id"], a["id"], {"state": st}); c.execute("COMMIT")
        try: path.unlink()                                  # minimise copies: BillService stored the original under outputs/classified
        except OSError: pass
        return st
    except Exception as e:
        transient = isinstance(e, RETRYABLE_EXC) and j["attempts"] < j["max_attempts"]
        c.execute("BEGIN IMMEDIATE")
        if transient:
            c.execute("UPDATE jobs SET state='queued', next_run=?, last_error=?, updated_at=? WHERE id=?", (time.time() + retry_delay * (2 ** (j["attempts"] - 1)), f"{type(e).__name__}", db.now(), j["id"])); c.execute("UPDATE attachments SET state='QUEUED', updated_at=? WHERE id=?", (db.now(), a["id"]))
        else:
            c.execute("UPDATE jobs SET state='dead', last_error=?, updated_at=? WHERE id=?", (type(e).__name__, db.now(), j["id"]))
            code = "invalid_document" if type(e).__name__ == "UserError" else "processing_error"
            c.execute("UPDATE attachments SET state='FAILED', error_code=?, error=?, updated_at=? WHERE id=?", (code, ("The document could not be processed: " + str(e))[:300] if code == "invalid_document" else f"Processing failed ({type(e).__name__}). A reviewer can retry.", db.now(), a["id"]))
            _notify(c, a["employee_id"], f"'{a['filename']}' could not be processed", None)
        rollup(c, a["email_id"]); db.audit(c, "worker", "process_failed", a["email_id"], a["id"], {"error": type(e).__name__, "transient": transient}); c.execute("COMMIT")
        log.error("job %s failed: %s", j["id"], type(e).__name__); return "retry" if transient else "failed"

def drain(processor, provider=None, limit=1000, stop=lambda: False):
    k = 0
    while k < limit and not stop() and step(processor, provider) is not None: k += 1
    return k

def purge_inbox(older_than_days):
    """Retention: delete leftover raw copies in ingest/inbox older than N days (successful jobs already delete theirs). Explicit command only."""
    cutoff = time.time() - older_than_days * 86400; n = 0
    for p in inbox_dir().glob("*"):
        if p.is_file() and p.stat().st_mtime < cutoff: p.unlink(); n += 1
    return n

# ------------------------------------------------------------------ health
def health(now=None, interval=60):
    """privacy-safe operational snapshot: counts, ages and error CODES only. status: ok | degraded | down"""
    t = time.time() if now is None else now; c = db.conn(); ws = worker_state(c)
    q = lambda sql, *a: c.execute(sql, a).fetchone()[0]
    oldest = c.execute("SELECT MIN(created_at) FROM jobs WHERE state='queued'").fetchone()[0]
    try: age = (datetime.now().astimezone() - datetime.fromisoformat(oldest)).total_seconds() if oldest else 0
    except Exception: age = 0
    h = dict(worker_pid=ws["pid"], last_loop_age_s=(None if not ws["last_loop_at"] else int(t - ws["last_loop_at"])), last_poll_ok_age_s=(None if not ws["last_poll_ok_at"] else int(t - ws["last_poll_ok_at"])), last_poll_error=ws["last_poll_error"],
             consecutive_poll_failures=ws["consecutive_poll_failures"], jobs_queued=q("SELECT COUNT(*) FROM jobs WHERE state='queued'"), jobs_processing=q("SELECT COUNT(*) FROM jobs WHERE state='processing'"), jobs_dead=q("SELECT COUNT(*) FROM jobs WHERE state='dead'"),
             messages_retrying=q("SELECT COUNT(*) FROM inbox_failures WHERE state='retry'"), messages_dead=q("SELECT COUNT(*) FROM inbox_failures WHERE state='dead'"), oldest_queued_job_age_s=int(age), stopped=bool(ws["stopped_at"] and float(ws["stopped_at"]) >= (ws["last_loop_at"] or 0)))
    acc = c.execute("SELECT status, COUNT(*) n FROM mail_accounts WHERE is_legacy=0 GROUP BY status").fetchall(); h["mail_accounts"] = {r["status"]: r["n"] for r in acc}
    bad = c.execute("SELECT COUNT(*) FROM mail_accounts WHERE is_legacy=0 AND status='active' AND consecutive_failures>0").fetchone()[0]
    problems = []
    if h["mail_accounts"].get("needs_reauth"): problems.append("mail_account_needs_reauth")
    if bad: problems.append("mail_account_poll_error")
    if h["last_loop_age_s"] is None or h["last_loop_age_s"] > max(3 * interval, 180): problems.append("worker_not_running")
    if h["last_poll_error"]: problems.append("poll_error:" + h["last_poll_error"])
    if h["messages_dead"] or h["jobs_dead"]: problems.append("dead_letters_present")
    if h["oldest_queued_job_age_s"] > 1800: problems.append("queue_backlog")
    h["problems"] = problems; h["status"] = "down" if "worker_not_running" in problems else ("degraded" if problems else "ok"); return h
