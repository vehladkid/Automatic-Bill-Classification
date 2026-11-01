"""SQLite persistence for ingestion (outputs/hr/ingest.db, next to the claims DB; HRBC_OUT respected). Idempotency is enforced by UNIQUE constraints, not by application checks alone."""
import os, sqlite3, json
from . import perms
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def out_dir(): return Path(os.environ.get("HRBC_OUT", ROOT / "outputs"))
def get_cursor(provider, key):
    r = conn().execute("SELECT uidvalidity,last_uid FROM poll_state WHERE provider=? AND mailbox_key=?", (provider, key)).fetchone(); return dict(r) if r else None
def set_cursor(c, provider, key, uidvalidity, last_uid):
    c.execute("INSERT INTO poll_state(provider,mailbox_key,uidvalidity,last_uid,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(provider,mailbox_key) DO UPDATE SET uidvalidity=excluded.uidvalidity,last_uid=excluded.last_uid,updated_at=excluded.updated_at", (provider, key, str(uidvalidity), int(last_uid), now()))
def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
SCHEMA = """
CREATE TABLE IF NOT EXISTS senders(email TEXT PRIMARY KEY, employee_id TEXT NOT NULL, active INTEGER DEFAULT 1, added_by TEXT, added_at TEXT);
CREATE TABLE IF NOT EXISTS webhook_events(provider TEXT, event_id TEXT, received_at TEXT, PRIMARY KEY(provider, event_id));
CREATE TABLE IF NOT EXISTS emails(id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT, message_id TEXT, sender TEXT, subject TEXT, received_at TEXT, employee_id TEXT,
  status TEXT, reason TEXT, n_attachments INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT, UNIQUE(provider, message_id));
CREATE TABLE IF NOT EXISTS attachments(id INTEGER PRIMARY KEY AUTOINCREMENT, email_id INTEGER NOT NULL, idx INTEGER, filename TEXT, mime TEXT, size INTEGER, sha256 TEXT, employee_id TEXT,
  state TEXT, error_code TEXT, error TEXT, doc_id TEXT, duplicate_of INTEGER, pages INTEGER, stored_path TEXT, created_at TEXT, updated_at TEXT, UNIQUE(email_id, idx));
CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, attachment_id INTEGER, state TEXT, attempts INTEGER DEFAULT 0, max_attempts INTEGER DEFAULT 3,
  next_run REAL DEFAULT 0, lease_until REAL DEFAULT 0, last_error TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS notifications(id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id TEXT, ts TEXT, text TEXT, doc_id TEXT, read INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS corrections(id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, field TEXT, original TEXT, corrected TEXT, actor TEXT, role TEXT, ts TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS ingest_audit(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, action TEXT, email_id INTEGER, attachment_id INTEGER, detail TEXT);
CREATE TABLE IF NOT EXISTS poll_state(provider TEXT, mailbox_key TEXT, uidvalidity TEXT, last_uid INTEGER DEFAULT 0, updated_at TEXT, PRIMARY KEY(provider, mailbox_key));
CREATE TABLE IF NOT EXISTS inbox_failures(key TEXT PRIMARY KEY, provider TEXT, uidvalidity TEXT, uid INTEGER, message_id TEXT, state TEXT, attempts INTEGER DEFAULT 0, max_attempts INTEGER DEFAULT 5,
  next_run REAL DEFAULT 0, error_code TEXT, first_seen TEXT, updated_at TEXT, resolved_at TEXT);
CREATE TABLE IF NOT EXISTS mail_accounts(id INTEGER PRIMARY KEY AUTOINCREMENT, public_id TEXT UNIQUE NOT NULL, owner_employee_id TEXT, provider TEXT NOT NULL, provider_account_id TEXT, address TEXT,
  mailbox_key TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', credential_ref TEXT, is_legacy INTEGER NOT NULL DEFAULT 0, last_sync_at TEXT, last_error_code TEXT, last_error_at TEXT,
  consecutive_failures INTEGER DEFAULT 0, next_poll_after REAL DEFAULT 0, created_by TEXT, created_at TEXT, updated_at TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS ux_mail_accounts_identity ON mail_accounts(provider, provider_account_id) WHERE provider_account_id IS NOT NULL AND status != 'disconnected';
CREATE UNIQUE INDEX IF NOT EXISTS ux_mail_accounts_cursor ON mail_accounts(provider, mailbox_key) WHERE status != 'disconnected';
CREATE TABLE IF NOT EXISTS worker_state(id INTEGER PRIMARY KEY CHECK(id=1), pid INTEGER, started_at TEXT, last_loop_at REAL, last_poll_ok_at REAL, last_poll_error TEXT, consecutive_poll_failures INTEGER DEFAULT 0, next_poll_after REAL DEFAULT 0, stopped_at TEXT);
"""
def _backfill_legacy_account(c):
    """Migration 4: every pre-existing email / failure record belonged to the ONE mailbox the worker polled. Represent it as a clearly identified legacy connection (is_legacy=1, credentials stay environment-managed)
    and stamp the old rows with it. Idempotent, one transaction. Does nothing on a database without ingestion history (the worker creates the legacy row on first start)."""
    c.execute("BEGIN IMMEDIATE")
    try:
        if c.execute("SELECT 1 FROM mail_accounts WHERE is_legacy=1").fetchone() is None:
            ps = c.execute("SELECT provider, mailbox_key FROM poll_state ORDER BY updated_at DESC, rowid DESC").fetchall()
            em = c.execute("SELECT provider FROM emails WHERE account_id IS NULL ORDER BY id DESC LIMIT 1").fetchone()
            if ps or em:
                prov, key = (ps[0]["provider"], ps[0]["mailbox_key"]) if ps else (em["provider"], "legacy:" + em["provider"])
                c.execute("INSERT INTO mail_accounts(public_id,owner_employee_id,provider,mailbox_key,status,credential_ref,is_legacy,created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                          ("acct_legacy", None, prov, key, "active", "environment", 1, "migration", now(), now()))
                c.execute("INSERT INTO ingest_audit(ts,actor,action,detail) VALUES(?,?,?,?)", (now(), "migration", "legacy_mail_account_created", json.dumps({"unlinked_poll_state_rows": max(0, len(ps) - 1)})))
        lg = c.execute("SELECT id FROM mail_accounts WHERE is_legacy=1").fetchone()
        if lg:
            c.execute("UPDATE emails SET account_id=? WHERE account_id IS NULL", (lg["id"],)); c.execute("UPDATE inbox_failures SET account_id=? WHERE account_id IS NULL", (lg["id"],))
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK"); raise
MIGRATIONS = [   # PRAGMA user_version = number applied; append only. An entry is SQL (ADD COLUMN tolerated if present) or a callable(conn) that must itself be idempotent and transactional.
    "ALTER TABLE emails ADD COLUMN body_signal TEXT",
    "ALTER TABLE emails ADD COLUMN auth_evidence TEXT",
    "ALTER TABLE emails ADD COLUMN account_id INTEGER REFERENCES mail_accounts(id)",
    "ALTER TABLE inbox_failures ADD COLUMN account_id INTEGER",
    _backfill_legacy_account,
    "ALTER TABLE mail_accounts ADD COLUMN client_ref TEXT",       # path of the OAuth client JSON that issued this account's refresh token (NULL = the environment's client)
]
def conn():
    d = out_dir(); perms.private_dir(d); perms.private_dir(d / "hr"); p = d / "hr" / "ingest.db"
    c = sqlite3.connect(p, timeout=15, isolation_level=None); c.row_factory = sqlite3.Row; c.execute("PRAGMA journal_mode=WAL"); c.executescript(SCHEMA)
    v = c.execute("PRAGMA user_version").fetchone()[0]
    for i in range(v, len(MIGRATIONS)):
        try: MIGRATIONS[i](c) if callable(MIGRATIONS[i]) else c.execute(MIGRATIONS[i])
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e): raise
        c.execute(f"PRAGMA user_version={i + 1}")
    perms.secure_db(p); return c
def audit(c, actor, action, email_id=None, attachment_id=None, detail=None):
    c.execute("INSERT INTO ingest_audit(ts,actor,action,email_id,attachment_id,detail) VALUES(?,?,?,?,?,?)", (now(), actor, action, email_id, attachment_id, json.dumps(detail or {}, default=str)))
