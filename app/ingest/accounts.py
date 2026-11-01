"""Connected mailboxes ('mail accounts'): one row per independently authenticated inbox in ingest.db.mail_accounts.
Each account owns: a stable id (public_id), its provider identity (provider + provider_account_id/address), its OWN credential file, its OWN cursor (poll_state row keyed by provider + mailbox_key),
a status (active | needs_reauth | disconnected) and sync diagnostics (last_sync_at, last_error_code, consecutive_failures, next_poll_after).
Credentials: never stored in the database. The row holds only a reference (path) to a 0600 file under HRBC_ACCOUNTS_DIR (default ~/.config/hrbc/accounts, directory 0700) - the same mechanism as the existing refresh-token file.
The legacy account (is_legacy=1) is the mailbox configured through the environment (worker.env); its credentials stay environment-managed and it cannot be disconnected from here.
OWNERSHIP: owner_employee_id records who the connection was registered for. It does NOT decide document ownership: the sender allow-list (policy.authorise) still maps each message to an employee, exactly as before.
Linking is an OPERATOR action (CLI, `python -m ingest.run mail-add`): the application has no authentication, so no web/UI path can create, change or remove a connection."""
import os, re, secrets, json
from pathlib import Path
from . import db, config, oauth

class AccountError(Exception):
    """message is safe to show (no secrets)"""

STATUSES = ("active", "needs_reauth", "disconnected")
PUBLIC_FIELDS = ("id", "public_id", "owner_employee_id", "provider", "address", "status", "is_legacy", "last_sync_at", "last_error_code", "last_error_at", "consecutive_failures", "created_at", "updated_at")

def accounts_dir():
    return Path(os.environ.get("HRBC_ACCOUNTS_DIR") or (Path.home() / ".config" / "hrbc" / "accounts"))

def _row(r): return dict(r) if r is not None else None

def public_view(a):
    """a row without credential_ref, mailbox_key or scheduling internals - safe for the UI/API"""
    return {k: a[k] for k in PUBLIC_FIELDS}

def get(account_id, c=None):
    c = c or db.conn(); return _row(c.execute("SELECT * FROM mail_accounts WHERE id=? OR public_id=?", (account_id if isinstance(account_id, int) else -1, str(account_id))).fetchone())

def list_accounts(include_disconnected=True, c=None):
    c = c or db.conn(); return [dict(r) for r in c.execute("SELECT * FROM mail_accounts" + ("" if include_disconnected else " WHERE status != 'disconnected'") + " ORDER BY is_legacy DESC, id")]

def runnable(c=None):
    """accounts the worker polls besides the legacy mailbox: status active, not legacy"""
    c = c or db.conn(); return [dict(r) for r in c.execute("SELECT * FROM mail_accounts WHERE is_legacy=0 AND status='active' ORDER BY id")]

def ensure_legacy(provider, address=None):
    """the environment-configured mailbox -> its legacy account row (created on first worker start of a fresh database; found/backfilled by the migration otherwise)."""
    c = db.conn(); key = provider.mailbox_key(); c.execute("BEGIN IMMEDIATE")
    try:
        r = c.execute("SELECT * FROM mail_accounts WHERE is_legacy=1").fetchone()
        if r is None:
            c.execute("INSERT INTO mail_accounts(public_id,provider,address,mailbox_key,status,credential_ref,is_legacy,created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                      ("acct_legacy", provider.name, address, key, "active", "environment", 1, "worker", db.now(), db.now()))
        elif (r["provider"], r["mailbox_key"], r["address"] or None) != (provider.name, key, address or r["address"]):
            c.execute("UPDATE mail_accounts SET provider=?, mailbox_key=?, address=COALESCE(?,address), updated_at=? WHERE id=?", (provider.name, key, address, db.now(), r["id"]))
            db.audit(c, "worker", "legacy_mail_account_reconfigured", detail={"provider": provider.name})
        r = c.execute("SELECT * FROM mail_accounts WHERE is_legacy=1").fetchone()
        c.execute("UPDATE emails SET account_id=? WHERE account_id IS NULL", (r["id"],)); c.execute("UPDATE inbox_failures SET account_id=? WHERE account_id IS NULL", (r["id"],))    # rows written by a pre-upgrade worker still running during the upgrade
        c.execute("COMMIT"); return dict(r)
    except Exception:
        c.execute("ROLLBACK"); raise

def _new_public_id(): return "acct_" + secrets.token_hex(5)

def add_account(owner_employee_id, provider, provider_account_id, address, refresh_token, admin, client_ref=None):
    """register (or re-authorise) a mailbox for an employee. A second account never touches the first one's credential file.
    Same provider identity already connected: same owner -> credentials replaced and status reset (re-authorisation); other owner -> refused. Returns the public view."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", str(owner_employee_id or "")): raise AccountError("invalid employee id")
    if not refresh_token: raise AccountError("no credential supplied")
    pid_ = str(provider_account_id or "").strip().lower()
    if not pid_: raise AccountError("provider account identity is required")
    from . import google_auth
    c = db.conn(); c.execute("BEGIN IMMEDIATE")
    try:
        ex = c.execute("SELECT * FROM mail_accounts WHERE provider=? AND provider_account_id=? AND status != 'disconnected'", (provider, pid_)).fetchone()
        if ex is not None:
            if ex["owner_employee_id"] != owner_employee_id: raise AccountError("this mailbox is already connected for another user")
            ref = ex["credential_ref"]; google_auth.write_private(ref, refresh_token)
            c.execute("UPDATE mail_accounts SET status='active', last_error_code=NULL, last_error_at=NULL, consecutive_failures=0, next_poll_after=0, client_ref=COALESCE(?,client_ref), updated_at=? WHERE id=?", (client_ref, db.now(), ex["id"]))
            db.audit(c, admin, "mail_account_reauthorised", detail={"account": ex["public_id"]}); aid = ex["id"]
        else:
            pub = _new_public_id(); ref = str(accounts_dir() / f"{pub}.token"); google_auth.write_private(ref, refresh_token)
            cur = c.execute("INSERT INTO mail_accounts(public_id,owner_employee_id,provider,provider_account_id,address,mailbox_key,status,credential_ref,is_legacy,created_by,created_at,updated_at,client_ref) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (pub, owner_employee_id, provider, pid_, address, f"acct:{pub}", "active", ref, 0, admin, db.now(), db.now(), client_ref))
            aid = cur.lastrowid; db.audit(c, admin, "mail_account_connected", detail={"account": pub, "provider": provider})
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK"); raise
    return public_view(get(aid))

def disconnect(account_id, admin):
    """stop polling this account and delete ITS stored credential. History (emails, documents) is kept. Other accounts are untouched. The legacy mailbox cannot be disconnected here."""
    c = db.conn(); c.execute("BEGIN IMMEDIATE")
    try:
        a = get(account_id, c)
        if a is None: raise AccountError("no such connection")
        if a["is_legacy"]: raise AccountError("the environment-configured (legacy) mailbox is managed in worker.env, not here")
        if a["status"] == "disconnected": c.execute("COMMIT"); return public_view(a)
        c.execute("UPDATE mail_accounts SET status='disconnected', credential_ref=NULL, next_poll_after=0, updated_at=? WHERE id=?", (db.now(), a["id"]))
        db.audit(c, admin, "mail_account_disconnected", detail={"account": a["public_id"]}); c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK"); raise
    ref = a["credential_ref"]
    if ref:
        try:
            p = Path(ref)
            if p.parent == accounts_dir() and p.is_file(): p.unlink()
        except OSError: pass
    return public_view(get(a["id"]))

# ------------------------------------------------------------------ sync state (non-secret diagnostics only: error CODES, never messages)
def record_sync_ok(c, account_id, t=None):
    c.execute("UPDATE mail_accounts SET last_sync_at=?, last_error_code=NULL, consecutive_failures=0, next_poll_after=0 WHERE id=?", (db.now(), account_id))   # updated_at is NOT touched: it signals a credential/status change (provider cache key)

def record_error(c, account_id, code, n, next_after, status=None):
    c.execute("UPDATE mail_accounts SET last_error_code=?, last_error_at=?, consecutive_failures=?, next_poll_after=?, status=COALESCE(?,status) WHERE id=? AND status != 'disconnected'",
              (str(code)[:60], db.now(), n, next_after, status, account_id))

# ------------------------------------------------------------------ provider construction (one provider, one token source, one cursor key per account)
def _read_private(path):
    p = Path(path)
    if not p.is_file(): raise AccountError("credential_missing")
    if p.stat().st_mode & 0o077: raise AccountError("credential_file_permissions")
    v = p.read_text().strip()
    if not v: raise AccountError("credential_missing")
    return v

def provider_for(account, token_source=None, base=None):
    """-> a provider for ONE account. Gmail only (the project's default transport). Raises AccountError(code) when the stored credential or client configuration is unusable."""
    if account["provider"] != "gmail": raise AccountError("unsupported_provider")
    from .gmail_api import GmailApiProvider
    if token_source is None:
        cf = account.get("client_ref") or os.environ.get("HRBC_OAUTH_CLIENT_FILE")          # each account uses the OAuth client that issued ITS refresh token
        if not cf: raise AccountError("client_not_configured")
        from . import google_auth
        try: cl = google_auth.load_client(cf)
        except config.ConfigError: raise AccountError("client_not_configured")
        if not account["credential_ref"]: raise AccountError("credential_missing")
        token_source = oauth.RefreshTokenSource(os.environ.get("HRBC_OAUTH_TOKEN_URL") or cl["token_uri"], cl["client_id"], cl["client_secret"], _read_private(account["credential_ref"]), persist_path=account["credential_ref"])
    return GmailApiProvider(token_source=token_source, base=base, key=account["mailbox_key"])
