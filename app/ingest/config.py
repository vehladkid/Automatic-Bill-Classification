"""Environment-driven configuration and validation for the ingestion worker. Secrets are read from the environment or from *_FILE paths (mode 0600 enforced); values are NEVER echoed in errors, logs or health output.
HRBC_ENV selects the mode: prod (default, strictest) | test (dedicated test mailbox, explicit trust allowed) | dev (local development)."""
import os, re, stat
from pathlib import Path
ENVS = ("prod", "test", "dev")
class ConfigError(Exception): pass
def env_name(): return os.environ.get("HRBC_ENV", "prod").strip().lower()
def _csv(name): return [x.strip().lower() for x in os.environ.get(name, "").split(",") if x.strip()]
def secret(name):
    """value of NAME, or the contents of the file named by NAME_FILE (must not be readable by group/other). None when unset."""
    v = os.environ.get(name)
    if v: return v
    f = os.environ.get(name + "_FILE")
    if not f: return None
    p = Path(f)
    if not p.is_file(): raise ConfigError(f"{name}_FILE does not point to a file")
    if stat.S_IMODE(p.stat().st_mode) & 0o077: raise ConfigError(f"{name}_FILE must not be readable by group/other (chmod 600)")
    return p.read_text().strip() or None
def policy():
    """sender/recipient policy, evaluated at call time. require_auth may only be disabled outside prod."""
    return dict(mailbox=(os.environ.get("HRBC_INGEST_MAILBOX") or "").strip().lower(), aliases=_csv("HRBC_INGEST_MAILBOX_ALIASES"), authserv=(os.environ.get("HRBC_INGEST_AUTHSERV_ID") or "").strip().lower(),
                require_auth=os.environ.get("HRBC_INGEST_REQUIRE_AUTH", "1") != "0", allowed_domains=_csv("HRBC_INGEST_ALLOWED_DOMAINS"))
def imap_settings():
    return dict(host=os.environ.get("HRBC_IMAP_HOST"), user=os.environ.get("HRBC_IMAP_USER"), folder=os.environ.get("HRBC_IMAP_FOLDER", "INBOX"), port=int(os.environ.get("HRBC_IMAP_PORT", "993")),
                auth=os.environ.get("HRBC_IMAP_AUTH", "password").lower(), mark_seen=os.environ.get("HRBC_IMAP_MARK_SEEN", "0") == "1", initial=os.environ.get("HRBC_IMAP_INITIAL", "new_only"),
                ca_file=os.environ.get("HRBC_IMAP_CA_FILE"), batch=int(os.environ.get("HRBC_IMAP_BATCH", "50")))
def validate(imap=True, gmail=False):
    """-> list of problems (strings, never secret values). Empty list = configuration usable. Fail-closed: the worker refuses to start with any problem."""
    P = []; env = env_name(); pol = policy()
    if env not in ENVS: P.append(f"HRBC_ENV must be one of {ENVS}")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", pol["mailbox"] or ""): P.append("HRBC_INGEST_MAILBOX (the expense mailbox address) is required")
    for a in pol["aliases"]:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", a): P.append("HRBC_INGEST_MAILBOX_ALIASES contains an invalid address")
    if not pol["require_auth"] and env == "prod": P.append("HRBC_INGEST_REQUIRE_AUTH=0 is not allowed in prod")
    if pol["require_auth"] and not pol["authserv"]: P.append("HRBC_INGEST_AUTHSERV_ID (authserv-id written by YOUR receiving mail server) is required when sender authentication is enforced")
    if env in ("test", "dev") and not os.environ.get("HRBC_OUT"): P.append("HRBC_OUT must be set explicitly outside prod so test data cannot land in the production store")
    if env == "prod" and any(k.startswith("HRBC_TEST_") for k in os.environ): P.append("HRBC_TEST_* variables must not be set in prod")
    if imap:
        s = imap_settings()
        if not s["host"]: P.append("HRBC_IMAP_HOST is required")
        if not s["user"]: P.append("HRBC_IMAP_USER is required")
        if s["auth"] not in ("password", "xoauth2"): P.append("HRBC_IMAP_AUTH must be password or xoauth2")
        try:
            if s["auth"] == "password" and not secret("HRBC_IMAP_PASSWORD"): P.append("HRBC_IMAP_PASSWORD (or HRBC_IMAP_PASSWORD_FILE) is required")
            if s["auth"] == "xoauth2":
                if os.environ.get("HRBC_OAUTH_CLIENT_FILE"):
                    from ingest import google_auth; google_auth.load_client(os.environ["HRBC_OAUTH_CLIENT_FILE"])
                    if not secret("HRBC_OAUTH_REFRESH_TOKEN"): P.append("HRBC_OAUTH_REFRESH_TOKEN (or HRBC_OAUTH_REFRESH_TOKEN_FILE) is required - run: python -m ingest.run google-auth")
                else:
                    url = os.environ.get("HRBC_OAUTH_TOKEN_URL", "")
                    if not url: P.append("HRBC_OAUTH_TOKEN_URL is required")
                    elif not url.startswith("https://") and not (env != "prod" and re.match(r"http://(127\.0\.0\.1|localhost)[:/]", url)): P.append("HRBC_OAUTH_TOKEN_URL must be https")
                    for n in ("HRBC_OAUTH_CLIENT_ID", "HRBC_OAUTH_CLIENT_SECRET", "HRBC_OAUTH_REFRESH_TOKEN"):
                        if not secret(n): P.append(f"{n} (or {n}_FILE) is required")
        except ConfigError as e: P.append(str(e))
        if s["ca_file"] and env == "prod": P.append("HRBC_IMAP_CA_FILE (extra trust) is not allowed in prod")
        if s["ca_file"] and not Path(s["ca_file"]).is_file(): P.append("HRBC_IMAP_CA_FILE does not exist")
        if s["initial"] not in ("new_only", "all"): P.append("HRBC_IMAP_INITIAL must be new_only or all")
        if env == "test":   # dedicated test mailbox guard: the configured mailbox/folder must be explicitly declared as the test one
            if not os.environ.get("HRBC_TEST_MAILBOX_USER") or os.environ.get("HRBC_TEST_MAILBOX_USER") != s["user"]: P.append("test mode: HRBC_IMAP_USER must equal HRBC_TEST_MAILBOX_USER")
            if not os.environ.get("HRBC_TEST_FOLDER") or os.environ.get("HRBC_TEST_FOLDER") != s["folder"]: P.append("test mode: HRBC_IMAP_FOLDER must equal HRBC_TEST_FOLDER")
            if s["mark_seen"]: P.append("test mode: HRBC_IMAP_MARK_SEEN must be 0 (never modify mailbox state)")
    if gmail:
        cf = os.environ.get("HRBC_OAUTH_CLIENT_FILE")
        try:
            if not cf: P.append("HRBC_OAUTH_CLIENT_FILE (the downloaded Google client JSON, mode 600) is required")
            else:
                from ingest import google_auth; google_auth.load_client(cf)
            if not secret("HRBC_OAUTH_REFRESH_TOKEN"): P.append("HRBC_OAUTH_REFRESH_TOKEN_FILE is required - run: python -m ingest.run google-auth")
        except ConfigError as e: P.append(str(e))
        if env == "test" and (os.environ.get("HRBC_TEST_MAILBOX_USER") or "").strip().lower() != pol["mailbox"]: P.append("test mode: HRBC_TEST_MAILBOX_USER must equal HRBC_INGEST_MAILBOX")
    return P
def require_valid(imap=True):
    p = validate(imap)
    if p: raise ConfigError("; ".join(p))
