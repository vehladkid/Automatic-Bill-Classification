"""Background runner and operator commands.   Run from the repo root:  cd app && ../.venv/bin/python -m ingest.run <command>
  worker [--imap | --gmail] [--interval 60] [--once]   poll the mailbox (if --imap) and process queued jobs; refuses to start on any configuration problem; single instance (lock); SIGTERM/SIGINT = finish the current job, then exit
  webhook [--port 8765]                      signed-webhook receiver (enqueue only)
  gmail-check                                read-only Gmail API status (profile counters + up to 5 INBOX ids; no content)
  check-config [--no-imap] [--gmail]                   validate environment variables (never prints values)           exit 0 ok / 2 problems
  health [--interval 60]                     privacy-safe JSON health snapshot                               exit 0 ok / 1 degraded or down
  check-perms [--fix]                        audit (or repair) permissions of everything under HRBC_OUT       exit 0 ok / 1 unsafe
  failures | requeue-dead [--key K] --admin NAME   list dead-lettered messages/fetch jobs; return them to retry
  add-sender EMAIL EMPLOYEE_ID --admin NAME  explicit authorisation of a sender address (aliases = more rows for the same employee)
  google-auth [--user you@gmail.com] [--no-browser]  one-time Google consent in the browser; stores the refresh token (0600) outside the repo
  mail-list                                  connected mailboxes (id, owner, status, last sync, error code); no credentials
  mail-add --employee ID --admin NAME [--no-browser] [--login-hint a@b]   OPERATOR action: Google consent for an ADDITIONAL Gmail mailbox; stores its own 0600 credential; never touches other accounts
  mail-disconnect ID --admin NAME            stop polling that mailbox and delete its stored credential (history kept)
  purge --days N                             delete leftover raw copies older than N days
Credentials only via environment variables or *_FILE paths (mode 0600); see config/env.example and docs/EMAIL_INGESTION_OPERATIONS.md."""
import sys, os, time, json, argparse, logging, signal, fcntl
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "app", ROOT / "scripts", ROOT / "scripts" / "taskB"): sys.path.insert(0, str(p))
from ingest import pipeline as PL, webhook as WH, db, config, perms, accounts as AC

def acquire_lock():
    p = db.out_dir() / "hr" / "worker.lock"; perms.private_dir(p.parent); fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o600)
    try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: os.close(fd); return None
    os.ftruncate(fd, 0); os.write(fd, str(os.getpid()).encode()); return fd

def run_worker(args, svc=None, prov=None, stop_flag=None, provider_factory=None):
    perms.apply_umask(); problems = config.validate(imap=False, gmail=True) if getattr(args, "gmail", False) else config.validate(imap=args.imap)
    if problems:
        for pr in problems: print("CONFIG ERROR:", pr, file=sys.stderr)
        return 2
    lock = acquire_lock()
    if lock is None: print("another worker already holds the lock; refusing to start a second one", file=sys.stderr); return 3
    stop = {"v": False}; sig = lambda *a: stop.__setitem__("v", True)
    if stop_flag is None: signal.signal(signal.SIGTERM, sig); signal.signal(signal.SIGINT, sig)
    is_stopped = (lambda: stop["v"]) if stop_flag is None else stop_flag
    if svc is None:
        import service as SV; svc = SV.BillService().load()
    if prov is None and getattr(args, "gmail", False):
        from ingest.gmail_api import GmailApiProvider; prov = GmailApiProvider()
    elif prov is None and args.imap:
        from ingest.provider import ImapProvider; prov = ImapProvider()
    log = logging.getLogger("ingest"); c = db.conn(); PL.worker_state(c)
    legacy = AC.ensure_legacy(prov, config.policy()["mailbox"] or None) if prov is not None else None; cache = {}
    c.execute("UPDATE worker_state SET pid=?, started_at=?, stopped_at=NULL WHERE id=1", (os.getpid(), db.now())); log.info("worker started (imap=%s, interval=%ss)", bool(prov), args.interval)
    while not is_stopped():
        try:
            c.execute("UPDATE worker_state SET last_loop_at=? WHERE id=1", (time.time(),))
            if prov: PL.poll_once(prov, account=legacy)
            PL.poll_accounts(provider_factory, cache)                                # each connected account: own provider, credential, cursor, backoff; failures isolated. Ingestion only: processing below is unchanged (one job at a time)
            PL.drain(svc.process, prov, stop=is_stopped)
        except Exception as e: log.error("loop error: %s", type(e).__name__)       # class name only; the loop keeps running and health reports it
        if args.once: break
        for _ in range(max(1, args.interval)):                                       # sleep in 1 s slices so a stop signal is honoured promptly
            if is_stopped(): break
            time.sleep(1)
    c.execute("UPDATE worker_state SET stopped_at=? WHERE id=1", (str(time.time()),)); log.info("worker stopped cleanly"); os.close(lock); return 0

def mail_add(args, authorize=None):
    """OPERATOR linking of an additional Gmail mailbox for --employee. No web surface: the application has no authentication, so only someone with shell access to the host (who can already read worker.env) can do this.
    The consent flow is the existing PKCE/state/scope-checked one (gmail.readonly, unchanged). The refresh token goes to a new per-account 0600 file; the first account's credential is never read or replaced."""
    from ingest import google_auth as GA, gmail_api as GM, oauth
    cf = args.client_file or os.environ.get("HRBC_OAUTH_CLIENT_FILE") or Path.home() / ".config" / "hrbc" / "google_client.json"
    try:
        tok = (authorize or GA.authorize_tokens)(cf, login_hint=args.login_hint, open_browser=not args.no_browser)
        cl = GA.load_client(cf); ts = oauth.RefreshTokenSource(cl["token_uri"], cl["client_id"], cl["client_secret"], tok["refresh_token"])
        addr = str(GM.GmailApiProvider(token_source=ts)._get("/profile").get("emailAddress") or "").strip().lower()
        if not addr: raise AC.AccountError("Google did not report the mailbox address")
        a = AC.add_account(args.employee, "gmail", addr, addr, tok["refresh_token"], args.admin, client_ref=str(Path(cf).expanduser().resolve()))
    except (GA.AuthError, config.ConfigError, AC.AccountError) as e: print("ERROR:", e, file=sys.stderr); return 2
    except Exception as e: print("ERROR: could not complete the connection:", getattr(e, "code", type(e).__name__), file=sys.stderr); return 2
    print(json.dumps({"connected": AC.public_view(a)}, default=str))
    return 0

def main(argv=None):
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("worker"); w.add_argument("--imap", action="store_true"); w.add_argument("--gmail", action="store_true"); w.add_argument("--interval", type=int, default=60); w.add_argument("--once", action="store_true")
    h = sp.add_parser("webhook"); h.add_argument("--port", type=int, default=8765); h.add_argument("--host", default="127.0.0.1")
    a = sp.add_parser("add-sender"); a.add_argument("email"); a.add_argument("employee_id"); a.add_argument("--admin", required=True)
    u = sp.add_parser("purge"); u.add_argument("--days", type=int, required=True)
    cc = sp.add_parser("check-config"); cc.add_argument("--no-imap", action="store_true"); cc.add_argument("--gmail", action="store_true")
    hh = sp.add_parser("health"); hh.add_argument("--interval", type=int, default=60)
    cp = sp.add_parser("check-perms"); cp.add_argument("--fix", action="store_true")
    ga = sp.add_parser("google-auth"); ga.add_argument("--client-file"); ga.add_argument("--token-file"); ga.add_argument("--user"); ga.add_argument("--no-browser", action="store_true"); ga.add_argument("--imap-scope", action="store_true")
    sp.add_parser("gmail-check")
    sp.add_parser("mail-list")
    ma = sp.add_parser("mail-add"); ma.add_argument("--employee", required=True); ma.add_argument("--admin", required=True); ma.add_argument("--no-browser", action="store_true"); ma.add_argument("--login-hint"); ma.add_argument("--client-file")
    md = sp.add_parser("mail-disconnect"); md.add_argument("account"); md.add_argument("--admin", required=True)
    sp.add_parser("failures")
    rq = sp.add_parser("requeue-dead"); rq.add_argument("--key"); rq.add_argument("--admin", required=True)
    args = ap.parse_args(argv); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.cmd == "google-auth":
        from ingest import google_auth as GA
        home = Path.home() / ".config" / "hrbc"
        try: GA.authorize(args.client_file or os.environ.get("HRBC_OAUTH_CLIENT_FILE") or home / "google_client.json", args.token_file or os.environ.get("HRBC_OAUTH_REFRESH_TOKEN_FILE") or home / "google_refresh_token", args.user, scope=GA.IMAP_SCOPE if args.imap_scope else GA.SCOPE, open_browser=not args.no_browser)
        except (GA.AuthError, config.ConfigError) as e: print("ERROR:", e, file=sys.stderr); return 2
        return 0
    if args.cmd == "mail-list":
        for a in AC.list_accounts(): print(json.dumps(AC.public_view(a), default=str))
        return 0
    if args.cmd == "mail-disconnect":
        try: print(json.dumps(AC.public_view(AC.disconnect(args.account, args.admin)), default=str)); return 0
        except AC.AccountError as e: print("ERROR:", e, file=sys.stderr); return 2
    if args.cmd == "mail-add": return mail_add(args)
    if args.cmd == "gmail-check":
        from ingest import gmail_api as GM
        p = config.validate(imap=False, gmail=True)
        if p: print(json.dumps({"ok": False, "problems": p}, indent=1)); return 2
        try:
            r = GM.check(); g = GM.granted_scope(oauth_tokens := __import__("ingest.oauth", fromlist=["x"]).RefreshTokenSource.from_env())
            r["scopes_granted_to_stored_token"] = g
            r["least_privilege"] = (None if g is None else set(g) <= {GM.SCOPE, "openid", "https://www.googleapis.com/auth/userinfo.email"})
            if r["least_privilege"] is False: r["action"] = "stored token is broader than needed: revoke it at myaccount.google.com/permissions, then run: python -m ingest.run google-auth"
            print(json.dumps(dict(ok=True, **r), indent=1)); return 0
        except Exception as e: print(json.dumps({"ok": False, "error": getattr(e, "code", type(e).__name__), "detail": str(e)[:160]}, indent=1)); return 1
    if args.cmd == "add-sender": PL.add_sender(args.email, args.employee_id, args.admin); print("ok"); return 0
    if args.cmd == "purge": print("deleted", PL.purge_inbox(args.days)); return 0
    if args.cmd == "check-config":
        p = config.validate(imap=not args.no_imap and not args.gmail, gmail=getattr(args, "gmail", False)); print(json.dumps({"ok": not p, "problems": p, "env": config.env_name()}, indent=1)); return 0 if not p else 2
    if args.cmd == "health": h_ = PL.health(interval=args.interval); print(json.dumps(h_, indent=1)); return 0 if h_["status"] == "ok" else 1
    if args.cmd == "check-perms":
        root = db.out_dir()
        if args.fix: print("fixed", perms.fix(root), "paths")
        bad = perms.audit(root); print(json.dumps({"root": str(root), "unsafe": len(bad), "examples": bad[:10]}, indent=1)); return 0 if not bad else 1
    if args.cmd == "failures":
        for r in db.conn().execute("SELECT key,state,attempts,max_attempts,error_code,first_seen,updated_at FROM inbox_failures ORDER BY state DESC, first_seen"): print(json.dumps(dict(r)))
        return 0
    if args.cmd == "requeue-dead": print("requeued", PL.requeue_failures(args.admin, args.key)); return 0
    if args.cmd == "webhook":
        problems = [x for x in config.validate(imap=False) if "IMAP" not in x]
        if problems or not os.environ.get("HRBC_WEBHOOK_SECRET"): print("CONFIG ERROR:", problems or "HRBC_WEBHOOK_SECRET is required", file=sys.stderr); return 2
        print(f"webhook receiver on {args.host}:{args.port} (enqueue only; put TLS in front)"); WH.serve(args.host, args.port); return 0
    if args.cmd == "worker": return run_worker(args)
if __name__ == "__main__": sys.exit(main())
