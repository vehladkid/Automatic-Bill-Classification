"""Multiple connected mailboxes: migration + legacy backfill, per-account credentials / cursors / status, ingestion from several inboxes, ownership, duplicates, failure isolation, refresh failures,
disconnect, retry / dead-letter, concurrent syncs, single-account regression, operator-only linking, tenant isolation. Local fake Gmail servers and FakeProvider only: no network, no Google, no real credentials."""
import json, os, sqlite3, stat, tempfile, threading, unittest, uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import _path
from test_ingest import Base, png, msg
from test_gmail_api import FakeGmail, Tok, mime
from ingest import accounts as AC, api, db, gmail_api as GM, oauth, pipeline as PL, run as RUN, google_auth as GA
from ingest.provider import FakeProvider, ProviderTransient
from tax.store import PermissionDenied

HR, E1, E2 = api.Principal("hr", "HR1"), api.Principal("employee", "E1"), api.Principal("employee", "E2")

class RevokedTok:
    scope = None
    def access_token(self): raise oauth.CredentialsExpired("invalid_grant")

class MA(Base):
    def setUp(self):
        super().setUp(); os.environ["HRBC_ACCOUNTS_DIR"] = os.path.join(self.out, "accounts"); self.addCleanup(os.environ.pop, "HRBC_ACCOUNTS_DIR", None); self.servers = []
    def mk(self, addr, owner="E1", token=None, tok=None, with_provider=True):
        a = AC.add_account(owner, "gmail", addr, addr, token or ("RT_" + addr), "admin"); row = AC.get(a["id"])
        g = FakeGmail(); base = g.start(); self.addCleanup(g.stop)
        p = GM.GmailApiProvider(token_source=tok or Tok(), base=base + "/gmail/v1/users/me", batch=5, key=row["mailbox_key"]) if with_provider else None
        return row, g, p
    def poll(self, p, row, **k): return PL.poll_once(p, account=row, initial=k.pop("initial", "all"), **k)
    def emails(self): return [dict(r) for r in db.conn().execute("SELECT * FROM emails ORDER BY id")]
    def sproc(self):
        seen = self.sources = []; base = self.processor()
        def proc(data, filename, owner, source): seen.append((owner, source)); return base(data, filename, owner, source)
        return proc

class Migration(MA):
    def old_db(self):
        """a database as the previous release left it: user_version 2, no mail_accounts, no account_id columns, one mailbox with history"""
        p = Path(self.out) / "hr" / "ingest.db"; p.parent.mkdir(parents=True, exist_ok=True); [q.unlink() for q in p.parent.glob("ingest.db*")]; c = sqlite3.connect(p)      # discard the fresh DB the base class made
        c.executescript("""CREATE TABLE emails(id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT, message_id TEXT, sender TEXT, subject TEXT, received_at TEXT, employee_id TEXT, status TEXT, reason TEXT, n_attachments INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT, body_signal TEXT, auth_evidence TEXT, UNIQUE(provider, message_id));
          CREATE TABLE poll_state(provider TEXT, mailbox_key TEXT, uidvalidity TEXT, last_uid INTEGER DEFAULT 0, updated_at TEXT, PRIMARY KEY(provider, mailbox_key));
          CREATE TABLE inbox_failures(key TEXT PRIMARY KEY, provider TEXT, uidvalidity TEXT, uid INTEGER, message_id TEXT, state TEXT, attempts INTEGER DEFAULT 0, max_attempts INTEGER DEFAULT 5, next_run REAL DEFAULT 0, error_code TEXT, first_seen TEXT, updated_at TEXT, resolved_at TEXT);
          INSERT INTO emails(provider,message_id,sender,subject,received_at,employee_id,status) VALUES('gmail','<a@x>','asha@corp.example','s','2026-10-01','E1','COMPLETED'),('gmail','<b@x>','ravi@corp.example','s','2026-10-02','E2','REJECTED');
          INSERT INTO poll_state VALUES('gmail','gmail:me','gmail',77,'2026-10-02');
          INSERT INTO inbox_failures(key,provider,uidvalidity,uid,state,attempts,error_code) VALUES('gmail:gmail:5','gmail','gmail',5,'retry',1,'provider_transient');
          PRAGMA user_version=2;"""); c.commit(); c.close()
    def test_existing_single_mailbox_database_is_migrated_and_backfilled(self):
        self.old_db(); c = db.conn()
        self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], len(db.MIGRATIONS))
        lg = c.execute("SELECT * FROM mail_accounts").fetchall(); self.assertEqual(len(lg), 1); lg = dict(lg[0])
        self.assertEqual((lg["is_legacy"], lg["provider"], lg["mailbox_key"], lg["status"], lg["credential_ref"], lg["public_id"]), (1, "gmail", "gmail:me", "active", "environment", "acct_legacy"))
        self.assertEqual([r[0] for r in c.execute("SELECT DISTINCT account_id FROM emails")], [lg["id"]]); self.assertEqual(c.execute("SELECT account_id FROM inbox_failures").fetchone()[0], lg["id"])
        self.assertEqual(db.get_cursor("gmail", "gmail:me")["last_uid"], 77)                                    # the existing cursor still resolves: the current mailbox keeps working
        self.assertEqual(c.execute("SELECT COUNT(*) FROM emails").fetchone()[0], 2)                                # nothing lost
    def test_migration_is_repeatable_and_idempotent(self):
        self.old_db(); db.conn(); db.conn(); db._backfill_legacy_account(db.conn()); db._backfill_legacy_account(db.conn())
        self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM mail_accounts").fetchone()[0], 1)
        c = db.conn(); c.execute("PRAGMA user_version=2"); db.conn(); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM mail_accounts").fetchone()[0], 1)     # re-running the migrations (duplicate-column tolerant) changes nothing
    def test_fresh_database_has_no_legacy_row_until_the_worker_registers_it(self):
        self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM mail_accounts").fetchone()[0], 0)
        fp = FakeProvider([]); a = AC.ensure_legacy(fp, "expenses@corp.example"); self.assertEqual((a["is_legacy"], a["mailbox_key"], a["address"]), (1, "fake", "expenses@corp.example")); self.assertEqual(AC.ensure_legacy(fp)["id"], a["id"])
    def test_rows_written_by_an_old_worker_after_the_upgrade_are_stamped_when_the_new_worker_starts(self):
        self.old_db(); db.conn(); c = db.conn(); c.execute("INSERT INTO emails(provider,message_id,sender,subject,employee_id,status) VALUES('gmail','<late@x>','asha@corp.example','s','E1','COMPLETED')")      # old code: no account_id column in its INSERT
        lg = AC.ensure_legacy(FakeProvider([])); self.assertEqual(c.execute("SELECT COUNT(*) FROM emails WHERE account_id IS NULL").fetchone()[0], 0); self.assertEqual(c.execute("SELECT account_id FROM emails WHERE message_id='<late@x>'").fetchone()[0], lg["id"])
    def test_failed_backfill_rolls_back(self):
        self.old_db(); c = db.conn(); c.execute("DELETE FROM mail_accounts"); c.execute("UPDATE emails SET account_id=NULL")
        with mock.patch.object(db, "now", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError): db._backfill_legacy_account(c)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM mail_accounts").fetchone()[0], 0); self.assertEqual(c.execute("SELECT COUNT(*) FROM emails WHERE account_id IS NULL").fetchone()[0], 2)

class TwoAccounts(MA):
    def test_two_connections_for_one_user_have_independent_credentials(self):
        a, _, _ = self.mk("a@gmail.com", token="REFRESH_TOKEN_A"); b, _, _ = self.mk("b@gmail.com", token="REFRESH_TOKEN_B")
        self.assertEqual((a["owner_employee_id"], b["owner_employee_id"]), ("E1", "E1")); self.assertNotEqual(a["credential_ref"], b["credential_ref"]); self.assertNotEqual(a["mailbox_key"], b["mailbox_key"])
        for r, v in ((a, "REFRESH_TOKEN_A"), (b, "REFRESH_TOKEN_B")): self.assertEqual(Path(r["credential_ref"]).read_text(), v); self.assertEqual(stat.S_IMODE(os.stat(r["credential_ref"]).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(Path(a["credential_ref"]).parent).st_mode), 0o700)
        raw = Path(self.out, "hr", "ingest.db").read_bytes() + b"".join(p.read_bytes() for p in Path(self.out, "hr").glob("ingest.db-*")); self.assertNotIn(b"REFRESH_TOKEN_A", raw); self.assertNotIn(b"REFRESH_TOKEN_B", raw)   # secrets are never in the database
        pv = AC.public_view(a); self.assertNotIn("credential_ref", pv); self.assertNotIn("REFRESH_TOKEN_A", json.dumps(pv)); self.assertEqual(len(AC.list_accounts()), 2)
    def test_adding_a_second_account_does_not_overwrite_the_first(self):
        a, _, _ = self.mk("a@gmail.com", token="RT1"); before = Path(a["credential_ref"]).read_text(); b, _, _ = self.mk("b@gmail.com", token="RT2")
        self.assertEqual(Path(a["credential_ref"]).read_text(), before); self.assertEqual(AC.get(a["id"])["status"], "active")
    def test_providers_are_built_per_account_from_their_own_stored_credential(self):
        a, _, _ = self.mk("a@gmail.com", token="RT_A", with_provider=False); b, _, _ = self.mk("b@gmail.com", token="RT_B", with_provider=False)
        cf = Path(self.out) / "client.json"; cf.write_text(json.dumps({"installed": {"client_id": "cid", "client_secret": "csec", "auth_uri": "https://x.example/auth", "token_uri": "https://x.example/token"}})); cf.chmod(0o600); os.environ["HRBC_OAUTH_CLIENT_FILE"] = str(cf); self.addCleanup(os.environ.pop, "HRBC_OAUTH_CLIENT_FILE", None)
        pa, pb = AC.provider_for(a), AC.provider_for(b); self.assertEqual((pa.tokens._refresh, pb.tokens._refresh), ("RT_A", "RT_B")); self.assertNotEqual(pa.mailbox_key(), pb.mailbox_key())
        self.assertEqual(GM.GmailApiProvider(token_source=Tok()).mailbox_key(), "gmail:me")                                                                   # legacy key unchanged
    def test_unusable_credential_files_are_reported_per_account(self):
        a, _, _ = self.mk("a@gmail.com", with_provider=False); os.chmod(a["credential_ref"], 0o644); cf = Path(self.out) / "c.json"; cf.write_text(json.dumps({"installed": {"client_id": "c", "client_secret": "s", "auth_uri": "https://x/a", "token_uri": "https://x/t"}})); cf.chmod(0o600)
        os.environ["HRBC_OAUTH_CLIENT_FILE"] = str(cf); self.addCleanup(os.environ.pop, "HRBC_OAUTH_CLIENT_FILE", None)
        with self.assertRaises(AC.AccountError) as e: AC.provider_for(a)
        self.assertEqual(e.exception.args[0], "credential_file_permissions"); os.environ.pop("HRBC_OAUTH_CLIENT_FILE")
        with self.assertRaises(AC.AccountError) as e: AC.provider_for(a)
        self.assertEqual(e.exception.args[0], "client_not_configured")
    def test_independent_cursors_and_ingestion_from_both_inboxes_with_colliding_provider_ids(self):
        a, ga, pa = self.mk("a@gmail.com"); b, gb, pb = self.mk("b@gmail.com")
        ga.add(0x100, mime("ma", [("a.png", "image/png", png(1))], rcpt="a@gmail.com")); gb.add(0x100, mime("mb", [("b.png", "image/png", png(2))], rcpt="b@gmail.com"))   # SAME Gmail id in two mailboxes, different messages
        self.assertEqual(self.poll(pa, a)["ingested"], 1); self.assertEqual(self.poll(pb, b)["ingested"], 1)
        cur = {r["mailbox_key"]: r["last_uid"] for r in db.conn().execute("SELECT * FROM poll_state")}; self.assertEqual(cur, {a["mailbox_key"]: 0x100, b["mailbox_key"]: 0x100})
        gb.add(0x101, mime("mb2", [("b2.png", "image/png", png(3))], rcpt="b@gmail.com")); self.assertEqual(self.poll(pa, a).get("ingested", 0), 0); self.assertEqual(self.poll(pb, b)["ingested"], 1)     # A's cursor did not move because of B
        cur = {r["mailbox_key"]: r["last_uid"] for r in db.conn().execute("SELECT * FROM poll_state")}; self.assertEqual((cur[a["mailbox_key"]], cur[b["mailbox_key"]]), (0x100, 0x101))
        em = self.emails(); self.assertEqual(sorted((e["message_id"], e["account_id"]) for e in em), sorted([("<ma@x>", a["id"]), ("<mb@x>", b["id"]), ("<mb2@x>", b["id"])]))
        self.assertEqual({e["employee_id"] for e in em}, {"E1"})
    def test_both_inboxes_reach_the_same_pipeline_with_correct_owner_and_source_account(self):
        a, ga, pa = self.mk("a@gmail.com"); b, gb, pb = self.mk("b@gmail.com")
        ga.add(1, mime("ma", [("a.png", "image/png", png(1))], rcpt="a@gmail.com")); gb.add(1, mime("mb", [("b.png", "image/png", png(2))], rcpt="b@gmail.com"))
        self.poll(pa, a); self.poll(pb, b); self.run_all(self.sproc())
        d = api.list_documents(E1); self.assertEqual(sorted(x["filename"] for x in d), ["a.png", "b.png"]); self.assertEqual({x["owner"] for x in d}, {"E1"})
        self.assertEqual(sorted((s["account_id"], o) for o, s in self.sources), sorted([(a["id"], "E1"), (b["id"], "E1")]))
        self.assertEqual({x["email"]["account_id"] for x in d}, {a["id"], b["id"]})
    def test_ownership_is_still_decided_by_the_sender_not_by_whose_inbox_it_is(self):
        a, ga, pa = self.mk("a@gmail.com", owner="E1")
        ga.add(1, mime("m1", [("x.png", "image/png", png(1))], sender="ravi@corp.example", rcpt="a@gmail.com")); ga.add(2, mime("m2", [("y.png", "image/png", png(2))], sender="stranger@evil.example", rcpt="a@gmail.com"))
        self.poll(pa, a); self.run_all(self.sproc()); self.assertEqual([o for o, _ in self.sources], ["E2"])                                       # E2's mail in E1's connected inbox belongs to E2; unknown sender rejected
        self.assertEqual(api.list_documents(E1), []); self.assertEqual(len(api.list_documents(E2)), 1); self.assertEqual({e["reason"] for e in self.emails() if e["status"] == "REJECTED"}, {"sender_not_authorised"})
    def test_connected_address_is_accepted_as_recipient_only_for_its_own_connection(self):
        a, ga, pa = self.mk("a@gmail.com"); b, gb, pb = self.mk("b@gmail.com")
        ga.add(1, mime("m1", [("x.png", "image/png", png(1))], rcpt="b@gmail.com"))                                                                  # addressed to B but found in A
        s = self.poll(pa, a); self.assertEqual(s["rejected"], 1); self.assertEqual(self.emails()[0]["reason"], "not_addressed_to_expense_mailbox")
        gb.add(1, mime("m2", [("y.png", "image/png", png(2))], rcpt="b@gmail.com")); self.assertEqual(self.poll(pb, b)["ingested"], 1)

class Duplicates(MA):
    def test_same_message_in_both_inboxes_is_stored_once_and_one_document_created(self):
        a, ga, pa = self.mk("a@gmail.com"); b, gb, pb = self.mk("b@gmail.com"); same = mime("shared", [("bill.png", "image/png", png(9))], rcpt="a@gmail.com")
        ga.add(0x10, same); gb.add(0x77, same.replace(b"a@gmail.com", b"b@gmail.com"))
        self.assertEqual(self.poll(pa, a)["ingested"], 1); s = self.poll(pb, b); self.assertEqual(s["duplicate"], 1); self.run_all(self.sproc())
        self.assertEqual(len(self.emails()), 1); self.assertEqual(len(api.list_documents(E1)), 1)
    def test_same_attachment_in_different_messages_is_not_processed_twice_per_employee(self):
        a, ga, pa = self.mk("a@gmail.com"); b, gb, pb = self.mk("b@gmail.com")
        ga.add(1, mime("m1", [("bill.png", "image/png", png(5))], rcpt="a@gmail.com")); gb.add(1, mime("m2", [("copy.png", "image/png", png(5))], rcpt="b@gmail.com"))
        self.poll(pa, a); self.poll(pb, b); self.run_all(self.sproc()); self.assertEqual(len(api.list_documents(E1)), 1); self.assertEqual([r[0] for r in db.conn().execute("SELECT state FROM attachments ORDER BY id")], ["COMPLETED", "DUPLICATE"])
    def test_repeated_polls_and_redelivery_create_nothing_new(self):
        a, ga, pa = self.mk("a@gmail.com"); ga.add(1, mime("m1", [("x.png", "image/png", png(1))], rcpt="a@gmail.com")); self.poll(pa, a); self.run_all(self.sproc())
        for _ in range(3): self.poll(pa, a)
        m = msg("<m1@x>", atts=[("x.png", "image/png", png(1))], rcpt="a@gmail.com"); m.provider = "gmail"; PL.ingest_message(m, AC.get(a["id"])); self.run_all(self.sproc()); self.assertEqual(len(self.emails()), 1); self.assertEqual(len(api.list_documents(E1)), 1)

class PerAccountClient(MA):
    def client(self, name, cid):
        f = Path(self.out) / name; f.write_text(json.dumps({"installed": {"client_id": cid, "client_secret": "sec-" + cid, "auth_uri": "https://x.example/a", "token_uri": "https://x.example/t-" + cid}})); f.chmod(0o600); return str(f)
    def test_accounts_authorised_by_different_oauth_clients_each_refresh_with_their_own_client(self):
        env = self.client("env.json", "ENVCLIENT"); new = self.client("new.json", "NEWCLIENT"); os.environ["HRBC_OAUTH_CLIENT_FILE"] = env; self.addCleanup(os.environ.pop, "HRBC_OAUTH_CLIENT_FILE", None)
        a, _, _ = self.mk("a@gmail.com", with_provider=False)                                                            # no client_ref: the environment's (old) client
        b = AC.add_account("E1", "gmail", "b@gmail.com", "b@gmail.com", "RT_B", "admin", client_ref=new); b = AC.get(b["id"])
        pa, pb = AC.provider_for(a), AC.provider_for(b); self.assertEqual((pa.tokens.cid, pb.tokens.cid), ("ENVCLIENT", "NEWCLIENT")); self.assertEqual((pa.tokens.url, pb.tokens.url), ("https://x.example/t-ENVCLIENT", "https://x.example/t-NEWCLIENT"))
        self.assertEqual((pa.tokens._refresh, pb.tokens._refresh), ("RT_a@gmail.com", "RT_B")); self.assertNotIn("client_ref", AC.public_view(b))
    def test_account_with_its_own_client_needs_no_environment_client(self):
        new = self.client("new.json", "NEWCLIENT"); os.environ.pop("HRBC_OAUTH_CLIENT_FILE", None); b = AC.add_account("E1", "gmail", "b@gmail.com", "b@gmail.com", "RT_B", "admin", client_ref=new)
        self.assertEqual(AC.provider_for(AC.get(b["id"])).tokens.cid, "NEWCLIENT")
    def test_reauthorising_with_a_new_client_updates_the_clients_reference_only(self):
        old = self.client("old.json", "OLD"); new = self.client("new.json", "NEW"); AC.add_account("E1", "gmail", "b@gmail.com", "b@gmail.com", "RT1", "admin", client_ref=old)
        b = AC.add_account("E1", "gmail", "b@gmail.com", "b@gmail.com", "RT2", "admin", client_ref=new); self.assertEqual(AC.get(b["id"])["client_ref"], new); self.assertEqual(len(AC.list_accounts()), 1)
        AC.add_account("E1", "gmail", "b@gmail.com", "b@gmail.com", "RT3", "admin"); self.assertEqual(AC.get(b["id"])["client_ref"], new)                 # no client given: keep the recorded one
    def test_mail_add_records_the_client_file_it_authorised_with(self):
        cf = self.client("new.json", "NEWCLIENT"); import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(GM.GmailApiProvider, "_get", return_value={"emailAddress": "n@gmail.com"}):
            rc = RUN.mail_add(SimpleNamespace(client_file=cf, login_hint=None, no_browser=True, employee="E1", admin="op"), authorize=lambda *a, **k: {"refresh_token": "RT", "scope": GA.SCOPE})
        self.assertEqual(rc, 0); self.assertEqual(AC.get("n@gmail.com" and AC.list_accounts()[0]["id"])["client_ref"], str(Path(cf).resolve()))

class Failures(MA):
    def test_one_failing_account_does_not_stop_the_other(self):
        a, ga, pa = self.mk("a@gmail.com"); b, gb, pb = self.mk("b@gmail.com"); ga.fail["status"] = 500; gb.add(1, mime("mb", [("b.png", "image/png", png(2))], rcpt="b@gmail.com"))
        res = PL.poll_accounts(factory=lambda acc: {a["id"]: pa, b["id"]: pb}[acc["id"]], now=1000, initial="all")
        self.assertEqual(res[a["public_id"]]["poll_error"], 1); self.assertEqual(res[b["public_id"]]["ingested"], 1)
        ra, rb = AC.get(a["id"]), AC.get(b["id"]); self.assertEqual((ra["last_error_code"], ra["consecutive_failures"], ra["status"]), ("provider_transient", 1, "active")); self.assertGreater(ra["next_poll_after"], 1000)
        self.assertEqual((rb["last_error_code"], rb["consecutive_failures"]), (None, 0)); self.assertIsNotNone(rb["last_sync_at"]); self.assertIsNone(ra["last_sync_at"])
        self.assertEqual(PL.poll_once(pa, account=a, now=1001)["skipped_backoff"], 1)                                                              # A backs off on its own
        ga.fail.clear(); ga.add(1, mime("ma", [("a.png", "image/png", png(1))], rcpt="a@gmail.com")); self.assertEqual(PL.poll_once(pa, account=a, now=1000 + 4000, initial="all")["ingested"], 1); self.assertEqual(AC.get(a["id"])["consecutive_failures"], 0)
    def test_factory_exception_and_unexpected_error_are_isolated(self):
        a, ga, pa = self.mk("a@gmail.com"); b, gb, pb = self.mk("b@gmail.com"); gb.add(1, mime("mb", [("b.png", "image/png", png(2))], rcpt="b@gmail.com"))
        def f(acc):
            if acc["id"] == a["id"]: raise RuntimeError("secret-looking detail xyz")
            return pb
        res = PL.poll_accounts(factory=f, now=10, initial="all"); self.assertEqual(res[a["public_id"]], {"account_error": 1}); self.assertEqual(res[b["public_id"]]["ingested"], 1)
        self.assertEqual(AC.get(a["id"])["last_error_code"], "RuntimeError"); self.assertNotIn("xyz", json.dumps(AC.get(a["id"]), default=str))                      # class name only
    def test_revoked_oauth_marks_only_that_account_needs_reauth_and_stops_polling_it(self):
        a, ga, pa = self.mk("a@gmail.com", tok=RevokedTok()); b, gb, pb = self.mk("b@gmail.com"); gb.add(1, mime("mb", [("b.png", "image/png", png(2))], rcpt="b@gmail.com")); calls = []
        fac = lambda acc: (calls.append(acc["id"]), {a["id"]: pa, b["id"]: pb}[acc["id"]])[1]
        res = PL.poll_accounts(factory=fac, now=1, initial="all"); self.assertEqual(res[a["public_id"]]["poll_error"], 1); self.assertEqual(res[b["public_id"]]["ingested"], 1)
        ra = AC.get(a["id"]); self.assertEqual((ra["status"], ra["last_error_code"]), ("needs_reauth", "credentials_rejected")); self.assertEqual(AC.get(b["id"])["status"], "active")
        h = PL.health(); self.assertIn("mail_account_needs_reauth", h["problems"]); self.assertEqual(h["mail_accounts"], {"needs_reauth": 1, "active": 1}); self.assertNotIn("@", json.dumps(h))
        res = PL.poll_accounts(factory=fac, now=99999, initial="all"); self.assertNotIn(a["public_id"], res)                                         # no longer polled until re-authorised
        AC.add_account("E1", "gmail", "a@gmail.com", "a@gmail.com", "RT_NEW", "admin"); ra = AC.get(a["id"]); self.assertEqual((ra["status"], ra["consecutive_failures"], ra["last_error_code"]), ("active", 0, None))
        self.assertEqual(Path(ra["credential_ref"]).read_text(), "RT_NEW"); self.assertEqual(len(AC.list_accounts()), 2)                                  # re-authorised in place, not duplicated
    def test_real_refresh_token_source_invalid_grant_is_a_permanent_account_error(self):
        import http.server
        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(s): s.rfile.read(int(s.headers.get("Content-Length", 0))); s.send_response(400); s.end_headers(); s.wfile.write(b'{"error":"invalid_grant"}')
            def log_message(s, *x): pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start(); self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        ts = oauth.RefreshTokenSource(f"http://127.0.0.1:{srv.server_port}/t", "cid", "csec", "RT"); a, ga, pa = self.mk("a@gmail.com", tok=ts)
        self.assertEqual(PL.poll_once(pa, account=a, initial="all")["poll_error"], 1); self.assertEqual(AC.get(a["id"])["status"], "needs_reauth")
    def test_retry_and_dead_letter_are_per_account_with_no_key_collision(self):
        a, b = self.mk("a@gmail.com", with_provider=False)[0], self.mk("b@gmail.com", with_provider=False)[0]; fa = FakeProvider([msg("<ma>", rcpt="a@gmail.com", atts=[("a.png", "image/png", png(1))])]); fb = FakeProvider([msg("<mb>", rcpt="b@gmail.com", atts=[("b.png", "image/png", png(2))])])
        fa.mailbox_key = lambda: a["mailbox_key"]; fb.mailbox_key = lambda: b["mailbox_key"]; fa.fail_fetch[1] = [ProviderTransient("x")] * 9                  # uid 1 exists in BOTH mailboxes; only A's fetch keeps failing
        with mock.patch.object(PL, "MAX_MSG_ATTEMPTS", 3):
            self.assertEqual(PL.poll_once(fa, account=a, now=1000, initial="all")["failed"], 1); self.assertEqual(PL.poll_once(fb, account=b, now=1000, initial="all")["ingested"], 1)
            keys = [r["key"] for r in db.conn().execute("SELECT key FROM inbox_failures")]; self.assertEqual(len(keys), 1); self.assertIn(a["public_id"], keys[0])
            PL.poll_once(fa, account=a, now=1000 + 100, initial="all"); st = PL.poll_once(fa, account=a, now=1000 + 100 + 1000, initial="all"); self.assertEqual(st["dead"], 1)
            row = dict(db.conn().execute("SELECT * FROM inbox_failures").fetchone()); self.assertEqual((row["state"], row["account_id"], row["attempts"]), ("dead", a["id"], 3))
            self.assertEqual(PL.requeue_failures("admin"), 1); fa.fail_fetch.clear(); st = PL.poll_once(fa, account=a, now=9000, initial="all"); self.assertEqual(st["recovered"], 1)
            self.assertEqual(db.conn().execute("SELECT state FROM inbox_failures").fetchone()[0], "resolved"); self.assertEqual(len([e for e in self.emails()]), 2)
    def test_failed_processing_job_is_retried_without_a_duplicate_document(self):
        a, ga, pa = self.mk("a@gmail.com"); ga.add(1, mime("m1", [("x.png", "image/png", png(1))], rcpt="a@gmail.com")); self.poll(pa, a)
        proc = self.processor(fail=[OSError("disk")]); self.assertEqual(PL.step(proc, retry_delay=0), "retry"); self.assertEqual(PL.step(proc, retry_delay=0), "COMPLETED"); self.assertEqual(len(api.list_documents(E1)), 1)

class Disconnect(MA):
    def test_disconnecting_one_account_keeps_the_other_and_the_history(self):
        a, ga, pa = self.mk("a@gmail.com"); b, gb, pb = self.mk("b@gmail.com"); ga.add(1, mime("ma", [("a.png", "image/png", png(1))], rcpt="a@gmail.com")); self.poll(pa, a); self.run_all(self.sproc())
        out = AC.disconnect(b["public_id"], "admin"); self.assertEqual(out["status"], "disconnected"); self.assertFalse(Path(b["credential_ref"]).exists()); self.assertTrue(Path(a["credential_ref"]).exists())
        self.assertIsNone(AC.get(b["id"])["credential_ref"]); self.assertEqual([x["public_id"] for x in AC.runnable()], [a["public_id"]]); self.assertEqual(PL.poll_once(pb, account=b)["skipped_inactive"], 1)
        ga.add(2, mime("ma2", [("a2.png", "image/png", png(2))], rcpt="a@gmail.com")); self.assertEqual(self.poll(pa, a)["ingested"], 1); self.assertEqual(len(self.emails()), 2); self.assertEqual(AC.disconnect(b["id"], "admin")["status"], "disconnected")
        c, _, _ = self.mk("b@gmail.com", token="RT_B2", with_provider=False); self.assertNotEqual(c["id"], b["id"]); self.assertEqual(c["status"], "active")                           # the address can be connected again afterwards
    def test_legacy_and_unknown_connections_cannot_be_disconnected(self):
        lg = AC.ensure_legacy(FakeProvider([]), "expenses@corp.example")
        with self.assertRaises(AC.AccountError): AC.disconnect(lg["id"], "admin")
        with self.assertRaises(AC.AccountError): AC.disconnect("acct_nope", "admin")

class Concurrency(MA):
    def test_concurrent_syncs_of_the_same_and_of_different_accounts_create_no_duplicates(self):
        a, ga, pa = self.mk("a@gmail.com"); b, gb, pb = self.mk("b@gmail.com")
        for i in range(6): ga.add(0x10 + i, mime(f"ma{i}", [(f"a{i}.png", "image/png", png(i))], rcpt="a@gmail.com")); gb.add(0x10 + i, mime(f"mb{i}", [(f"b{i}.png", "image/png", png(50 + i))], rcpt="b@gmail.com"))
        pa.batch = pb.batch = 50; errs = []
        def w(p, row):
            try: PL.poll_once(p, account=row, initial="all")
            except Exception as e: errs.append(repr(e))
        th = [threading.Thread(target=w, args=(p, r)) for p, r in ((pa, a), (pa, a), (pb, b), (pb, b), (pa, a))]; [t.start() for t in th]; [t.join() for t in th]
        self.assertEqual(errs, []); self.assertEqual(len(self.emails()), 12); self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 12); self.assertEqual(db.conn().execute("SELECT COUNT(DISTINCT key) FROM jobs").fetchone()[0], 12)
        self.run_all(self.processor()); self.assertEqual(len(api.list_documents(E1)), 12)

class Regression(MA):
    def test_legacy_mailbox_path_is_unchanged_and_stamped_with_the_legacy_account(self):
        fp = FakeProvider([]); lg = AC.ensure_legacy(fp, "expenses@corp.example"); fp.add(msg("<m1>", atts=[("x.png", "image/png", png(1))]))
        self.assertEqual(PL.poll_once(fp, account=lg, initial="all")["ingested"], 1); self.run_all(self.processor()); self.assertEqual(self.emails()[0]["account_id"], lg["id"])
        h = PL.health(); self.assertEqual(h["last_poll_error"], None); self.assertEqual(h["mail_accounts"], {}); self.assertNotIn("mail_account_poll_error", h["problems"])
        self.assertIsNotNone(AC.get(lg["id"])["last_sync_at"]); self.assertEqual(len(api.list_documents(E1)), 1)
    def test_no_account_argument_behaves_exactly_as_before(self):
        fp = FakeProvider([msg("<m1>", atts=[("x.png", "image/png", png(1))])]); self.assertEqual(PL.poll_once(fp, initial="all")["ingested"], 1); self.assertIsNone(self.emails()[0]["account_id"])
        self.assertEqual(PL.ingest_message(msg("<m2>", atts=[("y.png", "image/png", png(2))]))["status"], "QUEUED")
    def test_worker_loop_polls_legacy_and_connected_accounts_then_processes_one_job_at_a_time(self):
        os.environ["HRBC_IMAP_INITIAL"] = "all"; self.addCleanup(os.environ.pop, "HRBC_IMAP_INITIAL", None); b, gb, pb = self.mk("b@gmail.com"); gb.add(1, mime("mb", [("b.png", "image/png", png(2))], rcpt="b@gmail.com"))
        fp = FakeProvider([msg("<ml>", atts=[("l.png", "image/png", png(1))])]); svc = SimpleNamespace(process=self.sproc())
        rc = RUN.run_worker(SimpleNamespace(imap=False, gmail=False, interval=1, once=True), svc=svc, prov=fp, stop_flag=lambda: False, provider_factory=lambda acc: pb)
        self.assertEqual(rc, 0); self.assertEqual(sorted(x["filename"] for x in api.list_documents(E1)), ["b.png", "l.png"])                       # legacy + connected inbox both reached the same processor
        em = {e["message_id"]: e["account_id"] for e in self.emails()}; lg = db.conn().execute("SELECT id FROM mail_accounts WHERE is_legacy=1").fetchone()[0]; self.assertEqual(em, {"<ml>": lg, "<mb@x>": b["id"]})
        self.assertEqual(len(self.sources), 2)                                                                                                        # processed through the ONE existing drain loop

class Security(MA):
    def test_mailbox_connections_are_hr_only_and_expose_no_secrets(self):
        a, _, _ = self.mk("a@gmail.com", token="SECRETREFRESH")
        for p in (E1, E2): self.assertRaises(PermissionDenied, api.list_mail_accounts, p)
        out = api.list_mail_accounts(HR); self.assertEqual(len(out), 1); self.assertNotIn("SECRETREFRESH", json.dumps(out)); self.assertNotIn("credential_ref", out[0]); self.assertNotIn(self.out, json.dumps(out))
    def test_the_application_api_exposes_no_way_to_link_or_remove_a_mailbox(self):
        self.assertEqual([n for n in dir(api) if any(k in n for k in ("connect", "link_mail", "add_mail", "disconnect", "oauth"))], [])
    def test_a_mailbox_cannot_be_connected_for_a_second_user(self):
        AC.add_account("E1", "gmail", "a@gmail.com", "a@gmail.com", "RT1", "admin"); n = len(AC.list_accounts())
        with self.assertRaises(AC.AccountError): AC.add_account("E2", "gmail", "A@gmail.com", "a@gmail.com", "RT2", "admin")
        self.assertEqual(len(AC.list_accounts()), n); self.assertEqual(Path(AC.list_accounts()[0]["credential_ref"]).read_text(), "RT1")
    def test_invalid_input_is_refused(self):
        for emp in ("", "E1;DROP", "../x"):
            with self.assertRaises(AC.AccountError): AC.add_account(emp, "gmail", "a@gmail.com", "a@gmail.com", "RT", "admin")
        with self.assertRaises(AC.AccountError): AC.add_account("E1", "gmail", "", "", "RT", "admin")
        with self.assertRaises(AC.AccountError): AC.add_account("E1", "gmail", "a@gmail.com", "a@gmail.com", "", "admin")
        self.assertEqual(AC.list_accounts(), [])
    def _cli(self, authorize):
        cf = Path(self.out) / "g.json"; cf.write_text(json.dumps({"installed": {"client_id": "c", "client_secret": "s", "auth_uri": "https://x.example/a", "token_uri": "https://x.example/t"}})); cf.chmod(0o600)
        import io, contextlib; so, se = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se), mock.patch.object(GM.GmailApiProvider, "_get", return_value={"emailAddress": "New@Gmail.com"}):
            rc = RUN.mail_add(SimpleNamespace(client_file=str(cf), login_hint=None, no_browser=True, employee="E1", admin="op"), authorize=authorize)
        return rc, so.getvalue(), se.getvalue()
    def test_operator_linking_stores_a_private_per_account_credential_and_never_prints_it(self):
        first, _, _ = self.mk("a@gmail.com", token="FIRST_RT"); rc, so, se = self._cli(lambda *a, **k: {"refresh_token": "NEW_SECRET_RT", "scope": GA.SCOPE})
        self.assertEqual(rc, 0); self.assertNotIn("NEW_SECRET_RT", so + se); new = [x for x in AC.list_accounts() if x["address"] == "new@gmail.com"][0]; self.assertEqual(Path(new["credential_ref"]).read_text(), "NEW_SECRET_RT")
        self.assertEqual(Path(first["credential_ref"]).read_text(), "FIRST_RT"); self.assertEqual(stat.S_IMODE(os.stat(new["credential_ref"]).st_mode), 0o600)
        rc2, _, se2 = self._cli(lambda *a, **k: {"refresh_token": "OTHER", "scope": GA.SCOPE}); self.assertEqual(rc2, 0)                                       # same user + same mailbox = re-authorisation
        self.assertEqual(len([x for x in AC.list_accounts() if x["address"] == "new@gmail.com"]), 1)
    def test_failed_or_tampered_consent_creates_nothing(self):
        def bad(*a, **k): raise GA.AuthError("State mismatch: the response did not come from this authorization request. Nothing was stored.")
        rc, so, se = self._cli(bad); self.assertEqual(rc, 2); self.assertEqual(AC.list_accounts(), []); self.assertFalse(Path(self.out, "accounts").exists() and list(Path(self.out, "accounts").glob("*")))
    def test_oauth_scope_is_unchanged_read_only(self):
        self.assertEqual(GA.SCOPE, "https://www.googleapis.com/auth/gmail.readonly"); self.assertEqual(GM.SCOPE, GA.SCOPE)
    def test_users_cannot_see_each_others_mail_or_documents(self):
        a, ga, pa = self.mk("a@gmail.com", owner="E1"); b, gb, pb = self.mk("b@gmail.com", owner="E2")
        ga.add(1, mime("m1", [("x.png", "image/png", png(1))], rcpt="a@gmail.com")); gb.add(1, mime("m2", [("y.png", "image/png", png(2))], sender="ravi@corp.example", rcpt="b@gmail.com"))
        self.poll(pa, a); self.poll(pb, b); self.run_all(self.processor()); d1, d2 = api.list_documents(E1), api.list_documents(E2)
        self.assertEqual(([x["filename"] for x in d1], [x["filename"] for x in d2]), (["x.png"], ["y.png"])); self.assertEqual([e["sender"] for e in api.list_emails(E2)], ["ravi@corp.example"])
        self.assertRaises(PermissionDenied, api.get_document, E2, d1[0]["doc_id"]); self.assertRaises(PermissionDenied, api.email_detail, E1, api.list_emails(E2)[0]["id"])

if __name__ == "__main__": unittest.main()
