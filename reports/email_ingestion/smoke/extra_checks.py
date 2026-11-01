"""AUDIT extra checks (no models, temp HRBC_OUT, no network): webhook->IMAP fetch path, SPF/DKIM alignment, recipient check when HRBC_INGEST_MAILBOX is unset, file permissions. Usage: cd app && ../.venv/bin/python ../reports/email_ingestion/smoke/extra_checks.py"""
import os, sys, json, tempfile, stat, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; T = Path(tempfile.mkdtemp()); os.environ["HRBC_OUT"] = str(T)
for p in ("app", "scripts", "scripts/taskB"): sys.path.insert(0, str(ROOT / p))
from ingest import pipeline as PL, webhook as WH, db
from ingest.provider import Message, Attachment, ImapProvider
R = {}
PL.add_sender("asha@corp.example", "E1", "audit")
# 1 webhook -> fetch job with the (only real) provider
code, msg = WH.handle(json.dumps({"event_id": "e1", "message_id": "<w1@x>"}).encode(), {"X-Timestamp": str(int(time.time())), "X-Signature": WH.sign("s", str(int(time.time())), json.dumps({"event_id": "e1", "message_id": "<w1@x>"}).encode())}, "s"); R["webhook_accepts_signed_event"] = (code, msg)
prov = ImapProvider("127.0.0.1", "u", "p")
out = []
for _ in range(5):
    r = PL.step(lambda *a: None, prov, retry_delay=0); out.append(r)
    if r is None: break
    c = db.conn(); c.execute("UPDATE jobs SET next_run=0")
R["webhook_fetch_job_with_ImapProvider_outcomes"] = out; R["fetch_job_final_state"] = [dict(r) for r in db.conn().execute("SELECT key,state,attempts,last_error FROM jobs")]
R["emails_created_from_webhook_event"] = db.conn().execute("SELECT COUNT(*) FROM emails").fetchone()[0]
# 2 SPF/DKIM alignment: pass for an attacker's own domain while From claims the employee
forged = Message("fake", "<f1@x>", "asha@corp.example", "expenses@corp.example", "s", "", [], "mx.corp; dkim=pass header.d=evil.example; spf=pass smtp.mailfrom=bounce@evil.example")
os.environ["HRBC_INGEST_MAILBOX"] = "expenses@corp.example"; R["forged_From_with_auth_pass_for_other_domain_accepted_as_employee"] = PL.authorise(forged)
# 3 recipient check when mailbox unset
del os.environ["HRBC_INGEST_MAILBOX"]; other = Message("fake", "<f2@x>", "asha@corp.example", "someone.else@corp.example", "s", "", [], "dkim=pass"); R["any_recipient_accepted_when_HRBC_INGEST_MAILBOX_unset"] = PL.authorise(other)
# 4 permissions
dbp = T / "hr" / "ingest.db"; R["ingest_db_mode"] = oct(stat.S_IMODE(dbp.stat().st_mode)); R["inbox_dir_mode"] = oct(stat.S_IMODE(PL.inbox_dir().stat().st_mode))
R["worker_loop_error_handling"] = "run.py logs exception class name only and sleeps; no alert, no counter"
print(json.dumps(R, indent=1, default=str)); Path(__file__).with_name("extra_checks_result.json").write_text(json.dumps(R, indent=1, default=str))
