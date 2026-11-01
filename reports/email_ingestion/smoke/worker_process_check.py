"""LOCAL process-level check of the real CLI worker (`python -m ingest.run worker --imap`): starts it as a subprocess against the local IMAP stub (test mode, private test CA), waits for a message to be processed by the real models, checks `health`, tries to start a second worker (must be refused),
sends SIGTERM (must exit 0 cleanly) and verifies the lock and state. This is a LOCAL execution check only; it is not the production worker running against a real mailbox.
Usage: cd app && ../.venv/bin/python ../reports/email_ingestion/smoke/worker_process_check.py"""
import os, sys, json, time, signal, subprocess, tempfile, sqlite3
from pathlib import Path
from email.message import EmailMessage
ROOT = Path(__file__).resolve().parents[3]; TMP = Path(tempfile.mkdtemp(prefix="worker_chk_")); sys.path.insert(0, str(ROOT / "tests"))
import imap_stub as STUB
pki = STUB.make_pki(TMP / "pki"); jpg = next((ROOT / "datasets/synthetic_doctype/v1/images/rent_receipt").glob("*.jpg")).read_bytes()
m = EmailMessage(); m["Message-ID"] = "<w1@x>"; m["From"] = "asha@corp.example"; m["To"] = "expenses@corp.example"; m["Subject"] = "bill"; m["Authentication-Results"] = "mx.corp; dkim=pass header.d=corp.example"; m.set_content("bill attached"); m.add_attachment(jpg, maintype="image", subtype="jpeg", filename="rent.jpg")
srv = STUB.StubServer(pki["good"], [m]); port = srv.start()
(TMP / "pw").write_text("pw"); os.chmod(TMP / "pw", 0o600)
env = dict(os.environ, HRBC_ENV="test", HRBC_OUT=str(TMP / "out"), HRBC_INGEST_MAILBOX="expenses@corp.example", HRBC_INGEST_AUTHSERV_ID="mx.corp", HRBC_IMAP_HOST="localhost", HRBC_IMAP_PORT=str(port), HRBC_IMAP_USER="user", HRBC_IMAP_PASSWORD_FILE=str(TMP / "pw"),
           HRBC_IMAP_FOLDER="HRBC-TEST", HRBC_TEST_MAILBOX_USER="user", HRBC_TEST_FOLDER="HRBC-TEST", HRBC_IMAP_CA_FILE=pki["ca"], HRBC_IMAP_INITIAL="all", PYTHONPATH=str(ROOT / "app"))
APP = str(ROOT / "app"); PY = str(ROOT / ".venv/bin/python"); cli = lambda *a, **k: subprocess.run([PY, "-m", "ingest.run", *a], cwd=APP, env=env, capture_output=True, text=True, **k)
R = {}; R["check_config_rc"] = cli("check-config").returncode; cli("add-sender", "asha@corp.example", "E1", "--admin", "audit")
w = subprocess.Popen([PY, "-m", "ingest.run", "worker", "--imap", "--interval", "2"], cwd=APP, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
db = TMP / "out/hr/ingest.db"; t0 = time.time(); status = None
while time.time() - t0 < 120:
    time.sleep(2)
    if db.exists():
        try: status = sqlite3.connect(db).execute("SELECT status FROM emails").fetchone()
        except Exception: status = None
        if status and status[0] in ("COMPLETED", "REVIEW_REQUIRED", "FAILED"): break
R["email_status_after_worker_ran"] = status[0] if status else None; R["seconds_to_process"] = round(time.time() - t0, 1)
h = cli("health"); R["health_rc_while_running"] = h.returncode; R["health_while_running"] = json.loads(h.stdout)["status"]
second = cli("worker", "--imap", "--once"); R["second_worker_rc(3=refused)"] = second.returncode
w.send_signal(signal.SIGTERM)
try: R["worker_exit_code_after_SIGTERM"] = w.wait(timeout=60)
except subprocess.TimeoutExpired: w.kill(); R["worker_exit_code_after_SIGTERM"] = "TIMEOUT(killed)"
out = w.stdout.read(); R["log_lines"] = [l.split(" ", 2)[-1][:100] for l in out.splitlines() if "ingest" in l][:8]; R["clean_stop_logged"] = "worker stopped cleanly" in out
R["password_value_in_log"] = any(tok == "pw" or "pw" == tok.strip("\"',;:") for tok in out.split())
ws = sqlite3.connect(db).execute("SELECT stopped_at, last_poll_error FROM worker_state").fetchone(); R["stopped_at_recorded"] = bool(ws[0]); R["last_poll_error"] = ws[1]
R["lock_released(third_worker_can_start_once)"] = cli("worker", "--imap", "--once").returncode == 0
R["mailbox_commands"] = sorted(set(srv.log)); R["mailbox_modified"] = bool(srv.stored)
Path(__file__).with_name("worker_process_check_result.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))
