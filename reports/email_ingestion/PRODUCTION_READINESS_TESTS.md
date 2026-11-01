# Production-readiness tests: commands, counts, limitations

## Commands and results (2026-10-19)
| Command (from `tests/` unless noted) | Result |
|---|---|
| `../.venv/bin/python -m unittest discover -p "test_*.py"` | **387 tests, OK** (baseline before the work: 312 OK) |
| `../.venv/bin/python -m unittest test_ingest_hardening` | **75 new tests, OK** (24 s) |
| `../.venv/bin/python -m unittest test_ingest` | 38 tests OK (updated setup: mailbox + authserv now required; same assertions) |
| `cd app && ../.venv/bin/python ../reports/email_ingestion/smoke/local_imap_smoke.py` | completes: 8 messages from a local IMAP stub over verified TLS -> 5 ingested, 2 rejected (unknown sender; From claim backed only by an attacker-domain pass), 1 duplicate Message-ID; real OCR/Task A/Task B processed 2 documents (`COMPLETED`, `REVIEW_REQUIRED`); zip rejected; body-only bill `REVIEW_REQUIRED/possible_bill_in_body`; forwarded PDF found and de-duplicated; second poll ingested nothing; mailbox commands sent: CAPABILITY, EXAMINE, LOGIN, LOGOUT, UID FETCH, UID SEARCH (nothing modified); 0 permission violations under `HRBC_OUT`; poison message (pdfinfo missing) failed alone, later valid message ingested, recovered after backoff; 13 s |
| `cd app && ../.venv/bin/python ../reports/email_ingestion/smoke/worker_process_check.py` | the real CLI worker as a subprocess: `check-config` rc 0; message processed in 14 s; `health` ok; second worker refused (rc 3); SIGTERM -> exit 0, clean stop logged, stop recorded, lock released; password not in the log; mailbox not modified |
| `systemd-analyze verify deploy/hrbc-ingest-worker.service` | no errors (unit not installed) |
| `cd app && ../.venv/bin/python -m ingest.run check-perms [--fix]` on the real `outputs/` | before: 109 unsafe paths; `--fix` repaired 108; after: 0 |
| `sha256sum datasets/processed/taskB/test.jsonl`; `sha256sum -c reports/taskB/v1_snapshot/SHA256SUMS.txt` | `fd8bafb3ad13...` unchanged; 34/34 OK |

## The 75 new tests by area (vs the verification list in the task)
Verified TLS 9 (1) · OAuth/auth 10 (1,5) · Authorisation 14 (2,3) · Resilience 11 (4,5,6,8) · Webhook 7 (6) · IMAP cursor flow 2 (4,6) · Body/format 8 (7) · Permissions 5 (9) · Operation 9 (health, config, lock, shutdown). Smoke tests cover item 10.
Mutation checks (the fix is removed and the tests must fail): unverified TLS context -> 4/4 TLS tests fail; no per-message isolation -> 2/2 resilience tests fail; alignment check disabled -> 2/2 authorisation tests fail.

## Mock versus real
| Component | In tests |
|---|---|
| IMAP client code, TLS verification (real `ssl`), PKI (real openssl-issued certs, expired and mismatched included), XOAUTH2 SASL | **real code against a local IMAP server stub** (`tests/imap_stub.py`, 127.0.0.1) |
| OAuth refresh flow | **real code against a local HTTP token endpoint stub** |
| SQLite persistence, queue, leases, cursor, permissions, parser, validators | real |
| OCR, Task A, Task B, profile, field report | real models in the smoke and worker-process checks; **stub processor** in the unit tests |
| Provider for resilience/cursor/webhook tests | `FakeProvider` with failure injection (and the IMAP stub where noted) |
| `pdfinfo` missing | simulated by patching `subprocess.run` |
| Gmail, Microsoft 365, any real mailbox, real MTA `Authentication-Results`, real throttling responses, Google/Microsoft token endpoints | **not tested** |
| Streamlit UI | the new HR failure panel is not exercised by a browser; `ingest.api` functions are tested |

## Limitations
- The IMAP stub implements only the commands our client uses; a real server may behave differently (folder names, literal formats, throttling text, XOAUTH2 error challenges).
- Authentication-Results parsing follows RFC 8601 as commonly written; real headers may differ per MTA. Domain alignment uses a small built-in public-suffix list, not the full list.
- Backoff timing is tested with injected clocks, not in real time; the worker process check ran 14 s, not a soak test.
- No test covers multi-worker or multi-host operation (single worker by design) or SQLite at scale.
- Permissions: Linux file modes only; no encryption at rest; the service account separation described in the unit template is untested.
