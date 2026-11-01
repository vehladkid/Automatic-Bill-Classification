# Email ingestion worker: operations runbook

Status (honest): implemented and tested **locally** (local IMAP stub, private test CA, stub token endpoint, real OCR/Task A/Task B models). **Not** tested against a real provider mailbox, **not** deployed, **not** running. The Streamlit app still has **no real authentication** (demo identity selector): keep it bound to 127.0.0.1 and do not expose it or the worker's data directory to other users until SSO exists.

## What the worker does
`python -m ingest.run worker --imap` loops every N seconds: (1) retry due failures, (2) list new UIDs after the stored cursor (UIDVALIDITY+UID in `poll_state`), (3) per message fetch HEADERS -> authorise -> only then fetch the body -> ingest, (4) drain the job queue (OCR + Task A + Task B). The mailbox is opened **read-only**; nothing is flagged, moved or deleted unless `HRBC_IMAP_MARK_SEEN=1` (refused in test mode).
Authorisation = explicit sender allow-list (`add-sender`, several rows per employee = aliases) + expense mailbox/alias in To/Cc/Delivered-To + at least one **aligned** SPF/DKIM/DMARC pass written by your own server (`HRBC_INGEST_AUTHSERV_ID`). Fail closed when the mailbox or authserv-id is unset.

## One-time setup (human steps)
1. Create the expense mailbox, the credentials (app password, or an OAuth client + a refresh token obtained through the provider's consent screen; the consent step is not implemented here), and find your server's `authserv-id` in a received message's `Authentication-Results` header.
2. `mkdir -p ~/.config/hrbc && chmod 700 ~/.config/hrbc`; copy `deploy/worker.env.example` to `~/.config/hrbc/worker.env`, `chmod 600`, fill it in; put secrets in separate 0600 files (`*_FILE`).
3. `cd app && set -a; . ~/.config/hrbc/worker.env; set +a; ../.venv/bin/python -m ingest.run check-config` must print `"ok": true` (exit 0). It never prints values.
4. Register employees: `../.venv/bin/python -m ingest.run add-sender asha@company.example E1 --admin hr1` (repeat per address).
5. **Test mailbox first**: set `HRBC_ENV=test`, a separate `HRBC_OUT`, `HRBC_TEST_MAILBOX_USER`/`HRBC_TEST_FOLDER` equal to the configured user/folder. Send a handful of controlled test messages, run `worker --imap --once`, inspect, then switch to prod config.
6. `../.venv/bin/python -m ingest.run check-perms --fix` (everything under `HRBC_OUT` must be 0700/0600).
7. First run polls from the current end of the folder (`HRBC_IMAP_INITIAL=new_only`); set `all` once if you want existing mail ingested.

## Start / stop
Foreground: `cd app && ../.venv/bin/python -m ingest.run worker --imap --interval 60` (Ctrl-C or SIGTERM = finish the current job, record a clean stop, release the lock).
Supervised: install `deploy/hrbc-ingest-worker.service` as a user unit (edit paths), `systemctl --user daemon-reload`, then `systemctl --user start hrbc-ingest-worker` (the template is NOT installed or enabled by this repository). Only one worker can run: a second start exits with code 3. A worker with any configuration problem exits with code 2 and `ExecStartPre` blocks the unit.

## Inspect
- `python -m ingest.run health` : JSON, exit 0 ok / 1 degraded or down. Fields: last loop age, last successful poll age, last poll error CODE, consecutive poll failures, queued/processing/dead jobs, retrying/dead messages, oldest queued job age, problems (`worker_not_running`, `poll_error:<code>`, `dead_letters_present`, `queue_backlog`). Wire this to any monitor.
- `python -m ingest.run failures` : retrying and dead-lettered messages / webhook fetch jobs (codes, attempts, timestamps; never content).
- `journalctl --user -u hrbc-ingest-worker` : logs contain ids and codes only (sender is a short hash; no subjects, bodies, addresses of rejected mail, tokens or passwords).
- Email Inbox page (HR role): shows retrying/dead-lettered items and a "return to retry" button.

## Failure and recovery
| Symptom | Meaning | Action |
|---|---|---|
| `poll_error:credentials_rejected` | password/app password/OAuth refresh token rejected or revoked; the worker backs off up to 1 h | fix the credential, restart; polling resumes from the stored cursor, nothing is lost |
| `poll_error:tls_verification_failed` | certificate untrusted, expired or wrong host | fix the server certificate/host name; never disable verification (no switch exists); test CA trust is only allowed outside prod via `HRBC_IMAP_CA_FILE` |
| `poll_error:rate_limited` / `provider_transient` | provider throttling or network | automatic backoff (honours Retry-After) |
| message in `failures` state `retry` | an environment/storage error (e.g. `pdfinfo` missing, database busy); retried with exponential backoff, max `HRBC_INGEST_MAX_ATTEMPTS` (5) | fix the cause; it recovers by itself |
| state `dead` | attempts exhausted: dead-letter (the message is NOT lost and NOT marked completed; it is not an email row) | fix the cause, then `python -m ingest.run requeue-dead --admin NAME [--key K]` |
| attachment `FAILED` / `retries_exhausted` | processing kept failing | HR retries it in the Email Inbox page or sends it again |
| email `NO_DOCUMENT` | message without a supported attachment and no bill-like body | none |
| email `REVIEW_REQUIRED` + reason `possible_bill_in_body` | the bill seems to be in the body/HTML (not converted automatically) | a person opens the original mail and uploads/forwards the bill as PDF/JPG/PNG |
| worker killed (`SIGKILL`/power loss) | job lease expires (15 min) and the job is re-claimed with the same deterministic document id | nothing; no duplicate document is created |
| UIDVALIDITY changed (mailbox rebuilt) | UIDs restart | the folder is re-read once; Message-ID de-duplication prevents duplicates |
Backups: `HRBC_OUT/hr/*.db`, `results/`, `classified/` (all private files). Retention of raw copies: `python -m ingest.run purge --days 14`.

## Not done / out of scope
Real authentication (SSO) and per-user TLS in front of Streamlit; Gmail API / Microsoft Graph adapters and push subscriptions (the webhook receiver is provider-neutral and fetches by Message-ID through IMAP search); body-to-document conversion; multi-bill PDF splitting; historical backfill beyond `HRBC_IMAP_INITIAL=all`; malware scanning; encryption of SQLite at rest; verification against a live provider.
