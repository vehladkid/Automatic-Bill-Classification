# Multi-email ingestion and concurrent processing: feasibility audit

Date: 2026-10-27. Scope: audit only. No application code, schema, OAuth flow, worker/queue configuration, model or production data was changed. The only file added besides this report is the read-only probe `smoke/concurrency_probe.py`, which writes to a temp directory.
Labels: **[CONFIRMED]** read in code or measured here; **[DOC]** provider documentation, not verified against Google in this session; **[UNVERIFIED]** could not be checked.

## A. Executive summary

- **Premise mismatch (important).** The brief assumes PostgreSQL, a frontend account with a connected inbox, and per-user OAuth. The implementation is different: **SQLite** (`outputs/hr/ingest.db`, `hr.db`), a **Streamlit** app with a self-asserted employee ID (no authentication), and **one organisation-owned "expense mailbox"** that employees e-mail bills *to*. Sender address -> employee is an allow-list (`senders` table). There is no "connect my inbox" feature. [CONFIRMED]
- **Two Gmail accounts feeding one employee: not possible today**, and not by a small tweak. One OAuth refresh token (environment variable/file), one provider instance, one cursor key (`gmail:me`), one worker (flock) per deployment. [CONFIRMED]
- **Processing is sequential today, by design in three independent places** (worker loop, `BillService.LOCK`, 6 GB GPU). Measured: 2 threads through `process()` = 60.0 docs/min vs 60.6 sequential, i.e. no gain. [CONFIRMED]
- **A second inbox would add an ingestion source and reliability, not throughput.** Throughput is bounded by the single GPU/process, not by mailbox count. Removing the lock gives about 1.35x at 2 workers (measured, small sample) and fails with CUDA OOM / cuDNN errors at 4 on this 6 GB GPU. [CONFIRMED]
- **Verdict: PROCEED AFTER CHANGES**, with the scope reduced: add a `mail_accounts` model and per-account cursors; keep processing at concurrency 1 by default (optionally 2, opt-in) rather than building a worker farm.

## B. Current architecture [CONFIRMED]

| Concern | Finding | Evidence |
|---|---|---|
| Provider | Gmail REST API (`--gmail`, read-only scope) or IMAP (`--imap`, password or XOAUTH2). Not Microsoft Graph. | `app/ingest/gmail_api.py`, `provider.py` (ImapProvider), `run.py:25` |
| OAuth | Installed-app PKCE + loopback redirect via CLI `python -m ingest.run google-auth`. Stores **one** refresh token file (0600). Not a web flow; no session. | `google_auth.py` `authorize()` |
| Token use | `RefreshTokenSource.from_env()` reads one client file + one refresh token from env/`*_FILE`. Handles `invalid_grant` as `CredentialsExpired`. Rotating tokens persisted to one path. Tested only against a local fake endpoint, never Google. | `oauth.py` |
| Detection | **Polling** every `--interval` s (default 60). Webhook receiver exists (`webhook.py`) and only enqueues `fetch:` jobs; Gmail push/Pub/Sub/`users.watch`/`historyId` are **not implemented** (grep: none). | `run.py:36-45`, `webhook.py` |
| Retrieval | Gmail: list INBOX ids (<=3 pages x 100), fetch `format=metadata` headers, authorise, then `format=raw`; parsed by the same RFC822 parser as IMAP. | `gmail_api.py` `_ids`, `fetch_headers`, `fetch_message` |
| Cursor | `poll_state(provider, mailbox_key)` with `last_uid` = highest Gmail id (hex->int). Gmail `mailbox_key()` is the constant `"gmail:me"`. Documented gap: back-dated mail with lower id is missed. | `gmail_api.py:20,51`, `db.py:25` |
| Attachment -> document | `ingest_message()` validates (JPEG/PNG/PDF), SHA-256, stores a 0600 copy, inserts one `jobs` row per attachment (`key='att:<id>'`, UNIQUE). | `pipeline.py` `ingest_message` |
| Queue | **Exists, in SQLite**: `jobs` table with state/attempts/`next_run`/lease. `claim()` is an atomic `BEGIN IMMEDIATE` claim with a 900 s lease and crash recovery, orders by `id LIMIT 1`. No broker, Celery, Redis. | `pipeline.py:196-208` |
| Execution | One loop: `poll_once()` then `drain()` (one job at a time) then sleep. | `run.py` loop, `pipeline.py:267` |
| Single instance | `fcntl.flock` on `outputs/hr/worker.lock`; a second worker exits 3. | `run.py:19-22` |
| Processing lock | `BillService.process()` runs entirely inside `with LOCK:` (a `threading.Lock`). | `service.py:29,156` |
| LLM | Off by default; when on, `BoundedSemaphore(1)` and keep-alive 0 because the LLM and OCR/A/B do not fit together on 6 GB. | `llm/service.py:12`, `docs/TAX_AND_LLM.md:108` |
| Job -> user | `attachments.employee_id` copied from the sender allow-list result; passed as `owner` into `process()`; stored in `record.json`. | `policy.py:43-58`, `pipeline.py:step` |
| Storage | `emails`, `attachments`, `jobs`, `inbox_failures`, `poll_state`, `senders`, `notifications`, `corrections`, `ingest_audit`, `worker_state` in SQLite (WAL, 15 s timeout). **Results (`record.json`, OCR, images) are files** under `outputs/results/<doc_id>/`, read by scanning the directory. No PostgreSQL code or dependency (`psql` binary exists on the machine; the project does not use it). | `db.py`, `service.py:list_records` |
| Idempotency | `emails UNIQUE(provider, message_id)`; `jobs.key UNIQUE`; attachment dedupe by `(employee_id, sha256)`; deterministic `doc_id` from attachment id + sha so a retry reuses/clears the same result folder. | `db.py:17`, `pipeline.py:75,_doc_id`, `service.py:160-170` |
| Retries / failures | Per-message `inbox_failures` with backoff and dead-letter (5), per-job retry (3) with backoff, `requeue-dead`, HR retry button. Poll-level backoff for auth/rate-limit errors. | `pipeline.py` `record_failure`, `step`, `poll_once` |
| Frontend | Streamlit, in-process, no REST. "Employee ID" text box; `ingest.api` enforces owner==user. Single-inbox assumption: no mail-account UI at all. | `tax_ui.py:24`, `inbox_ui.py`, `ingest/api.py` |

## C. Bottleneck analysis

Measured on this machine (RTX 4050 6 GB laptop, 16 logical CPUs, 15 GB RAM) with the real models, `HRBC_OUT` in a temp dir, LLM off. Another process (pid 132599, not started by this audit) already held about 1.3 GB of GPU memory, so all peaks below include it.

| Measurement | Result |
|---|---|
| Model load (OCR + Task A + Task B + second classifier), once per worker | 11.2 s; GPU 1,311 -> 2,604 MiB (about 1.3 GB per loaded copy) |
| Sequential `process()`, 18 sample bills (incl. PDFs), warm | wall 17.8 s, **60.6 docs/min**, mean 0.99 s, median 0.73 s, max 4.71 s |
| Mean stage time (GPU stages) | OCR 0.48 s, Task A 0.10 s, Task B (LayoutLMv3) 0.14 s |
| 2 threads calling the real `process()` | wall 18.0 s, **60.0 docs/min** -> `LOCK` serialises it [CONFIRMED] |
| Component-level, lock bypassed in a scratch process only, 17 images, interleaved runs | 1 thread: 77.9 / 81.9 / 80.2 docs/min. 2 threads: 108.6 / 111.7 / 109.6 docs/min (about 1.35x), 0 errors, **peak GPU 5,652-5,670 MiB of 6,141** |
| Same, 4 threads | **10 of 17 documents failed**: `CUDA out of memory`, `cuDNN CUDNN_STATUS_INTERNAL_ERROR`, ONNX Runtime errors; peak 5,772 MiB. The 267 docs/min printed by the probe is an artefact of failures and must be ignored. |

Interpretation:
- **Measured bottleneck: GPU memory and the single process, not e-mail.** Each document costs about 1 s of mostly GPU time; a second thread helps a little (the CPU/IO parts overlap) but a third/fourth thread exhausts 6 GB.
- **Measured, small sample:** 17 images of the bundled demo set, three repeats; no multi-page or large scans, no CPU/GPU utilisation traces. The 1.35x figure is not a guarantee.
- **Suspected, not measured:** polling and processing share one thread, so a long `drain()` (up to 1000 jobs) delays the next poll; Gmail list cost; SQLite writer contention with several workers; LLM-on path (documented 4.6 s/doc plus GPU contention, from `docs/TAX_AND_LLM.md`, not re-run).
- Only 15 historical results exist in `outputs/results`; no production-load logs exist to analyse. [CONFIRMED]

Teacher's concern, answered with evidence: a long queue behind one worker is real for this architecture, but it is caused by sequential processing on one GPU. More inboxes feed the same queue faster and do not drain it faster (Scenario A). Scenario B only helps up to a small, hardware-bounded factor (about 2 workers here).

## D. Multi-email compatibility

Already works:
- Idempotent ingest, UNIQUE keys, job leases that would tolerate multiple consumers at the DB level (atomic `BEGIN IMMEDIATE` claim).
- Dedupe by `(employee_id, sha256)`, so the same bill arriving in two inboxes for one employee is processed once.
- Document ownership and tenant isolation in `ingest.api`.
- Aliases: `HRBC_INGEST_MAILBOX_ALIASES` accepts extra *recipient addresses*, but they must be delivered into the same single polled mailbox (e.g. forwarding). This is the zero-code way to get "two addresses, one employee" today. [CONFIRMED in `policy.py:43`, not run live]

Must change (each is a concrete defect for two accounts):
1. **No account entity.** Nothing stores credentials, status, or cursor per account. Add a table (below).
2. **Cursor and failure keys collide.** Gmail `mailbox_key` is the constant `"gmail:me"`; `inbox_failures.key = "gmail:gmail:<id>"`. Gmail ids are only unique per mailbox, so two accounts would overwrite each other's cursor and failure records.
3. **`emails UNIQUE(provider, message_id)`** has no account column. `message_id` is the RFC Message-ID header, so a message addressed to both inboxes is stored once (good for one user, wrong if the inboxes belong to different tenants). Needs `account_id`.
4. **One credential per process** (`RefreshTokenSource.from_env()`); a provider per account is needed.
5. **Fetch jobs** (`fetch:<provider>:<message_id>`) do not say which account to fetch from.
6. **Authorisation model.** Today the sender allow-list decides the employee. A per-employee "connected inbox" implies a different rule (mail in a connected inbox belongs to that account's owner). That is a policy decision, not just a column, and has security consequences (see H).
7. **Worker topology.** Single flock and one-thread loop; a disconnected account currently raises poll-level backoff (`next_poll_after` in the one `worker_state` row), which would stall every account.
8. **Frontend.** No account-management screen, and no real login to attach an account to (identity is a text box).

Cannot be verified here: behaviour with a real second Google account (no accounts were connected, by instruction); Google quotas, consent-screen and verification effects (see F).

## E. Proposed architecture (smallest reliable change)

```
Mail account A (Gmail/IMAP, own token+cursor) --+
                                                +--> poll per account --> ingest_message (account_id) --> jobs (SQLite, existing)
Mail account B (own token+cursor) --------------+                                                          |
                                                                                    worker(s): claim (existing lease), N = HRBC_WORKERS (default 1, max 2)
                                                                                                           |
                                                                           OCR -> Task A -> Task B (shared models, bounded GPU gate)
                                                                                                           |
                                                                              record.json + attachments/emails rows -> Streamlit (per employee)
```

Lifecycle: (1) poll each *active* account independently, errors isolated per account; (2) headers -> authorise -> raw fetch; (3) `ingest_message` writes `emails(account_id, ...)`, attachments, one job each; (4) a worker claims a job (lease), runs OCR/A/B, writes the result under the deterministic `doc_id`; (5) rollup updates email status, notification is written; (6) the UI lists by `owner`, with the source account shown.

Answering "which do we need":
- **Multiple email accounts:** yes, if the product requirement is multiple inboxes. It is an ingestion feature.
- **Multiple queues:** **no.** One SQLite `jobs` table is enough at this volume.
- **Multiple workers:** **at most 2, opt-in, after measurement**; the default stays 1. Do not add Celery/Redis/BullMQ: the existing leased SQLite queue already gives atomic claim, retry, dead-letter; a broker adds an operational component and does not remove the GPU limit.
- **Separate stages:** optionally split I/O (poll/download) from inference. Cheap form: poll in its own thread so downloads continue while the GPU thread works. Splitting OCR vs A vs B into separate stages is not justified by the measurements (they share one GPU and total about 0.7 s).

## F. Google OAuth and provider constraints

Confirmed from code: the current flow can authorise one account per run; a second `google-auth` run to a different `--token-file` would store a second token without touching the first, but nothing reads it. Scope is `gmail.readonly` only. The flow is a localhost CLI consent, so it cannot run from a deployed web page; a multi-user product needs a web-server redirect flow with `state` bound to the logged-in session.

[DOC] / [UNVERIFIED] (no Google access in this audit):
- Apps in "Testing" with an external user type: refresh tokens expire after 7 days (the repo's own docs say the same), and only listed test users can consent. Production use needs the consent screen published.
- `gmail.readonly` is a *restricted* scope; production apps serving users beyond testers generally require Google verification and a security assessment. Check the current requirements before committing to the "connect your Gmail" product.
- Gmail API has per-user and per-project quota units; polling every 60 s per account is small, but call cost (list + metadata + raw per message) scales with accounts and backlog. Confirm in Cloud Console.
- Push (`users.watch` + Pub/Sub + `historyId`) is not implemented, needs a public HTTPS endpoint and renewals, and the deployment is a local laptop with a loopback webhook (`--host 127.0.0.1`). Polling is the supportable choice here.
- Cursor weakness: max-id cursor can miss back-dated mail; `history.list` with `historyId` would be the robust replacement.

## G. Required changes

| Area | Item | Status |
|---|---|---|
| DB | `mail_accounts(id, employee_id, provider, address, status, last_sync_at, last_error_code, credential_ref, created_at)`; add `account_id` to `emails`, `poll_state` key, `inbox_failures` key, fetch-job key; `UNIQUE(account_id, message_id)` | Requires a change |
| DB | Migration framework exists (`MIGRATIONS` + `PRAGMA user_version`) but only supports `ADD COLUMN`; UNIQUE change needs a table rebuild | Requires a change |
| OAuth | Per-account token storage (encrypted at rest, or referenced files 0600), per-account refresh, revoke/disconnect, re-consent | Requires a change |
| OAuth | Web redirect flow tied to a real session | Blocked on authentication (none exists) |
| Backend | Provider-per-account loop in `poll_once`; per-account backoff instead of the single `worker_state` row | Requires a change |
| Worker | Poll in a separate thread from `drain` | Requires a change (optional) |
| Worker | Bounded concurrency (`HRBC_WORKERS`, default 1) and a GPU gate; remove/replace `BillService.LOCK` only if the models are shown thread-safe | Requires additional measurement |
| Queue | Atomic claim with lease, retry, dead-letter, deterministic doc id | Already supported |
| Dedupe | Per-employee content hash across inboxes | Already supported |
| Dedupe | Message-ID uniqueness scoped per account | Requires a change |
| Frontend | Account list/connect/disconnect/status; show source account per document | Requires a change |
| Frontend | Real authentication | Blocked (not in scope of this project today) |
| Google | Verification, test users, quotas, scope review | Blocked by external dependency |
| Tests | Two-account fixtures using `tests/imap_stub.py` / fake Gmail server; collision, disconnect-isolation, retry-no-duplicate tests | Requires a change |
| Deployment | Service file keeps single instance; `HRBC_WORKERS` env; no new services | Requires a change |

## H. Performance expectations

- No linear scaling. Evidence supports about 1.35x at 2 concurrent inference threads on this 6 GB GPU, and failures at 4. This is one small benchmark; do not quote it as a guarantee.
- A second inbox will not change documents/min. It can improve ingestion resilience and reduce the chance that one broken mailbox blocks intake.
- Reproducible benchmark to extend: `reports/email_ingestion/smoke/concurrency_probe.py` (temp `HRBC_OUT`). Before deciding on concurrency, add: a representative set (50-100 real bills, mixed single-page images and multi-page PDFs, not only the demo set), workers = 1, 2, 3, five repeats each, record per-document latency p50/p95, docs/min, `nvidia-smi` GPU memory at 0.2 s intervals, CPU utilisation, error rate, with the Streamlit app and Ollama stopped. Include a run with the LLM on.

## I. Security and reliability

- **Credentials:** keep per-account tokens out of the repo and logs (current code already redacts). Multiple accounts increase the secret surface; store them encrypted or as 0600 files per account.
- **Tenant isolation:** currently enforced in `ingest.api`, but identity is self-asserted (`tax_ui.identity`), so a user can type another employee's ID. This must be fixed (real auth) before users can connect their own inboxes.
- **Authorisation rule change risk:** if "mail in a connected inbox belongs to its owner" replaces the sender allow-list, a forwarded or spoofed message in a connected inbox would be attributed to that user. Keep authentication-results checks and the per-account `address` match.
- **Retry/duplicates:** existing UNIQUE keys and deterministic ids already cover retried jobs; adding `account_id` must not weaken them.
- **Failure isolation:** a revoked account must set only that account to `needs_reauth` and be skipped; today a `ProviderAuthError` aborts the whole poll and delays all polling.
- **Concurrency:** SQLite WAL with `timeout=15` is adequate for 1-2 workers; busy errors are already mapped to `database_busy` retries.
- Pre-existing observation: a previous audit file (`PRODUCTION_READINESS.md`) says ingest.db was 0644 before hardening; this audit did not re-check live permissions because no `outputs/hr` directory exists.

## J. Implementation plan (ordered, low risk)

1. Decide the product rule: "connected inbox owner = document owner" vs the current sender allow-list. Decide how users authenticate. (Blocks everything user-facing.)
2. Migration: `mail_accounts`; `account_id` on `emails`, cursor/failure/fetch keys. Backfill a single legacy account so existing single-mailbox deployments behave identically.
3. Provider factory by account; per-account token source; `poll_once` loops accounts with isolated error state.
4. Tests with two fake mailboxes: id collisions, same Message-ID in both, one account revoked, retry without duplicate document.
5. Benchmark per section H; only then add `HRBC_WORKERS` (default 1) with a GPU memory gate. Keep `LOCK` semantics until thread-safety is shown.
6. Frontend: account list, connect/disconnect, status, source column.
7. Google: web OAuth flow, test users, verification decision. Consider `history.list` cursor.

## K. Verification evidence

Commands run (all read-only on project data):
- Directory and code reads: `app/ingest/{db,run,config,pipeline,gmail_api,google_auth,oauth,policy,api,webhook,provider}.py`, `app/service.py`, `app/llm/{service,config}.py`, `app/inbox_ui.py`, `app/tax_ui.py:24`, `scripts/taskA/predict.py`, `scripts/taskB/predict.py`, `deploy/*`, `config/env.example`, existing docs and `reports/email_ingestion/*`.
- `grep` for postgres/redis/celery/pubsub/historyId/threading: no database server, broker, push or Pub/Sub code found.
- `cd tests && HRBC_OUT=<temp> ../.venv/bin/python -m unittest test_ingest test_ingest_hardening test_gmail_api test_google_auth` -> **Ran 129 tests, OK** (34 s). These use local stubs/fake Gmail; none touch Google.
- `smoke/concurrency_probe.py` (two runs, results in section C). First run's 4-thread result contained errors and is reported as failed.
- Pre-existing artefacts consulted: `smoke/worker_process_check_result.json` (14 s per `PRODUCTION_READINESS.md`, 20.0 s in the JSON, to process the first message; whether that includes model load was not checked; second worker refused rc 3), `docs/TAX_AND_LLM.md` (LLM latency).
- After the benchmark `outputs/results` still holds the original 15 results; benchmark output went to a temp directory that was removed.

Limitations: no live Google/IMAP account used; no multi-account test possible by instruction; Google constraints are from documentation knowledge; benchmark is a small single-machine sample; the unguarded run bypassed the service lock in a scratch process and is not a statement that `process()` is thread-safe; the 1.3 GB GPU use by another process affected the headroom measurements.

**Recommendation: PROCEED AFTER CHANGES** (see Executive summary).

> **Update 2026-10-27:** the implementation phase is documented in [MULTI_EMAIL_IMPLEMENTATION_REPORT.md](MULTI_EMAIL_IMPLEMENTATION_REPORT.md) (operator-only linking, per-account cursors/credentials, migration results, remaining blockers). Findings above describe the code **before** that change.
