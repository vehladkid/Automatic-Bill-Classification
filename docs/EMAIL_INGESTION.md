# Email-to-document ingestion

**Status (updated after the production-hardening pass): implemented and tested locally (local IMAP stub with verified TLS, stub token endpoint, real OCR/Task A/Task B models). NOT verified against a live mailbox, NOT deployed, NOT running.** Operations, security policy and recovery: `docs/EMAIL_INGESTION_OPERATIONS.md`; audit and readiness: `reports/email_ingestion/`. Sections below that describe polling, SPF/DKIM, `\Seen` marking or the 60 s loop are superseded by the runbook (cursor polling, aligned-evidence policy, read-only mailbox).
Remaining external steps are listed in section 5.

## 1. Flow
`provider (IMAP poll | signed webhook -> fetch) -> authorise sender -> per-attachment validation -> sha256 dedupe -> store raw copy -> enqueue job -> worker: BillService.process (OCR, Task A, Task B, optional LLM, profile, field report) -> persist -> notify`
Code: `app/ingest/{provider,webhook,validate,pipeline,api,db,run}.py`; UI: `app/inbox_ui.py` (page "Email Inbox"); processing: `app/service.py::BillService.process(data, filename, owner, source)` (backward compatible).

| Requirement | Implementation |
|---|---|
| Provider API / push preferred | `MailProvider` abstraction. `WebhookHandler` (HMAC-SHA256 over `timestamp.body`, 5-min tolerance, event-id dedupe, **enqueue only**). `ImapProvider` polling as fallback (`BODY.PEEK`, marks `\Seen` only after durable queueing). Gmail/Graph adapters would implement `fetch_new/fetch_by_id/mark_processed`. |
| Authorised mailbox / senders | `HRBC_INGEST_MAILBOX` recipient check; `senders` allow-list maps From address -> employee id (`python -m ingest.run add-sender`). **SPF/DKIM pass required** in `Authentication-Results` (default on), because the From header can be forged. The employee is determined by the allow-list, never by email text. |
| Formats | PDF/JPEG/PNG by **file signature**, extension and declared MIME must agree. Archives, office files, executables, scripts, HTML rejected (no archive-bomb surface). |
| Limits | 20 MB/file, 10 pages, 10 attachments, 50 MB/email, 60 MP images (all env-configurable). Encrypted/corrupt/empty PDFs and images rejected with a specific code and instruction. Tiny inline images (logos) ignored. |
| Idempotency | UNIQUE(provider, message_id), UNIQUE(email_id, idx), UNIQUE job key, webhook event table, content-hash dedupe per employee (same file for another employee is not a duplicate). Completed attachments are never reprocessed. |
| Background + retry | `jobs` table with leases (a crashed worker's job is recovered), transient errors retried with exponential backoff up to 3 attempts, then `FAILED/retries_exhausted`; manual retry for processing failures only. |
| Partial failure | Each attachment has its own state; one bad attachment never blocks the others. Email status is rolled up: `RECEIVED -> VALIDATING -> QUEUED -> PROCESSING -> REVIEW_REQUIRED / COMPLETED / FAILED` (+ `REJECTED`). |
| Tenant isolation | Every `ingest.api` function takes a `Principal`; employees see only their own emails/documents; HR sees all; forbidden and non-existent are indistinguishable. Legacy un-owned web uploads stay visible in the demo's Results page and are hidden from the Email Inbox for employees. |
| Privacy | Logs contain ids, codes and a hash of the sender only; no OCR text, subjects of rejected mail, or bodies. Email body is never parsed or acted on. Raw inbox copy is deleted after successful processing; `ingest.run purge --days N` removes leftovers. |
| Notifications | The product had none; in-app notification rows are shown on the Email Inbox page. No outbound email is sent. |
| Review + correction | `api.correct_field`: validated, reason required, original value kept, actor/role/time stored; corrected values are labelled `human_correction`. |

## 2. Run it
```bash
cd /home/tejvir-singh/Desktop/ML-project/app
../.venv/bin/python -m ingest.run add-sender asha@corp.example E1 --admin hr1
set -a; source ../config/env.example; set +a          # after editing real values in a PRIVATE copy, not this file
../.venv/bin/python -m ingest.run worker --imap --interval 60      # polls the mailbox, processes jobs
../.venv/bin/python -m ingest.run webhook --port 8765               # optional; put behind TLS / your provider's push subscription
```
## 3. Tests
`.venv/bin/python -m unittest discover -s tests -v` (169 tests). `tests/test_ingest.py` covers: multi-attachment and multi-page mail, duplicate deliveries/webhooks/files, forged or unknown senders, wrong mailbox, zip/exe/html, type mismatch, corrupt/empty/oversize/too-many-pages/bomb/encrypted, retries and lease recovery, partial failure, tenant isolation, corrections. (A mutation check confirmed the isolation tests fail when the check is removed.)
Also run once end-to-end with the real models and a mocked provider: 4 attachments -> 3 processed + 1 zip rejected, redelivery created nothing, a second employee saw nothing.

## 4. Limits
- No live-mailbox verification; IMAP polling needs an app password or OAuth you must provision. Gmail/Graph adapters and push subscriptions are not written.
- Authentication-Results is trusted as written by **your** mail server; if your server does not add it, set `HRBC_INGEST_REQUIRE_AUTH=0` knowingly or rely on a dedicated mailbox with provider-side sender rules.
- Worker is a single process using SQLite (WAL); for multiple workers/hosts move to Postgres + a real queue. SQLite files are unencrypted at rest.
- No HTTP REST layer exists in this app (Streamlit, in-process); adding FastAPI would be a second framework, so the contract is `ingest/api.py`.
- Malware scanning (ClamAV) is not integrated.

## 5. External setup still required
Mailbox + credentials, SPF/DKIM-adding MTA or provider, TLS termination for the webhook, secrets storage, sender allow-list population, retention policy decision, and an access-control/SSO layer to replace the demo role selector.
