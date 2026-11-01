# Email ingestion and automatic bill processing: completeness audit

> **Superseded in part (2026-10-19):** the blockers listed here were addressed in the production-hardening pass; current status, evidence and remaining blockers are in `PRODUCTION_READINESS.md` and `PRODUCTION_READINESS_TESTS.md`. This file is kept unchanged as the pre-fix audit record.

Audit date 2026-10-19. Method: read every file of the pipeline (`app/ingest/*.py`, `app/service.py`, `app/inbox_ui.py`, `app/tax_ui.py::identity`, `docs/EMAIL_INGESTION.md`, `config/env.example`, `tests/test_ingest.py`), searched the machine for deployment/scheduler artefacts, ran the existing tests, and executed two controlled experiments (scripts and raw results in `reports/email_ingestion/smoke/`). No model, Task A/B asset, frozen test, V1 snapshot or V2 asset was touched; no real mailbox was contacted; no credentials exist or were printed. No fix was implemented.

## A. Executive verdict
**Partially implemented.**
The core engine exists and works end to end: a message is authorised, its attachments are validated and de-duplicated, jobs are queued, a worker runs real OCR + Task A + Task B + profile classification, and results are persisted and shown in a Streamlit page. That path was exercised here with the real models, driven through the real IMAP client code against a local IMAP server (not a live provider).
It is **not fully automatic today** and I found implementation defects, not only missing verification: nothing is deployed or scheduled, no credentials/provider are configured, the only real provider is password-IMAP with unverified TLS, a single failing message can block all later mail, the webhook path cannot complete for the only real provider, and sender authentication can be satisfied by an attacker's own domain. Bills that are only in the email body are silently marked completed.

## B. Feature matrix
Status key: complete / partial / missing / unverified (code exists, never run against the real thing).

### B1. Email ingestion
| Capability | Status | Responsible code | Evidence | Missing work / failure mode |
|---|---|---|---|---|
| Connect to provider securely | **partial** | `provider.py::ImapProvider._c` (`IMAP4_SSL`, no `ssl_context`) | smoke: connected to a **self-signed** local server without error, so certificates are not verified (Python's default for this call is an unverified context) | pass a verifying `ssl.create_default_context()`; password could be sent to a man-in-the-middle |
| Authenticate with intended credentials/OAuth | **partial** | `ImapProvider.__init__` reads `HRBC_IMAP_HOST/USER/PASSWORD` | smoke: LOGIN works against the local server; bad credentials raise `imaplib.IMAP4.error` | password/app-password only; no OAuth2 (XOAUTH2/Gmail API/Graph). Providers that disabled basic auth cannot be used. Not verified live |
| Fetch automatically without manual export | **partial** | `ImapProvider.fetch_new` (`UID SEARCH UNSEEN`, `BODY.PEEK[]`), `run.py worker --imap` | smoke: 7 messages fetched | works only if someone starts the worker with real env vars; nothing does (section C) |
| Poll or receive notifications | **partial** | poll loop `run.py` (default 60 s); webhook `webhook.py` | smoke: poll ingested 7 then 0 on the next poll | webhook accepts a generic signed `{event_id, message_id}` only; no Gmail Pub/Sub / Graph subscription; **webhook jobs always die with the IMAP provider** (`fetch_by_id` not implemented: `NotImplementedError` x3 -> job `dead`, 0 emails created; `extra_checks_result.json`) |
| Pagination | partial | `[:50]` per poll, oldest first, messages marked Seen after ingest | smoke (7 msgs) only | no cursor; fine for UNSEEN but cannot backfill |
| Date ranges / historical backfill | **missing** | - | - | only UNSEEN messages are ever fetched |
| Retries / rate limits | **partial** | loop-level only: `run.py` catches any exception, logs the class name, sleeps | smoke: bad credentials surface as one exception | no backoff, no rate-limit handling, a new TCP+TLS+LOGIN connection per `mark_processed` (one per message) |
| Expired credentials | **partial** | exception swallowed in `run.py` | smoke | no distinct handling, no alert; mailbox silently stops being read |
| Filter bill-related emails | **partial** | allow-list on From + optional recipient check (`pipeline.authorise`) | tests + smoke (stranger rejected, recorded as `REJECTED`) | no content-based relevance filter: every attachment of an allowed sender is processed; irrelevant files end as `REVIEW_REQUIRED`/non-expense |
| Bodies, HTML, forwarded, embedded content | **partial** | `parse_rfc822` (`walk()` over parts); body never read (`body_ignored`) | smoke: forwarded `message/rfc822` with a PDF inside was found (then correctly de-duplicated); **a bill that exists only in the HTML body produced email status `COMPLETED` with 0 documents** | body/HTML invoices (e-commerce, food delivery) and download links are not captured and not flagged |
| Identify/download PDF and image attachments | **complete** (for attachments) | `parse_rfc822`, `validate.py` | smoke + `test_ingest.py` | - |
| Multiple attachments / multiple bills per email | **partial** | one job per attachment (`ingest_message`) | smoke: 2 attachments -> 2 documents | several bills inside **one PDF** are not split: classification and fields use page 1 only (`service.py` warning) |
| Prevent re-processing of email/attachment | **complete** | `UNIQUE(provider,message_id)`, `UNIQUE(email_id,idx)`, per-employee sha256 check, `UNIQUE jobs.key` | smoke: re-sent Message-ID ignored; same PDF in a new mail and in a forwarded mail -> `DUPLICATE`; `test_ingest.py` | missing Message-ID falls back to `imap-uid-N`, unstable if UIDVALIDITY changes |
| Track message/attachment IDs and status | **complete** | tables `emails`, `attachments`, `jobs`, `ingest_audit` (`db.py`) | smoke DB rows | - |
| Handle failures without losing emails | **partial** | per-attachment states; message is marked Seen only after durable insert | smoke OK for validation/processing failures | **poison message**: any unexpected exception type in `ingest_message` (simulated with `pdfinfo` missing -> `FileNotFoundError`) aborts `poll_once`, later messages in that poll are not ingested and the same message fails every cycle (no quarantine). Rejected senders are marked Seen and recorded but nobody is notified |

### B2. Automatic processing (stage by stage)
| # | Stage | Status | Implementation / entry point | Evidence | Gap |
|---|---|---|---|---|---|
| 1 | Ingest email | partial | `pipeline.ingest_message` | smoke, 30 mocked tests | see B1 |
| 2 | Extract attachments | complete | `provider.parse_rfc822` | smoke | - |
| 3 | Validate / reject unsafe files | complete | `validate.validate` (signature allow-list, extension/MIME agreement, size/pages/pixels, encrypted/corrupt PDF; zip/exe/html denied) | smoke: zip rejected `unsupported_type`; tests for each code | no malware scan (documented) |
| 4 | Deduplicate | complete | sha256 per employee | smoke | exact duplicates only; near-duplicate scans (re-photographed) are processed again |
| 5 | OCR / PDF text | complete | worker -> `BillService.process` (`pdf_words` text layer, else RapidOCR GPU) | smoke: PDF used `pdf_text_layer`, JPG went through RapidOCR | OCR text of pages > 1 saved, models use page 1 only |
| 6 | Classify document type | partial | `doc_profile.profile_classify` (rules over OCR + fields) | smoke: salary slip -> `non_expense_document`, receipt PDF -> `expense_bill` | the new 21-class document-type dataset/baseline is **not** integrated; rules-only, flagged `rules_only`; the 12 tax-evidence classes have no trained model or real evaluation |
| 7 | Expense-category classifier | complete (existing) | Task A via `BillService` | smoke: route `auto`; category `food` for the receipt | Task A reported metrics include template-duplicated synthetic test docs (see doc-type audit) |
| 8 | Field extraction | partial | Task B (13 base fields: merchant, invoice no./date, subtotal, tax, total, CGST/SGST/IGST, HSN/SAC, employee, currency, payment method) + rule extractors for profile fields (`doc_fields.py`) | smoke: `invoice_date`, `merchant_name`, `total_amount` extracted for the PDF; slip -> "invoice_date not found" | Task B frozen test F1 about 0.68 (per project reports): extraction is automatic but not reliable; always routed to review when a field is missing/failed |
| 9 | Category vs tax evidence vs document type kept separate | complete | `profile` record: `expense_category`, `document_type`, `benefit_category`, `tax_section_candidates` | smoke records; `inbox_ui.py` shows them separately | candidates are suggestions only; eligibility never asserted |
| 10 | Validate values, flag uncertainty | complete | `pipeline.review_reasons`, `field_report` | smoke: `REVIEW_REQUIRED` with reason | thresholds not tuned on real mail |
| 11 | Persist | complete | `outputs/results/<doc_id>/record.json` + `outputs/classified/...` + `outputs/hr/ingest.db` (SQLite) | smoke (temp dir) | SQLite files default mode 0644, inbox dir 0775 (only raw attachment files are 0600); no encryption at rest, no migrations (`CREATE IF NOT EXISTS`) |
| 12 | Associate with email, attachment, user | complete | `attachments.doc_id`, `source` metadata, `senders` allow-list | smoke: document -> email -> employee E1; E2 sees 0 | owner = allow-list entry; sender must be pre-registered by CLI |
| 13 | Expose via API / frontend | partial | `ingest/api.py` (in-process service layer, no HTTP), Streamlit page "Email Inbox" (`inbox_ui.py`) | smoke through `api.list_documents/get_document`; UI not driven by a browser in this audit | no REST API; **no authentication** (demo role selector; optional HR PIN) |
| 14 | Status, errors, audit | partial | states + `ingest_audit` + notification rows; `logging` | smoke | no metrics, health endpoint or alerting; notifications are in-app only |
| 15 | Safe retries / recovery | partial | job leases, exponential backoff, 3 attempts, manual `retry_attachment` | `test_ingest.py` (mock processor), smoke (no crash injection with real models) | poll-level failure isolation missing (B1); a `dead` fetch job produces no notification |

## C. End-to-end pipeline (actual path)
1. **Arrival**: message lands in the mailbox. Nothing in this repository subscribes to it.
2. **Trigger A (polling)**: `python -m ingest.run worker --imap` (a manual command, loop with `sleep(interval)`). `ImapProvider.fetch_new` -> `UID SEARCH UNSEEN` -> `UID FETCH BODY.PEEK[]` (<= 50) -> `parse_rfc822` -> `Message`.
   **Trigger B (webhook)**: `python -m ingest.run webhook` (plain `http.server`, 127.0.0.1, no TLS) verifies HMAC, writes a `fetch:` job. The worker would call `provider.fetch_by_id`, which the IMAP provider does not implement.
3. **Ingest (synchronous, in the poll loop)**: `ingest_message` -> sender allow-list + recipient + Authentication-Results check -> per-attachment validation + sha256 dedupe -> raw copy to `outputs/ingest/inbox` (0600) -> rows `emails`/`attachments`/`jobs` in one `BEGIN IMMEDIATE` transaction -> **then** `mark_processed` (`\Seen`). *Async boundary 1: the job queue (SQLite table).*
4. **Worker (same process, `drain`)**: `claim` (lease) -> `BillService.process`: PDF text layer or RapidOCR -> Task A -> Task B -> optional local LLM (off by default) -> profile -> field report -> files in `outputs/results/<doc_id>`.
5. **Outcome**: attachment `COMPLETED`/`REVIEW_REQUIRED`/`FAILED`, email status rolled up, in-app notification, raw copy deleted. *Async boundary 2: none; the same process polls and works, so a long OCR job delays polling.*
6. **Read**: Streamlit "Email Inbox" -> `ingest.api` (tenant isolation) -> record files + SQLite. A person must be in the sidebar identity selector (no login).
Nothing runs this chain continuously on this machine: no process is running, no cron/systemd unit or Docker/compose file exists, `outputs/hr/ingest.db` has never been created in the real output directory, and no `HRBC_IMAP_*` variables are set.

## D. Remaining work (dependency order)
**Blockers for fully automatic operation**
1. Poll-loop failure isolation: wrap each message in `try/except`, record failures (quarantine state + notification), never abort the batch; add a poison-message test.
2. Transport and auth hardening: verifying TLS context; require SPF/DKIM **alignment** with the From domain (or DMARC pass) instead of substring `pass`; make `HRBC_INGEST_MAILBOX` mandatory; fail fast on missing configuration at worker start.
3. Production provider adapter: OAuth2 (Gmail API or Microsoft Graph, or IMAP XOAUTH2) with token refresh and distinct handling of expired/revoked credentials, `fetch_by_id`, date/cursor-based listing, rate-limit backoff, one connection per batch.
4. Verify against a dedicated test mailbox (live provider) with a controlled set of messages; record the result.
5. Deployment: a supervised worker (systemd unit or container) separate from the Streamlit app, restart policy, config/secrets injection, log shipping; TLS termination for the webhook if used; a health check and an alert for "worker stopped / last successful poll too old / dead jobs".
6. Real authentication (SSO) for the application and protected file/DB permissions (0600/0700, or a real database).
**Important but not blocking**
7. Body/HTML-only invoices and download links: capture or at least flag (an email with zero documents must not become `COMPLETED`).
8. Split multi-bill PDFs; run classification/extraction on all pages.
9. Integrate a trained document-type model once real labelled data exists (currently rules-only; 12 classes have no real data).
10. Move SQLite to a server database with migrations if more than one worker/host is needed; malware scan; near-duplicate detection; notify on rejected senders.

## E. Test evidence
| What | Command / method | Result | Real or mocked |
|---|---|---|---|
| Existing ingestion + related suites | `cd tests && ../.venv/bin/python -m unittest test_ingest test_profile_autofill test_workflow test_scenarios` | **124 tests OK** (`test_ingest.py` has 30 of them) | `FakeProvider`, stub processor in most tests; `ImapProvider` and `parse_rfc822` are referenced by **no** existing test |
| Full project suite (previous task) | `unittest discover` | 312 OK | as above |
| Local-IMAP end-to-end smoke | `cd app && ../.venv/bin/python ../reports/email_ingestion/smoke/local_imap_smoke.py` | 7 messages fetched over TLS; 6 emails + 1 rejected recorded; 2 documents processed by the real OCR + Task A + Task B (`COMPLETED`, `REVIEW_REQUIRED`); forwarded/duplicate/zip/unknown-sender/body-only cases behaved as listed in B; second poll fetched 0; E2 saw 0 documents; no raw files left; bad credentials raised; poison message blocked later mail. 18.5 s | **Real**: our IMAP client code, parser, pipeline, DB, models, API. **Not real**: the server (a 60-line local IMAP stub), the provider, TLS trust, Authentication-Results from a real MTA |
| Webhook/auth/permission checks | `.../extra_checks.py` | webhook job dead after 3 tries with `ImapProvider`; forged-domain auth accepted; any recipient accepted when mailbox unset; db mode 0644 | Real code, no network, no models |
| Live provider | not run | **no credentials, no test mailbox** | - |
Safety: temp `HRBC_OUT`, no outbound network, no emails sent, no real mail read, fixtures were synthetic or public-licence documents, no personal content in logs.

## F. Final answers
1. **Is email scraping complete?** No. IMAP polling code and the allow-list/dedup/queue logic work against a local IMAP stub, but there is no OAuth, TLS is unverified, no live test, no backfill, body/HTML bills are ignored, and one bad message can stall the batch.
2. **Is automatic attachment processing complete?** Largely yes for PDF/JPG/PNG attachments (validation, dedup, queue, worker, retries, partial failure). Not for multi-bill PDFs or body-only bills.
3. **Is OCR automatic?** Yes: the worker uses the PDF text layer or RapidOCR with no manual step (verified in the smoke run).
4. **Is document-type and expense classification automatic?** Expense category (Task A) yes. Document type is automatic but rules-based; the new document-type classes have no trained model, and tax-evidence suggestions are candidates only.
5. **Is field extraction automatic?** Yes, automatically run and validated, but accuracy is limited (Task B frozen-test F1 about 0.68) and missing/failed fields route the document to human review.
6. **Are results persisted and accessible through the application?** Persisted (files + SQLite) and readable through `ingest.api` and the Streamlit Email Inbox page, with tenant isolation. No REST API and no real login.
7. **Does it process new emails continuously or on a schedule?** Only if someone manually starts `ingest.run worker --imap`; the loop then polls every 60 s. No scheduler, service, container or monitoring is set up and nothing is running.
8. **What prevents fully automatic operation today?** (a) No mailbox, credentials or provider integration is configured or live-tested, and the only adapter is password-IMAP (no OAuth, unverified TLS); (b) no supervised/scheduled worker or deployment; (c) the poll loop has no per-message failure isolation; (d) the webhook path cannot fetch with the real provider; (e) sender authentication can be satisfied by another domain; (f) no application authentication.
9. **Single next implementation task:** build the production provider adapter with per-message failure isolation (OAuth2, verified TLS, `fetch_by_id`, cursor, backoff) and validate it against a dedicated test mailbox, as items 1-4 of section D.
