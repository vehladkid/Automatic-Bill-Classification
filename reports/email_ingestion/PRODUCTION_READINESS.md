# Email ingestion: production-readiness status (tracked checklist)

Baseline before changes (2026-10-19): suite 312 OK; Task B frozen test sha256 `fd8bafb3ad1383e90a2f8de6a30260db2847d64eb74b6d5ab5f9e57e9dbab88c`; V1 snapshot 34/34 OK. After: suite **387 OK**, same hash, 34/34 OK. No model, Task A/B asset, frozen test, V1 snapshot or V2 asset was touched; nothing was downloaded; no real mailbox was contacted. Test detail: `PRODUCTION_READINESS_TESTS.md`. Runbook: `docs/EMAIL_INGESTION_OPERATIONS.md`.

## Verdict (four levels)
| Level | Status |
|---|---|
| Implemented and locally tested | **Yes** for everything checked below (local IMAP stub with real TLS verification against a private CA, stub OAuth token endpoint, fake provider, real OCR/Task A/Task B models, real CLI worker process) |
| Tested against a dedicated real provider mailbox | **No.** No test mailbox or credentials exist. Real provider behaviour (Gmail/Outlook IMAP quirks, XOAUTH2 acceptance, real Authentication-Results formats, throttling messages) is unverified |
| Configured for deployment | **Partly.** A systemd unit template, an env template and a runbook exist and the unit passes `systemd-analyze verify`; nothing is installed, no environment file with real values exists |
| Actually running continuously | **No.** No worker is running; continuous operation was not enabled and is not claimed. The CLI worker was run only as a short local subprocess (14 s to process a message, health ok, second instance refused, SIGTERM exit 0) |

## Re-verification of the audit's findings against the code (before fixing)
| Audit finding | Verified | Where |
|---|---|---|
| IMAP TLS unverified | yes: `IMAP4_SSL` called without `ssl_context` -> `_create_stdlib_context()` (no verification) | old `provider.py` |
| Webhook cannot fetch with IMAP | yes: `fetch_by_id` not implemented -> job `dead` | old `provider.py`/`pipeline.step` |
| Poison message aborts the poll | yes, plus: the transaction was left open on exceptions and a missing `pdfinfo` raised `FileNotFoundError` | old `pipeline.ingest_message`, `validate.py` |
| Auth satisfied by attacker's domain | yes: substring `spf=pass`/`dkim=pass`, no alignment, any Authentication-Results header | old `pipeline.authorise` |
| Mailbox check skipped when unset | yes | old `pipeline.authorise` |
| Body-only bill marked COMPLETED | yes (`rollup`: no attachments -> COMPLETED) | old `pipeline.rollup` |
| DB 0644, dirs 0775 | yes; 109 world/group-accessible paths in the real `outputs/` | filesystem |
| No scheduler/deployment | yes | repository, machine |
**Additional defects found while verifying (not in the audit):** (1) `mark_processed` set `\Seen` = it modified the mailbox; (2) the processing job generated a random document id, so a crash after processing but before the database update created a second document on retry; (3) UIDVALIDITY would have read `0` after the first call (imaplib pops untagged responses); (4) a fetch job's attempt counter was off by one (found by a test); (5) the service wrote results with the default umask.

## Checklist (all tested locally unless noted)
### Phase 2 security
- [x] **2.1 Verified TLS.** `tls.context()` = `create_default_context()` (CERT_REQUIRED + hostname check, TLS >= 1.2); no switch to disable verification exists. Extra trust only via `HRBC_IMAP_CA_FILE`, refused in prod. Tests: self-signed, untrusted CA, hostname mismatch, expired certificate all rejected; default provider verifies; secrets never logged; mutation check (unverified context) fails 4/4 tests.
- [x] **2.2 Authorisation policy** (`ingest/policy.py`). Expense mailbox required (fail closed, also in `config.validate`); aliases and Cc/Delivered-To recipients; explicit employee allow-list is the authorisation (aliases = several addresses per employee); SPF/DKIM/DMARC only as supporting evidence: must come from our own `authserv-id`, must be aligned with the From domain (relaxed, small public-suffix guard), DMARC fail rejects; multiple From addresses rejected; optional domain allow-list; REQUIRE_AUTH=0 refused in prod; forwarded mail uses the outer sender only (inner employee headers not trusted, employee forwarding a vendor bill accepted); IMAP flow fetches HEADERS and authorises before any body/attachment is downloaded. Mutation check (alignment disabled) fails 2/2.
- [x] **2.3 Permissions.** Inventory: `HRBC_OUT/hr/ingest.db(+wal,shm)`, `hr/hr.db`, `hr/worker.lock`, `ingest/inbox/*` (raw copies), `results/<doc>/*` (record, OCR text, page images), `classified/`, `unclassified/`, LLM cache, optional OAuth refresh-token store. All created 0700/0600 (`perms.py`, umask in service/worker/Streamlit); `ingest.run check-perms [--fix]` audits/repairs; the real `outputs/` was repaired (108 paths) and now audits clean; secret `*_FILE` paths must be 0600. **Authentication gap remains:** the app has no real user authentication (self-asserted employee id; optional HR PIN). In an explicit `HRBC_ENV=prod` the HR role is disabled without `HRBC_HR_PIN`, but any user can still claim any employee id. Do not call the application production-secure; keep Streamlit on 127.0.0.1 until SSO exists.
### Phase 3 reliability
- [x] **3.1 Per-message isolation, bounded retries, dead-letter.** `poll_once` -> `_handle_ref` per message; failures recorded in `inbox_failures` (safe code, attempts, next_run, exponential backoff, max 5); `dead` is terminal until `requeue-dead` / HR button; nothing is recorded as an email or COMPLETED while unprocessed. `pdfinfo`-missing regression: failing PDF retried, later valid mails ingested, cursor moves on, recovers after backoff. Provider-wide errors (auth, TLS, rate limit) back off at poll level with `Retry-After`. Mutation check (no isolation) fails 2/2.
- [x] **3.2 Cursor and idempotency.** Cursor `(UIDVALIDITY, last_uid)` in `poll_state`, advanced per message after ingest or recorded failure; `new_only` first run (no accidental backfill) or explicit `all`; crash between ingest and cursor update, UIDVALIDITY change, repeated polls and mail arriving between polls create no duplicates (Message-ID, attachment hash, job key UNIQUE). Deterministic document id per attachment (`YYYYMMDDTHHMMSS_<hash>`), `BillService.process` returns the finished record for a reused id and cleans partial leftovers: a crash mid-processing resumes with the same id and one document.
- [x] **3.3 Webhook.** The receiver is provider-neutral (signed `{event_id, message_id}`): no provider-specific push contract (Gmail Pub/Sub, Graph subscriptions deliver history/resource ids, not RFC 5322 Message-IDs) is implemented. IMAP `fetch_by_id` = `UID SEARCH HEADER Message-ID` (read-only). A job is marked done only after durable ingest; missing message / transient / permanent errors retry with backoff and end dead-lettered and visible; repeated deliveries create one job and one email.
### Phase 4 bill content
- [x] **4.1** Body/HTML/forwarded bills: bill-likelihood signal (cue labels + amount mentions; text discarded) -> email `REVIEW_REQUIRED/possible_bill_in_body`; no bill and no attachment -> `NO_DOCUMENT`; neither is ever `COMPLETED`. Body-to-document conversion is **not** implemented (flagged for review, as allowed).
- [x] **4.2** Unsupported, corrupt, empty, executable, archive (rejected, never extracted: no archive-bomb surface), too many/too large attachments reported per attachment with codes; multiple attachments independent; existing OCR/classification/Task A/Task B path unchanged.
### Phase 5 provider adapter
- [x] **5.1 IMAP adapter** (`provider.py`): verified TLS, read-only mailbox (EXAMINE; no STORE/EXPUNGE unless `HRBC_IMAP_MARK_SEEN=1`), password or **XOAUTH2** (`oauth.py`: RFC 6749 refresh-token flow, caching, expiry, rotating refresh tokens persisted 0600, permanent (`invalid_grant`...) vs transient (429/5xx/network, Retry-After) errors, https required in prod), classified provider errors, cursor listing, fetch-by-id, per-poll backoff. **OAuth2 is implemented and tested against a local stub token endpoint and a local XOAUTH2 IMAP stub only; it has not been exercised with Google or Microsoft** (the one-time interactive consent that yields the refresh token is a human step and not implemented; scopes are documented in `oauth.py`/`runbook`).
- [x] **5.2 Test-mailbox guard** (`config.validate`): `HRBC_ENV=test` requires `HRBC_TEST_MAILBOX_USER`/`HRBC_TEST_FOLDER` equal to the configured user/folder, an explicit separate `HRBC_OUT`, and forbids mark-seen; `HRBC_TEST_*` is an error in prod.
### Phase 6 verification
- [x] 6.1 75 new regression tests (+ updated `test_ingest.py`); 6.2 the local IMAP smoke test was rewritten for the hardened provider and completes (`smoke/local_imap_smoke_result.json`; the pre-hardening result is kept next to it).
### Phase 7 operation
- [x] **7.1** `config.validate` / `ingest.run check-config` (never prints values; worker exits 2), `health` JSON (exit 0/1), codes-only logs, single-instance lock (second worker exits 3), SIGTERM/SIGINT graceful stop (finish job, record stop, release lock; SIGKILL recovered by lease), `failures` / `requeue-dead` commands, unit template (`UMask=0077`, hardening flags, `ExecStartPre=check-config`), env template, runbook. **Not enabled, not installed, not running.**

## Remaining blockers and follow-ups
1. **Live verification**: a dedicated test mailbox + credentials, then a controlled run (blocker for any claim about a real provider).
2. **Real application authentication** (SSO): until then employee identity is self-asserted; do not expose the UI or data directory.
3. OAuth consent/refresh-token provisioning and provider-specific quirks (blocker only if the provider disallows app passwords).
4. A person must install/enable the supervisor and a monitor on `health`; verify continuous operation in that environment.
5. Follow-ups, not done here: multi-bill PDF splitting, historical backfill beyond `HRBC_IMAP_INITIAL=all`, body-to-document conversion, Gmail API/Graph adapters and push subscriptions, malware scanning, SQLite encryption at rest/Postgres for multiple workers, notifying on rejected senders.
