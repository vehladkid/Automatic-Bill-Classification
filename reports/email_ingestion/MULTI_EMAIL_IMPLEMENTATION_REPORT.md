# Multi-mailbox ingestion: implementation report

Date: 2026-10-27. Follows `MULTI_EMAIL_FEASIBILITY_AUDIT.md`. Labels: **[VERIFIED]** run/measured in this session; **[NOT VERIFIED]** not exercised.

## 1. Decision at the authentication boundary

The application has **no user authentication**: `tax_ui.identity()` is a text box (Employee ID), the HR role is protected by an optional PIN, and in non-prod the HR role is open. A submitted employee ID is not proof of identity, so **no web/UI path was built that links, re-authorises or removes a mailbox**. Self-service "connect my Gmail" is **BLOCKED** until real authentication exists.

What was built instead is the part that does not depend on it: an **operator-only CLI** (`python -m ingest.run mail-add / mail-disconnect / mail-list`), the same trust model as the existing `google-auth` and `add-sender --admin` commands (a person with shell access to the host, who can already read `worker.env`). If you do not want operator linking either, remove the three CLI branches in `run.py`; nothing else depends on them. The Streamlit page shows connections **read-only, HR role only**.

Ownership is unchanged: **the sender allow-list still decides which employee a document belongs to.** `mail_accounts.owner_employee_id` is a registration record; it never overrides the sender mapping. (Test: E2's mail arriving in E1's connected inbox is processed for E2.)

## 2. Implemented

| Area | Change |
|---|---|
| Schema | New `mail_accounts` table (stable `id` + `public_id`, owner, provider, provider account id + address, `mailbox_key` (links to the cursor in `poll_state`), `status` active / needs_reauth / disconnected, `credential_ref`, `is_legacy`, `last_sync_at`, `last_error_code`/`_at`, `consecutive_failures`, `next_poll_after`, `created_by`, timestamps). Partial unique indexes: one non-disconnected account per (provider, provider account id) and per (provider, mailbox_key). `emails.account_id`, `inbox_failures.account_id` added. |
| Migration | Appended migrations 3-5 to the existing `MIGRATIONS`/`PRAGMA user_version` mechanism (entries may now be callables). Migration 5 creates `acct_legacy` and stamps old rows in one `BEGIN IMMEDIATE` transaction, idempotent, rolled back on error. `emails UNIQUE(provider, message_id)` was **kept, not rebuilt**: see section 4. |
| Credentials | Never in the database. The row holds a path to a per-account 0600 file under `HRBC_ACCOUNTS_DIR` (default `~/.config/hrbc/accounts`, dir 0700), the project's existing mechanism. Permissions are checked on read. Disconnect deletes only that account's file. |
| OAuth | `google_auth._run_flow` extracted from `authorize()` (same PKCE, `state`, loopback, scope and refresh-token checks; scope is still only `gmail.readonly`). New `authorize_tokens()` returns the token to the caller and writes nothing; `authorize()` behaves as before. |
| Per-account ingestion | `accounts.provider_for` builds one `GmailApiProvider` per account with its own token source and its own cursor key (`GmailApiProvider(key=...)`; legacy still `gmail:me`). `pipeline.poll_accounts` polls each active non-legacy account independently; any error is recorded on that account only. |
| Failure isolation | `poll_once(account=...)` keeps backoff/consecutive-failure state on the account row. Revoked/expired OAuth sets `needs_reauth` and stops polling that account until re-authorised. Retry/dead-letter keys are namespaced per account (`gmail@acct_x:...`) because provider ids are only unique per mailbox. |
| Recipient policy | The registered address of a connected mailbox is accepted as a recipient **only for mail read from that connection** (otherwise mail to `b@...` would be rejected as "not addressed to the expense mailbox"). Fail-closed base configuration (`HRBC_INGEST_MAILBOX`) is still required. |
| Source metadata | `emails.account_id`; the processing `source` dict now carries `account_id` (stored in each `record.json`). |
| Health | `PL.health()` gains `mail_accounts` counts and problems `mail_account_needs_reauth` / `mail_account_poll_error` (no addresses in output). |
| UI/API | `api.list_mail_accounts` (HR only, no credential data). HR-only read-only expander in `inbox_ui.py`; document caption shows the source mailbox. No link/disconnect controls. |
| Worker | `run_worker` registers the legacy mailbox, polls it as before, then `poll_accounts`, then the same `drain` (one job at a time). |

Not changed: `service.py`, models, thresholds, `BillService.LOCK`, worker count, queue design, flock single-instance rule, `tax_ui.py`.

## 3. Files changed

`app/ingest/db.py`, `pipeline.py`, `run.py`, `google_auth.py`, `gmail_api.py`, `policy.py`, `api.py`; new `app/ingest/accounts.py`; `app/inbox_ui.py`; new `tests/test_multi_account.py` (36 tests); this report. (No Git repository exists, so the file list comes from a diff against a pre-change backup.)

## 4. Behaviour notes

- **De-duplication.** Messages: global on `(provider, message_id)` where `message_id` is the RFC Message-ID, so the same message arriving in two connected inboxes is stored once. Because ownership is decided by the sender, this cannot attribute it to the wrong employee. Documents: per employee + content hash, across inboxes. Gmail hex ids are never used as a global key (cursor and failure keys are per account). Limitation: the first inbox to ingest a duplicate is recorded as its source.
- **Cursor.** Still the highest Gmail id per mailbox (documented weakness with back-dated mail is unchanged). Each account has its own `poll_state` row.
- **Concurrency.** Unchanged: one worker process, one job at a time, `LOCK` retained, no new queue. Ingestion (`poll_*`) and processing (`drain`) remain separate functions.
- **Webhook `fetch:` jobs** carry no account and are stored with `account_id` NULL (legacy provider only). [Known gap]
- Only Gmail is supported for additional accounts; IMAP stays legacy-only.

## 5. Test results [VERIFIED]

- New: `tests/test_multi_account.py`, 36 tests, **OK**. Covers all 16 requested areas (migration of an old single-mailbox DB incl. backfill, idempotence, rollback; two connections per user; independent credentials; independent cursors incl. identical Gmail ids in two mailboxes; ingestion from both; ownership; duplicates; independent failures; real `RefreshTokenSource` `invalid_grant`; disconnect; retry/dead-letter/requeue; concurrent polls; single-account regression incl. the real `run_worker` loop; unauthorised linking; tenant isolation).
- Mutation check: re-introducing the old un-namespaced failure key and the constant `gmail:me` cursor made the relevant tests fail; files restored and diffed identical.
- Targeted regression: `test_ingest test_ingest_hardening test_gmail_api test_google_auth test_multi_account` = **165 tests OK** (129 pre-existing + 36 new).
- Full suite: **473 tests, 4 failures + 3 errors + 1 skipped.** The identical 7 fail on the **original code** (baseline run from the backup: 438 tests, same 7). Cause: the GPU was occupied by other processes (a 4.3 GB `xlmr_run.py` job; Task A hit `CUDA out of memory`) and `test_demo_smoke` expects `demo/sample_bills/corrupt_image.png`, which does not exist. `test_hr_folders.test_flow` passes alone and fails only after those. Model-dependent tests were therefore **not verified in a clean GPU state**.
- Real-data migration check on a **copy** of the live worker database: user_version 2 -> 5, 19/19 emails stamped to the legacy `gmail:me` connection, cursor intact. The live database was not touched.
- Not run: any live Google call, any real second account, any end-to-end real-model processing of mail from a second inbox.

## 6. Still blocked / not done

1. **User authentication** (blocks self-service linking, per-user visibility of connections, and any web OAuth callback). Needs a product decision: SSO/login, then bind `state` to the session.
2. **Web OAuth callback.** The current flow is a loopback CLI consent; the deployment is a laptop with no public HTTPS callback.
3. **Google production path** [DOC, not verified]: Testing-mode 7-day refresh-token expiry, test-user list, `gmail.readonly` restricted-scope verification, quotas.
4. Concurrency benchmark on a representative document set; the lock and worker count are untouched.
5. Provider-account identity is the mailbox address (no `openid` scope requested, to avoid widening scopes).

## 7. Manual steps and risks

- **The live worker (PID 132599) still runs the old code and was not restarted.** Restart it to load the new code. The new schema is backward compatible with the old worker (it leaves `account_id` NULL; the next worker start stamps those rows to the legacy account). The first new-code process that opens the live DB (worker, Streamlit, or `health`) performs the migration.
- To connect another mailbox: `cd app && ../.venv/bin/python -m ingest.run mail-add --employee E1 --admin <you>` (browser consent, Gmail account must be a test user while the OAuth app is in Testing). Check with `mail-list`. A `needs_reauth` status means re-run `mail-add` for the same address and employee.
- Disconnect removes the local credential only; **revoke the grant at myaccount.google.com/permissions** as well.
- Backup the live `ingest.db` before restarting on new code (it migrates in place; the migration is additive).
- Risk: the self-asserted identity means employee-level tenant isolation is only as strong as the unauthenticated selector; connections are therefore HR-only. Re-using one OAuth client for all accounts means one client-secret compromise affects all.
