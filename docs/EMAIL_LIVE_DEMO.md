# Live email-to-bill demo (dedicated test mailbox)

**Status: NOT yet run against a real mailbox.** Everything below is verified only against the local IMAP stub (`reports/email_ingestion/smoke/local_imap_smoke.py`, `tests/test_ingest*.py`). The live smoke test is a manual step because it needs a mailbox credential that only you can create.

## Gmail with Google OAuth (Gmail API, read-only)
**Status.** Implemented and unit-tested against local fake servers. Live, against the real Google account: client JSON validated, refresh token stored and **refreshes successfully**, `gmail-check` **passes** (profile counters + up to 5 INBOX ids, no content read). **Not yet tested live:** message fetch, attachment retrieval, the full email-to-bill run (needs one controlled test email from you).

**Transport and scope.** Gmail API via `app/ingest/gmail_api.py` (`worker --gmail`), scope `https://www.googleapis.com/auth/gmail.readonly` only (operations used: `messages.list`, `messages.get` format metadata/raw, `users.getProfile`; nothing can send, modify or delete). The IMAP transport would need the broad `https://mail.google.com/` scope; gmail.readonly is not valid for IMAP. Google refuses to *narrow* an already-granted scope on refresh (`invalid_scope`), so the stored token keeps working but stays broad until you replace it.
**Your stored token currently has the broad `https://mail.google.com/` grant** (made by the first run of `google-auth`, before the default was changed). To get least privilege: (1) revoke at https://myaccount.google.com/permissions (remove this app), (2) in Google Cloud > OAuth consent screen > Data Access remove `https://mail.google.com/` and keep `gmail.readonly`, (3) re-run `google-auth` (now requests gmail.readonly), (4) `gmail-check` must show `"least_privilege": true`. Google Cloud changes are manual; nothing here changes them.
Other Cloud prerequisites: **Gmail API enabled** for the project (error `accessNotConfigured` otherwise), your account listed under Test users, app in Testing mode (refresh tokens then expire after 7 days: re-run `google-auth`).

**Private files (outside the repo, mode 600; `.gitignore` also blocks them):** `~/.config/hrbc/google_client.json` (client JSON), `~/.config/hrbc/google_refresh_token`, `~/.config/hrbc/worker.env` (paths only; replace `YOUR_GMAIL@gmail.com` in two places with the address of the mailbox).
```bash
cd /home/tejvir-singh/Desktop/ML-project/app && set -a; . ~/.config/hrbc/worker.env; set +a
../.venv/bin/python -m ingest.run google-auth --user <test address>   # only if no valid token / after revoking; opens the browser
../.venv/bin/python -m ingest.run gmail-check                          # read-only smoke test (no mail content)
../.venv/bin/python -m ingest.run check-config --gmail                 # "ok": true
../.venv/bin/python -m ingest.run add-sender <address you send FROM> E1 --admin demo
../.venv/bin/python -m ingest.run worker --gmail --once                # 1st run: records the newest INBOX id only; existing mail is NOT read
# send ONE email with one JPG/PNG/PDF bill to the mailbox, wait a minute, then:
../.venv/bin/python -m ingest.run worker --gmail --once                # ingests mail newer than the cursor (max 5 per poll)
../.venv/bin/python -m ingest.run worker --gmail --once                # repeat: must create nothing new
```
Close the Streamlit app first (6 GB GPU), run the app afterwards (`cd .. && set -a; . ~/.config/hrbc/worker.env; set +a; bash app/run_app.sh`) and open Email Inbox. Local smoke (no Google): `cd tests && ../.venv/bin/python -m unittest test_gmail_api test_google_auth`.

**Troubleshooting.** `credentials_rejected` / `invalid_grant`: token revoked or expired (7-day Testing limit) -> re-run `google-auth`. `invalid_scope` on refresh: do not pass a narrower scope; replace the token as above. `accessNotConfigured`: enable the Gmail API. `access_denied` in the browser: add the account under Test users. Missing/readable-by-others files: `check-config` names the problem (never the value). Sender rejected: Gmail's `Authentication-Results` must show an aligned SPF/DKIM pass; mail you send to yourself may not qualify, so send from a second account.
Limit: cursor = highest Gmail message id seen; back-dated/imported mail with a lower id is not picked up; only INBOX, 3 pages of 100 ids per poll.

## Generic mailbox prerequisites (app-password route)
1. A **dedicated** mailbox (never a personal inbox). Gmail example: new account, 2-Step Verification on, create an **App password** (Google Account > Security), IMAP enabled.
2. Config (secrets in a 0600 file, never committed):
```bash
mkdir -p ~/.config/hrbc && chmod 700 ~/.config/hrbc
printf '%s' 'YOUR-APP-PASSWORD' > ~/.config/hrbc/imap_password && chmod 600 ~/.config/hrbc/imap_password
cp deploy/worker.env.example ~/.config/hrbc/worker.env && chmod 600 ~/.config/hrbc/worker.env
```
Edit `worker.env`: `HRBC_ENV=test`, `HRBC_OUT=$HOME/hrbc_live_demo`, `HRBC_INGEST_MAILBOX=<the mailbox address>`, `HRBC_IMAP_HOST=imap.gmail.com`, `HRBC_IMAP_USER=<same address>`, `HRBC_IMAP_AUTH=password`, `HRBC_IMAP_PASSWORD_FILE=/home/<you>/.config/hrbc/imap_password`, `HRBC_INGEST_AUTHSERV_ID=mx.google.com`, `HRBC_TEST_MAILBOX_USER=<same address>`, `HRBC_TEST_FOLDER=INBOX`, `HRBC_IMAP_INITIAL=new_only`.
Sender checks require an aligned SPF/DKIM/DMARC pass in the `Authentication-Results` header that Gmail writes; mail you send from another Gmail address normally has it.

## Run
```bash
cd /home/tejvir-singh/Desktop/ML-project/app
set -a; . ~/.config/hrbc/worker.env; set +a
../.venv/bin/python -m ingest.run check-config                       # must print "ok": true; missing/invalid settings give a clear error, values are never printed
../.venv/bin/python -m ingest.run add-sender <your-sending-address> E1 --admin demo
../.venv/bin/python -m ingest.run worker --imap --once               # one poll (debug). Without --once: --interval 60 (Ctrl-C = clean stop)
../.venv/bin/python -m ingest.run health
```
Then send an email with a JPG/PNG/PDF bill from the registered address to the mailbox and run the worker again (close the Streamlit app first: the 6 GB GPU fits one model copy). View it:
```bash
cd .. && set -a; . ~/.config/hrbc/worker.env; set +a; bash app/run_app.sh   # sidebar: role employee, ID E1 (or HR) > Email Inbox
```
Pass criteria: one row per attachment (no duplicates after running the worker twice), showing sender, subject, file, status, category, merchant/amount/date (marked "(unverified)" when confidence is low) and the review reasons.

## Behaviour and limits
- Read-only IMAP, cursor = UIDVALIDITY+UID, Message-ID de-duplication, per-message isolation, bounded retries then dead-letter, leases for crashed workers.
- The worker is a separate process; the UI never starts it.
- Bill only in the email body: marked REVIEW_REQUIRED (`possible_bill_in_body`), not converted.
- One PDF with several bills: **not split**; the whole PDF is one record and only page 1 is classified/extracted.
- OAuth2 (XOAUTH2) refresh-token login is implemented and tested against stubs only; the consent step is not implemented, so use an app password for the demo.
- Accounts, passwords and the live smoke test are manual.
