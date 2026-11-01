"""Service-layer 'API' for ingested documents. The app has no HTTP backend (Streamlit calls Python in-process) and adding one would introduce a second framework, so the contract is
this module; every function takes a Principal and enforces tenant isolation itself (employee -> own documents only; hr -> all). Unauthorised and non-existent look identical (no enumeration)."""
import json, re
from collections import namedtuple
from . import db
import service as SV
from tax.store import PermissionDenied
import doc_fields as DF
S = DF.S
Principal = namedtuple("Principal", "role user_id")
DENY = "Not found or not permitted."

def _can(p, owner): return p.role == "hr" or (p.role == "employee" and bool(owner) and owner == p.user_id)
def _need(p, owner):
    if not _can(p, owner): raise PermissionDenied(DENY)

def _att_by_doc(c):
    return {r["doc_id"]: dict(r) for r in c.execute("SELECT a.*, e.sender, e.subject, e.received_at, e.provider, e.message_id, e.account_id, e.status AS email_status FROM attachments a JOIN emails e ON e.id=a.email_id WHERE a.doc_id IS NOT NULL")}

def effective_fields(doc_id, rec=None):
    """final extracted fields with human corrections laid over them (original kept in the corrections table)"""
    rec = rec or SV.load_record(doc_id); base = json.loads(json.dumps(((rec.get("task_b_enhanced") or rec.get("task_b") or {}).get("fields")) or {}, default=float))
    c = db.conn()
    for r in c.execute("SELECT * FROM corrections WHERE doc_id=? ORDER BY id", (doc_id,)):
        base[r["field"]] = dict(base.get(r["field"], {}), value=json.loads(r["corrected"]), source="human_correction", confidence=None, corrected_by=r["actor"], corrected_at=r["ts"], needs_review=False)
    return base

LOW_CONF = 0.6      # below this (or no calibrated confidence) a field is shown as unverified in the inbox

def summary(rec):
    """merchant / amount / date for the inbox table + review reasons. A value is 'verified' only if V1 gave it a calibrated confidence >= LOW_CONF and no LLM conflict flagged it; missing stays None."""
    fl = ((rec.get("task_b_enhanced") or {}).get("fields")) or ((rec.get("task_b") or {}).get("fields")) or {}; out = {}
    for k, f in (("merchant", "merchant_name"), ("amount", "total_amount"), ("date", "invoice_date")):
        x = fl.get(f)
        if not x or x.get("value") is None: out[k] = dict(value=None, verified=False); continue
        c = x.get("confidence"); out[k] = dict(value=x["value"], verified=bool(c is not None and c >= LOW_CONF and not x.get("needs_review")), confidence=c)
    reasons = list(rec.get("warnings") or [])
    ta = rec.get("task_a") or {}
    if ta and ta.get("route") not in (None, "auto"): reasons.append("expense category below confidence threshold")
    if (rec.get("profile") or {}).get("status") == "needs_review": reasons.append("document type needs review")
    for k, v in out.items():
        if v["value"] is None: reasons.append(f"{k} not found")
        elif not v["verified"]: reasons.append(f"{k} unverified")
    return out, reasons

def list_documents(p, status=None, category=None, fy=None, employee=None, doc_type=None, channel=None):
    c = db.conn(); att = _att_by_doc(c); rows = []
    for rec in SV.list_records():
        owner = rec.get("owner")
        if not _can(p, owner): continue
        if employee and owner != employee: continue
        pf = rec.get("profile") or {}; a = att.get(rec["doc_id"]); st = a["state"] if a else rec["status"]
        row = dict(doc_id=rec["doc_id"], owner=owner, filename=rec["original_filename"], uploaded_at=rec["uploaded_at"], channel=(rec.get("source") or {}).get("channel", "web_upload"), status=st,
                   category=rec.get("category"), document_type=pf.get("document_type"), benefit_category=pf.get("benefit_category"), financial_year=(pf.get("validity") or {}).get("financial_year"),
                   email=({k: a[k] for k in ("sender", "subject", "received_at", "email_id", "id", "account_id")} if a else None))
        sm, why = summary(rec); ta = rec.get('task_a') or {}; row.update(category_confidence=ta.get('confidence'), category_route=ta.get('route'), merchant=sm["merchant"], amount=sm["amount"], date=sm["date"], needs_review=bool(why) or st in ("REVIEW_REQUIRED", "FAILED"), review_reasons=why)
        if status and row["status"] != status: continue
        if category and row["category"] != category: continue
        if doc_type and row["document_type"] != doc_type: continue
        if fy and row["financial_year"] != fy: continue
        if channel and row["channel"] != channel: continue
        rows.append(row)
    return rows

def get_document(p, doc_id):
    rec = SV.load_record(doc_id)
    if rec is None: raise PermissionDenied(DENY)
    _need(p, rec.get("owner"))
    c = db.conn(); a = c.execute("SELECT a.*, e.sender, e.subject, e.received_at, e.provider, e.account_id FROM attachments a JOIN emails e ON e.id=a.email_id WHERE a.doc_id=?", (doc_id,)).fetchone()
    return dict(record=rec, fields=effective_fields(doc_id, rec), field_report=rec.get("field_report"), profile=rec.get("profile"), email=dict(a) if a else None,
                corrections=[dict(r) for r in c.execute("SELECT * FROM corrections WHERE doc_id=? ORDER BY id", (doc_id,))])

def list_emails(p, status=None):
    c = db.conn(); q = "SELECT * FROM emails"; a = []
    conds = []
    if p.role != "hr": conds.append("employee_id=?"); a.append(p.user_id)
    if status: conds.append("status=?"); a.append(status)
    if conds: q += " WHERE " + " AND ".join(conds)
    return [dict(r) for r in c.execute(q + " ORDER BY id DESC LIMIT 200", a)]

def email_detail(p, email_id):
    c = db.conn(); e = c.execute("SELECT * FROM emails WHERE id=?", (email_id,)).fetchone()
    if e is None: raise PermissionDenied(DENY)
    _need(p, e["employee_id"])
    return dict(email=dict(e), attachments=[dict(r) for r in c.execute("SELECT * FROM attachments WHERE email_id=? ORDER BY idx", (email_id,))],
                history=[dict(r) for r in c.execute("SELECT * FROM ingest_audit WHERE email_id=? ORDER BY id", (email_id,))])

def correct_field(p, doc_id, field, new_value, reason):
    rec = SV.load_record(doc_id)
    if rec is None: raise PermissionDenied(DENY)
    _need(p, rec.get("owner"))
    if field not in DF.ALL_FIELDS: raise ValueError(f"unknown field {field!r}")
    if not str(reason or "").strip(): raise ValueError("a reason is required for a correction")
    t = DF.ALL_FIELDS[field]
    if t == "amount":
        v = S.parse_amount(new_value)
        if v is None or v < 0: raise ValueError("amount must be a non-negative number")
    elif t == "date":
        v = S.parse_date(str(new_value))
        if not v: raise ValueError("date is not valid")
    elif t == "gstin":
        v = S.norm_gstin(new_value)
        if not v or not S.gstin_valid_format(v): raise ValueError("GSTIN format is not valid")
    else:
        v = str(new_value).strip()
        if not v: raise ValueError("value is empty")
    old = effective_fields(doc_id, rec).get(field, {}).get("value")
    c = db.conn(); c.execute("INSERT INTO corrections(doc_id,field,original,corrected,actor,role,ts,reason) VALUES(?,?,?,?,?,?,?,?)", (doc_id, field, json.dumps(old, default=str), json.dumps(v), p.user_id, p.role, db.now(), reason))
    db.audit(c, p.user_id, "field_corrected", None, None, {"doc_id": doc_id, "field": field}); return v

def retry_attachment(p, attachment_id):
    c = db.conn(); a = c.execute("SELECT * FROM attachments WHERE id=?", (attachment_id,)).fetchone()
    if a is None: raise PermissionDenied(DENY)
    _need(p, a["employee_id"])
    if a["state"] != "FAILED" or a["error_code"] not in ("processing_error", "retries_exhausted"): raise ValueError("Only attachments that failed during processing can be retried; rejected or invalid files must be re-sent.")
    if not a["stored_path"] or not (db.out_dir() / a["stored_path"]).exists(): raise ValueError("The stored copy is gone (retention); ask the employee to resend.")
    c.execute("BEGIN IMMEDIATE")
    c.execute("UPDATE jobs SET state='queued', attempts=0, next_run=0, lease_until=0, last_error=NULL, updated_at=? WHERE attachment_id=?", (db.now(), attachment_id))
    c.execute("UPDATE attachments SET state='QUEUED', error=NULL, error_code=NULL, updated_at=? WHERE id=?", (db.now(), attachment_id)); db.audit(c, p.user_id, "retry", a["email_id"], attachment_id); c.execute("COMMIT")

def notifications(p, unread_only=True):
    c = db.conn(); q = "SELECT * FROM notifications WHERE employee_id=?" + (" AND read=0" if unread_only else "") + " ORDER BY id DESC LIMIT 50"
    return [dict(r) for r in c.execute(q, (p.user_id,))]

def history(p, doc_id):
    d = get_document(p, doc_id); return dict(corrections=d["corrections"], email=d["email"])

def list_failures(p):
    """HR only: dead-lettered / retrying messages and webhook fetch jobs (codes and counts only, no message content)"""
    if p.role != "hr": raise PermissionDenied(DENY)
    return [dict(r) for r in db.conn().execute("SELECT key,state,attempts,max_attempts,error_code,first_seen,updated_at FROM inbox_failures WHERE state IN ('retry','dead') ORDER BY state DESC, first_seen LIMIT 200")]

def requeue_failures(p, key=None):
    if p.role != "hr": raise PermissionDenied(DENY)
    from . import pipeline
    return pipeline.requeue_failures(p.user_id, key)

def list_mail_accounts(p):
    """HR only, READ-ONLY: connected mailboxes and their sync status (no credentials, no credential paths). Connecting / disconnecting is an operator CLI action (python -m ingest.run mail-add / mail-disconnect):
    the app has no authentication, so it exposes no way to link or remove a mailbox. Employees cannot see mailbox connections."""
    if p.role != "hr": raise PermissionDenied(DENY)
    from . import accounts
    return [accounts.public_view(a) for a in accounts.list_accounts()]
