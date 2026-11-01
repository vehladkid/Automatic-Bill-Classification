"""Local SQLite store for employee profiles, declaration claims, HR decisions and an append-only audit log.
DB path: <outputs>/hr/hr.db  (HRBC_OUT respected). Role checks live here (not only in the UI): employees touch only their own claims, only role 'hr' decides,
and nobody approves their own claim. This is a single-machine demo store - there is NO authentication; production needs SSO and a real access policy."""
import os, json, sqlite3, uuid, threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
def db_path(): return Path(os.environ.get("HRBC_OUT", ROOT / "outputs")) / "hr" / "hr.db"
_L = threading.Lock()
CLAIM_TYPES = ["hra_rent", "meal_voucher", "school_fee", "invest_80c", "home_loan_principal", "home_loan_interest", "education_loan_interest",
               "health_insurance", "nps_employee", "ev_loan_interest", "ev_lease", "ev_reimbursement", "other"]
HR_STATES = {"draft", "submitted", "approved", "rejected", "needs_documents"}

class PermissionDenied(Exception): pass
class StoreError(Exception): pass

def _now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def conn():
    from ingest import perms
    p = db_path(); perms.private_dir(p.parent.parent); perms.private_dir(p.parent)
    c = sqlite3.connect(p, timeout=10); c.row_factory = sqlite3.Row; perms.secure_db(p)
    c.executescript("""CREATE TABLE IF NOT EXISTS employees(employee_id TEXT, fy TEXT, name TEXT, profile TEXT, updated_at TEXT, PRIMARY KEY(employee_id, fy));
    CREATE TABLE IF NOT EXISTS claims(id TEXT PRIMARY KEY, employee_id TEXT, fy TEXT, type TEXT, subtype TEXT, amount REAL, details TEXT, docs TEXT, doc_hashes TEXT, source TEXT,
        hr_status TEXT, hr_note TEXT, hr_reviewer TEXT, created_at TEXT, updated_at TEXT);
    CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, role TEXT, action TEXT, claim_id TEXT, detail TEXT);""")
    return c

def _sync_folders():
    try:
        from . import hr_folders; hr_folders.sync()
    except Exception: pass                                      # the on-disk view must never break a database write

def _audit(c, actor, role, action, claim_id=None, detail=None):
    c.execute("INSERT INTO audit(ts,actor,role,action,claim_id,detail) VALUES(?,?,?,?,?,?)", (_now(), actor, role, action, claim_id, json.dumps(detail or {}, default=str)))

def _claim(row):
    d = dict(row)
    for k in ("details", "docs", "doc_hashes", "source"): d[k] = json.loads(d[k]) if d.get(k) else ({} if k in ("details", "source") else [])
    return d

def save_profile(employee_id, fy, name, profile, actor, role="employee"):
    if role != "hr" and actor != employee_id: raise PermissionDenied("An employee can only edit their own salary profile.")
    if not str(employee_id).strip(): raise StoreError("employee_id is required")
    with _L, conn() as c:
        c.execute("INSERT OR REPLACE INTO employees VALUES(?,?,?,?,?)", (employee_id, fy, name, json.dumps(profile), _now())); _audit(c, actor, role, "save_profile", None, {"employee_id": employee_id, "fy": fy})

def verify_profile(employee_id, fy, hr_actor, role):
    """HR marks the salary profile as checked against payroll records. Autofill labels profile-derived values 'payroll_verified' only after this."""
    if role != "hr": raise PermissionDenied("Only HR can verify a salary profile.")
    p = get_profile(employee_id, fy)
    if p is None: raise StoreError("no profile to verify")
    if p.get("employee_id") == hr_actor: raise PermissionDenied("HR users cannot verify their own profile.")
    prof = {k: v for k, v in p.items() if k not in ("employee_id", "name", "fy")}; prof.update(_verified_by=hr_actor, _verified_at=_now())
    with _L, conn() as c: c.execute("UPDATE employees SET profile=?, updated_at=? WHERE employee_id=? AND fy=?", (json.dumps(prof), _now(), employee_id, fy)); _audit(c, hr_actor, role, "verify_profile", None, {"employee_id": employee_id, "fy": fy})

def get_profile(employee_id, fy):
    with conn() as c: r = c.execute("SELECT * FROM employees WHERE employee_id=? AND fy=?", (employee_id, fy)).fetchone()
    return None if r is None else dict(json.loads(r["profile"]), employee_id=employee_id, name=r["name"], fy=fy)

def list_employees():
    with conn() as c: return [dict(r) for r in c.execute("SELECT DISTINCT employee_id, name FROM employees ORDER BY employee_id")]

def add_claim(employee_id, fy, ctype, amount, details=None, subtype=None, docs=None, doc_hashes=None, source=None, actor=None, role="employee"):
    actor = actor or employee_id
    if role != "hr" and actor != employee_id: raise PermissionDenied("An employee can only add claims to their own declaration.")
    if ctype not in CLAIM_TYPES: raise StoreError(f"unknown claim type {ctype!r}")
    try: amount = float(amount)
    except (TypeError, ValueError): raise StoreError("amount must be a number")
    if amount <= 0: raise StoreError("amount must be positive")
    cid = uuid.uuid4().hex[:12]
    with _L, conn() as c:
        c.execute("INSERT INTO claims VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (cid, employee_id, fy, ctype, subtype, amount, json.dumps(details or {}), json.dumps(docs or []), json.dumps(doc_hashes or []),
                                                                         json.dumps(source or {}), "draft", None, None, _now(), _now()))
        _audit(c, actor, role, "add_claim", cid, {"type": ctype, "amount": amount})
    _sync_folders(); return get_claim(cid)

def get_claim(cid):
    with conn() as c: r = c.execute("SELECT * FROM claims WHERE id=?", (cid,)).fetchone()
    return None if r is None else _claim(r)

def list_claims(employee_id=None, fy=None, status=None):
    q, a = "SELECT * FROM claims WHERE 1=1", []
    for col, v in (("employee_id", employee_id), ("fy", fy), ("hr_status", status)):
        if v: q += f" AND {col}=?"; a.append(v)
    with conn() as c: return [_claim(r) for r in c.execute(q + " ORDER BY created_at, id", a)]

def update_claim(cid, actor, role="employee", **changes):
    cl = get_claim(cid)
    if cl is None: raise StoreError("claim not found")
    if role != "hr" and cl["employee_id"] != actor: raise PermissionDenied("An employee can only edit their own claims.")
    ok = {"amount", "details", "docs", "doc_hashes", "subtype"}
    bad = set(changes) - ok
    if bad: raise StoreError(f"cannot change {sorted(bad)}")
    reset = cl["hr_status"] in ("approved", "rejected", "submitted", "needs_documents")
    with _L, conn() as c:
        for k, v in changes.items(): c.execute(f"UPDATE claims SET {k}=? WHERE id=?", (json.dumps(v) if isinstance(v, (dict, list)) else v, cid))
        c.execute("UPDATE claims SET source=? WHERE id=?", (json.dumps(dict(cl["source"], corrections=(cl["source"].get("corrections") or []) + [dict(ts=_now(), actor=actor, role=role, old={k: cl[k] for k in changes}, new=changes)])), cid))
        c.execute("UPDATE claims SET hr_status='draft', hr_note=NULL, hr_reviewer=NULL, updated_at=? WHERE id=?", (_now(), cid)) if reset else None
        _audit(c, actor, role, "update_claim", cid, {"fields": list(changes), "status_reset_to_draft": reset})
    _sync_folders(); return get_claim(cid)

def delete_draft(cid, actor, role="employee"):
    cl = get_claim(cid)
    if cl is None: raise StoreError("claim not found")
    if role != "hr" and cl["employee_id"] != actor: raise PermissionDenied("An employee can only delete their own claims.")
    if cl["hr_status"] != "draft": raise StoreError("Only draft claims can be deleted; submitted/decided claims stay on record.")
    with _L, conn() as c: c.execute("DELETE FROM claims WHERE id=?", (cid,)); _audit(c, actor, role, "delete_draft", cid)
    _sync_folders()

def submit(employee_id, fy, actor, role="employee"):
    if role != "hr" and actor != employee_id: raise PermissionDenied("An employee can only submit their own declaration.")
    with _L, conn() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM claims WHERE employee_id=? AND fy=? AND hr_status IN ('draft','needs_documents')", (employee_id, fy))]
        for i in ids: c.execute("UPDATE claims SET hr_status='submitted', updated_at=? WHERE id=?", (_now(), i))
        _audit(c, actor, role, "submit", None, {"claims": ids})
    _sync_folders(); return ids

def mark_submitted(cid, actor):
    """Send ONE draft claim to HR (used when a processed bill is auto-submitted); other drafts of the employee are untouched."""
    cl = get_claim(cid)
    if cl is None: raise StoreError("claim not found")
    if cl["employee_id"] != actor: raise PermissionDenied("An employee can only submit their own claim.")
    with _L, conn() as c:
        c.execute("UPDATE claims SET hr_status='submitted', updated_at=? WHERE id=? AND hr_status='draft'", (_now(), cid)); _audit(c, actor, "employee", "auto_submit_from_bill", cid, {})
    _sync_folders()

def hr_decide(cid, decision, note, reviewer, role, allow_without_docs=False):
    if role != "hr": raise PermissionDenied("Only an HR user can approve or reject claims.")
    if decision not in ("approved", "rejected", "needs_documents"): raise StoreError("decision must be approved / rejected / needs_documents")
    if not str(reviewer or "").strip(): raise StoreError("reviewer name is required for the audit trail")
    cl = get_claim(cid)
    if cl is None: raise StoreError("claim not found")
    if cl["employee_id"] == reviewer: raise PermissionDenied("HR users cannot decide their own claims.")
    if cl["hr_status"] == "draft": raise StoreError("The employee has not submitted this claim yet.")
    if decision in ("rejected", "needs_documents") and not str(note or "").strip(): raise StoreError("A reason is required when rejecting or requesting documents.")
    if decision == "approved" and not cl["docs"] and not (allow_without_docs and str(note or "").strip()): raise StoreError("This claim has no supporting document. Request documents, or approve with an explicit written reason.")
    with _L, conn() as c:
        c.execute("UPDATE claims SET hr_status=?, hr_note=?, hr_reviewer=?, updated_at=? WHERE id=?", (decision, note, reviewer, _now(), cid))
        _audit(c, reviewer, role, "hr_" + decision, cid, {"note": note, "without_docs": bool(allow_without_docs and not cl["docs"])})
    _sync_folders(); return get_claim(cid)

def audit_log(claim_id=None, limit=200):
    with conn() as c:
        rows = c.execute("SELECT * FROM audit WHERE claim_id=? ORDER BY id DESC LIMIT ?", (claim_id, limit)) if claim_id else c.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]
