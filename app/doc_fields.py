"""Extension field registry for employee/tax documents. These fields EXTEND the Task B schema (scripts/taskB/schema.py) - names that exist there are reused unchanged
(merchant_name, invoice_number, invoice_date, subtotal, tax_total, total_amount, cgst, sgst, igst, hsn_sac, employee_name, currency, payment_method).
Each extension field maps to an existing normaliser type (text / id / amount / date) so grounding and arithmetic checks are shared with app/llm/validate.py.
extract_lite() is a deterministic regex extractor used when the LLM is off; it only returns what is literally printed."""
import re, sys
from pathlib import Path
_T = Path(__file__).resolve().parents[1] / "scripts" / "taskB"
for _p in (_T, _T.parent):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import schema as S

EXT_FIELDS = {"institution": "text", "provider": "text", "insurer": "text", "lender": "text", "landlord_name": "text", "tenant_name": "text", "student_name": "text",
              "fee_type": "text", "policy_number": "id", "landlord_pan": "id", "loan_type": "text", "loan_account": "id", "rent_amount": "amount",
              "premium_amount": "amount", "interest_amount": "amount", "principal_amount": "amount", "taxable_value": "amount", "discount": "amount", "round_off": "amount",
              "rental_period": "text", "period": "text", "financial_year": "text", "payment_status": "text",
              "payment_reference": "id", "student_reference": "id", "fee_period": "text", "vehicle_details": "text", "eligible_fee_amount": "amount"}
ALL_FIELDS = {**S.FIELDS, **EXT_FIELDS}
PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_LBL = lambda labels, cap=r"([A-Za-z][A-Za-z .&'-]{2,60}?)": re.compile(r"(?:%s)\s*[:\-]?\s*%s\s*(?:\n|$|,|;|\s{2,}|\d|(?=\s+[A-Z][A-Za-z]+\s+(?:Name|No|Number)\b))" % ("|".join(labels), cap), re.I)

def financial_year_of(date_iso):
    """'2026-03-12' -> '2025-26'; None if not a valid date"""
    d = S.parse_date(date_iso) if date_iso else None
    if not d: return None
    y, m = int(d[:4]), int(d[5:7]); y0 = y if m >= 4 else y - 1
    return f"{y0}-{str(y0 + 1)[2:]}"

def extract_lite(text):
    """Literal-only extraction. Each value: {'value','source':'regex','evidence'}. Missing stays missing."""
    t = str(text or ""); out = {}
    STOP = re.compile(r"\b(?:rent|for|towards|toward|vide|rs|inr|being|of the|on|dated|amount|sum|paid|pan)\b|[\u20b9\d]", re.I)
    def put(k, v, ev):
        v = STOP.split(v or "")[0] if k in ("student_name", "landlord_name", "tenant_name", "institution", "insurer", "lender") else v
        if v and len(v.strip(" .,:;-")) >= 3 and k not in out: out[k] = dict(value=v.strip(" .,:;-"), source="regex", evidence=ev.strip()[:80])
    for k, labels in (("student_name", ["student name", "name of student", "name of the student"]), ("landlord_name", ["landlord name", "name of landlord", "owner name", "name of owner"]),
                      ("tenant_name", ["tenant name", "name of tenant", "received from"]), ("institution", ["school name", "name of school", "institute", "college name"]),
                      ("insurer", ["insurer", "insurance company"]), ("lender", ["lender", "bank name", "name of bank"])):
        m = _LBL(labels).search(t)
        if m: put(k, m.group(1), m.group(0))
    m = re.search(r"policy\s*(?:no|number|#)\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9/-]{4,24})", t, re.I)
    if m: put("policy_number", m.group(1), m.group(0))
    if re.search(r"landlord|owner|rent", t, re.I):
        pans = sorted(set(PAN.findall(t.upper())))
        if len(pans) == 1: put("landlord_pan", pans[0], "single PAN in a rent document")     # with >1 PAN we cannot tell landlord from tenant: left missing on purpose
    m = re.search(r"(?:for the month of|month of|rent for|for the period|period)\s*[:\-]?\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ,'\-]*\d{2,4}(?:\s*(?:to|-|till)\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ,'\-]*\d{2,4})?)", t, re.I)
    if m: put("rental_period", m.group(1), m.group(0))
    return out
