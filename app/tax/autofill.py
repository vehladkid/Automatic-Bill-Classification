"""Document -> proposed declaration item, with provenance for every value. Rules-based; never infers eligibility from a bill alone.
Each value is labelled: extracted (read from the document), employee_declared (salary profile entered by the employee), payroll_verified (profile verified by HR),
calculated (derived by a stated rule), assumption (default we had to pick - always listed). Missing FY / regime / amount => needs_review, not a guess."""
import re, datetime as dt
from . import engine as E, store as ST
import doc_fields as DF
S = DF.S
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

def period_to_months(text):
    """'April 2026' -> ['2026-04']; 'April 2026 to June 2026' -> 3 months; anything else -> []"""
    pts = re.findall(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ,'\-]*(\d{2,4})", str(text or "").lower())
    if not pts: return []
    f = lambda m, y: (int(y) + (2000 if int(y) < 100 else 0), MONTHS[m])
    a = f(*pts[0]); b = f(*pts[-1]) if len(pts) > 1 else a
    out = []; y, m = a
    while (y, m) <= b and len(out) <= 24: out.append(f"{y}-{m:02d}"); y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out

def _v(fields, ext, *names):
    for n in names:
        x = fields.get(n) or ext.get(n)
        if x and x.get("value") not in (None, ""): return x
    return None

def _name_close(a, b):
    a, b = S.norm_text(re.sub(r"\b(mr|mrs|ms|shri|smt|dr)\b\.?", "", str(a or ""), flags=re.I)), S.norm_text(re.sub(r"\b(mr|mrs|ms|shri|smt|dr)\b\.?", "", str(b or ""), flags=re.I))
    return bool(a and b and (a in b or b in a or S.similarity(a, b) >= 0.8))

def autofill(rec, fields, employee_id, existing_claims=(), profile=None, doc_hashes=None):
    """rec: processed record (with 'profile'); fields: effective fields (human corrections applied). Returns a dict; creates nothing."""
    pf = (rec.get("profile") or {}); ct = pf.get("primary_claim_type"); ext = pf.get("ext_fields") or {}; doc_id = rec["doc_id"]
    out = dict(doc_id=doc_id, claim_type=ct, fy=None, regime=None, values=[], conflicts=[], missing=[], duplicates=[], assumptions=[], status="needs_review", notes=[])
    prov = lambda name, x, kind="extracted": dict(field=name, value=x["value"], kind=kind, source=dict(doc_id=doc_id, evidence=x.get("evidence"), bbox=x.get("bbox"), extractor=x.get("source")))
    if not ct: out["notes"].append("No supported tax-document type was detected with enough evidence; nothing is auto-filled."); return out
    if rec.get("owner") not in (None, employee_id): out["notes"].append("Document belongs to another employee."); out["status"] = "denied"; return out
    amt = _v(fields, ext, "rent_amount", "premium_amount", "interest_amount", "total_amount")
    if ct == "home_loan_interest": amt = _v(fields, ext, "interest_amount") or amt
    dd = _v(fields, ext, "invoice_date")
    if amt: out["values"].append(prov("amount", dict(amt, value=float(amt["value"]))))
    else: out["missing"].append("amount (not found on the document - enter it yourself)")
    # financial year: from the document date only, never assumed
    fy = DF.financial_year_of(dd["value"]) if dd else None
    if fy and fy in E.available_years(): out["fy"] = fy; out["values"].append(dict(field="financial_year", value=fy, kind="calculated", source=dict(doc_id=doc_id, rule="FY = 1 Apr - 31 Mar containing the document date", date=dd["value"])))
    elif fy: out["notes"].append(f"Financial year {fy} has no configured tax rules (config/tax_rules); claim needs review."); out["fy"] = None
    else: out["notes"].append("Document date not found, so the financial year cannot be determined.")
    prof = profile or (ST.get_profile(employee_id, out["fy"]) if out["fy"] else None); verified = bool(prof and prof.get("_verified_by"))
    if prof:
        kind = "payroll_verified" if verified else "employee_declared"
        for k in ("city", "basic", "hra_received", "chosen_regime"):
            if prof.get(k) not in (None, "", "undecided"): out["values"].append(dict(field="profile." + k, value=prof[k], kind=kind, source=dict(profile_fy=out["fy"], verified_by=prof.get("_verified_by"), verified_at=prof.get("_verified_at"))))
        if not verified: out["notes"].append("Salary profile has not been verified by HR; profile-derived values are employee-declared.")
        out["regime"] = prof.get("chosen_regime") if prof.get("chosen_regime") in ("old", "new") else None
        if not out["regime"]: out["missing"].append("tax regime not chosen in the salary profile (claims valid only under some regimes are flagged at calculation time)")
    elif out["fy"]: out["missing"].append("salary profile for this financial year (needed to evaluate the claim)")
    details = {}
    if ct == "hra_rent":
        pn = _v(fields, ext, "rental_period"); months = period_to_months(pn["value"]) if pn else []
        ln, lp, tn = _v(fields, ext, "landlord_name"), _v(fields, ext, "landlord_pan"), _v(fields, ext, "tenant_name")
        if len(months) == 1: details["month"] = months[0]; out["values"].append(dict(field="month", value=months[0], kind="calculated", source=dict(doc_id=doc_id, rule="parsed from rental period", text=pn["value"])))
        elif len(months) > 1: out["notes"].append(f"Receipt covers {len(months)} months ({months[0]} to {months[-1]}); create one claim per month - not auto-split."); out["missing"].append("single rent month")
        else: out["missing"].append("rent month (rental period not found)")
        if ln: details["landlord_name"] = ln["value"]; out["values"].append(prov("landlord_name", ln))
        else: out["missing"].append("landlord name")
        if lp: details["landlord_pan"] = lp["value"]; out["values"].append(prov("landlord_pan", lp))
        if tn and prof and prof.get("name") and not _name_close(tn["value"], prof["name"]): out["conflicts"].append(dict(field="tenant_name", document=tn["value"], profile=prof["name"], message="Tenant name on the receipt does not match the employee profile name."))
        if prof and prof.get("hra_received", 0) in (0, None): out["conflicts"].append(dict(field="hra_received", message="Salary profile shows no HRA component; HRA exemption cannot apply."))
    if ct == "school_fee":
        for k in ("student_name", "institution", "fee_type"):
            x = _v(fields, ext, k)
            if x: details[{"student_name": "child_name"}.get(k, k)] = x["value"]; out["values"].append(prov(k, x))
        out["missing"].append("tuition-fee component (only tuition qualifies; read it from the receipt breakup)"); out["notes"].append("Not every school fee is deductible: development, transport, hostel and donation parts are excluded.")
    if ct == "health_insurance":
        for k in ("insurer", "policy_number"):
            x = _v(fields, ext, k)
            if x: out["values"].append(prov(k, x))
        out["missing"].append("beneficiary (self/family or parents) and payment mode - not reliably readable from a receipt")
    if ct in ("home_loan_interest", "education_loan_interest", "ev_loan_interest"):
        x = _v(fields, ext, "principal_amount")
        if x: out["values"].append(prov("principal_amount", x)); out["notes"].append("Principal is not deductible as interest; enter it as a separate 80C (home loan) claim if applicable.")
        out["missing"].append("loan sanction / repayment-start date (needed for 80E and 80EEB windows)")
    if ct == "ev_lease": out["notes"].append("EV lease/reimbursement treatment is decided by payroll/tax team; no amount is auto-allowed.")
    d = _v(fields, ext, "invoice_number"); vend = _v(fields, ext, "merchant_name", "provider", "institution", "insurer", "lender")
    if vend: details["vendor"] = vend["value"]
    if d: details["doc_number"] = d["value"]
    if dd: details["doc_date"] = dd["value"]
    # conflicts / duplicates against what the employee already declared
    for c in existing_claims:
        if doc_hashes and set(doc_hashes) & set(c.get("doc_hashes") or []): out["duplicates"].append(dict(claim_id=c["id"], why="same document already attached to this claim"))
        elif c["type"] == ct == "hra_rent" and details.get("month") and (c.get("details") or {}).get("month") == details["month"]:
            out["duplicates"].append(dict(claim_id=c["id"], why=f"rent for {details['month']} already declared"))
            if amt and abs(float(amt["value"]) - float(c["amount"])) > 0.5: out["conflicts"].append(dict(field="amount", document=amt["value"], existing=c["amount"], message=f"Existing claim {c['id']} has a different amount for the same month."))
        if c["type"] == "hra_rent" and ct == "hra_rent" and details.get("landlord_pan") and (c.get("details") or {}).get("landlord_pan") not in (None, details["landlord_pan"]): out["conflicts"].append(dict(field="landlord_pan", message=f"Landlord PAN differs from claim {c['id']}."))
    out["details"] = details; out["amount"] = float(amt["value"]) if amt else None
    out["status"] = "ready_for_review" if (out["amount"] and out["fy"] and not out["conflicts"] and not out["duplicates"]) else "needs_review"
    if not out["duplicates"] and not out["conflicts"] and out["status"] == "ready_for_review": out["notes"].append("Review every value before adding it; autofill is a draft, not an eligibility decision.")
    return out

def create_claim(af, employee_id, actor, role="employee", edits=None, doc_hashes=None):
    """Create a DRAFT claim from an autofill result, applying the user's edits. Raises if there is nothing safe to create (no fabricated amounts / FY)."""
    edits = edits or {}; amount = edits.get("amount", af.get("amount")); fy = edits.get("fy", af.get("fy"))
    if not amount or not fy: raise ST.StoreError("Amount and financial year are required; autofill could not determine them and they were not guessed.")
    details = dict(af["details"], **edits.get("details", {}))
    if af.get("amount"): details["document_amount"] = af["amount"]          # amount actually read from the document, kept apart from what the employee claims
    cl = ST.add_claim(employee_id, fy, af["claim_type"], amount, details, docs=[af["doc_id"]], doc_hashes=doc_hashes or [], actor=actor, role=role,
                      source=dict(autofill=dict(doc_id=af["doc_id"], values=af["values"], conflicts=af["conflicts"], missing=af["missing"], duplicates=af["duplicates"], created_by=actor, edits=edits)))
    return cl
