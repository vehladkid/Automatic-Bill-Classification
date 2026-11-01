"""Fixed income-tax quota per bill label (config/bill_tax_quota.json), default employee profiles, and the per-bill rebate shown after classification.
ESTIMATE ONLY: the numbers come from config/tax_rules (SECONDARY_SOURCES_ONLY) and the saved salary profile; HR/payroll decides."""
import json
from pathlib import Path
from . import engine as E, store as ST, bill_effect as BE
CFG = Path(__file__).resolve().parents[2] / "config" / "bill_tax_quota.json"
_cfg = lambda: json.loads(CFG.read_text())

def seed_default_profiles():
    """Save the default employee profiles for every configured FY, only where none exists yet (never overwrites an employee's edits). Idempotent; call at app/worker start."""
    made = []
    for eid, d in _cfg()["default_profiles"].items():
        if eid.startswith("_"): continue
        for fy in E.available_years():
            if ST.get_profile(eid, fy) is None:
                ST.save_profile(eid, fy, d["name"], dict(d["profile"]), actor=eid, role="employee"); made.append((eid, fy))
    return made

def default_employee_ids(): return [k for k in _cfg()["default_profiles"] if not k.startswith("_")]

def label_quota(label, fy=None):
    """-> dict(label, claim_type, section, regime, cap, cap_unit, quota_text). cap is the rupee limit from the FY rules (None = formula / no cap / no deduction)."""
    c = dict(_cfg()["labels"].get(label) or _cfg()["labels"]["non_expense"]); out = dict(label=label, claim_type=c.get("claim_type"), section=c.get("section"), regime=c.get("regime"), cap=None, cap_unit=c.get("cap_unit", "per year"), quota_text=c["quota_text"])
    if c.get("cap_path"):
        try:
            v = E.load_rules(fy or E.available_years()[-1])
            for k in c["cap_path"].split("."): v = v[k]
            out["cap"] = v
        except Exception: pass
    return out

import re
_MONEY = re.compile(r"\d[\d,]*\.\d{2}")
_AMT_KEYS = ("total deposit", "total contribution", "received rs", "receivedrs", "premium paid", "total amount", "grand total", "net amount", "amount paid", "total")
def _num(x): return float(x.replace(",", ""))
def literal_fallback(ocr_text, fields):
    """Fill ONLY a missing total_amount / invoice_date from text literally printed on the document (no guessing): the last 2-decimal amount on a 'total / received / deposits' line,
    the printed 'Date dd-mm-yyyy', else a printed 'Financial Year YYYY-YY' (mapped to its first day). Returns (fields, notes, extra_details: tuition_component / meals when printed)."""
    fields = dict(fields or {}); notes = []; extra = {}; t = str(ocr_text or ""); lines = t.splitlines()
    if not (fields.get("total_amount") or {}).get("value"):
        done = False
        for key in _AMT_KEYS:
            for ln in lines:
                if key in ln.lower().replace(" ", "") or key in ln.lower():
                    m = _MONEY.findall(ln)
                    if m: fields["total_amount"] = dict(value=_num(m[-1]), source="ocr_literal_fallback", evidence=ln.strip()[:80]); notes.append(f"amount read literally from: '{ln.strip()[:60]}'"); done = True; break
            if done: break
    mfy = re.search(r"financial\s*year\s*[:\-]?\s*(\d{4})\s*[-/]\s*(\d{2,4})", t, re.I)
    if mfy:                                            # a printed 'Financial Year' beats a stray date on the page (e.g. an account-opening date)
        fields["invoice_date"] = dict(value=f"{mfy.group(1)}-04-01", source="ocr_literal_fallback", evidence=mfy.group(0)); notes.append("financial year printed on the document (mapped to its first day)")
    elif not (fields.get("invoice_date") or {}).get("value"):
        m = re.search(r"\bdate\s*[:\-]?\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", t, re.I)
        if m: fields["invoice_date"] = dict(value=f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}", source="ocr_literal_fallback", evidence=m.group(0)); notes.append("date read literally from the document")
    for ln in lines:                                    # tuition head printed on a fee receipt -> the only part of the receipt that qualifies under 80C
        if re.search(r"tuition\s*fee", ln, re.I) and _MONEY.findall(ln): extra["tuition_component"] = _num(_MONEY.findall(ln)[-1]); notes.append(f"tuition component read literally from: '{ln.strip()[:50]}'"); break
    for pat, sub in ((r"public\s*provident\s*fund|\bppf\b", "ppf"), (r"\belss\b", "elss"), (r"national\s*savings?\s*certificate|\bnsc\b", "nsc"), (r"tax[- ]?saver", "tax_saver_fd"), (r"sukanya", "sukanya_samriddhi"), (r"life\s*insurance|\blic\b", "life_insurance")):
        if re.search(pat, t.replace(" ", " "), re.I) or re.search(pat.replace(r"\s*", ""), t, re.I): extra["subtype"] = sub; notes.append(f"instrument printed on the document: {sub}"); break
    meals = [int(x) for x in re.findall(r"credit\D{0,2}(\d+)\s*coupon", t, re.I)]
    if meals: extra["meals"] = sum(meals); notes.append(f"{sum(meals)} coupons counted from the statement lines")
    if not (fields.get("invoice_date") or {}).get("value"):
        m = re.search(r"(?:statement\s*)?period\s*[:\-]?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})", t, re.I)
        if m: mon = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"].index(m.group(1).lower()) + 1; fields["invoice_date"] = dict(value=f"{m.group(2)}-{mon:02d}-01", source="ocr_literal_fallback", evidence=m.group(0)); notes.append("statement period printed on the document")
    return fields, notes, extra

def demo_quota(label):
    """Notional yearly allowance (rupees) for a label WITHOUT a statutory deduction, or None when the demo switch is off / the label has a real section."""
    d = _cfg().get("demo_rebate") or {}
    if not d.get("enabled") or label_quota(label)["claim_type"]: return None
    return float((d.get("quota_by_label") or {}).get(label, d.get("default_quota", 0))) or None

def _amount_fy(rec, fields, ocr_text=None):
    from . import autofill as AF
    fields, _, _ = literal_fallback(ocr_text, fields) if ocr_text else (fields or {}, [], {})
    amt = (fields.get("total_amount") or {}).get("value"); d = (fields.get("invoice_date") or {}).get("value")
    return (float(amt) if isinstance(amt, (int, float)) and amt > 0 else None), (AF.DF.financial_year_of(d) if d else None)

def lower_income(prof, amount):
    """Profile copy whose taxable salary is lower by `amount` (the engine ignores negative other_income, so the notional allowance is taken off special allowance, then bonus)."""
    p = dict(prof); left = float(amount)
    for k in ("special_allowance", "bonus"):
        take = min(left, float(p.get(k) or 0)); p[k] = float(p.get(k) or 0) - take; left -= take
    return p

def _demo_fy(fy):
    """DEMO only: a bill whose date is missing or outside the configured tax years is counted in the latest configured FY (clearly noted)."""
    yrs = E.available_years()
    if fy in yrs: return fy, None
    return yrs[-1], f"demo: the bill's own financial year ({fy or 'not readable'}) has no configured tax rules, so it is counted in FY {yrs[-1]}"

def _demo_effect(rec, fields, owner, quota, ocr_text):
    """Saving from a notional allowance: real slabs/cess, taxable income reduced by min(bill amount, label quota). DEMO ONLY."""
    q = label_quota(rec.get("category") or "non_expense"); amt, fy = _amount_fy(rec, fields, ocr_text)
    eff = dict(estimate_only=True, demo_allowance=True, synthetic_profile=False, gst_on_bill=BE.gst_components(fields or {}), gst_note="Shown for information only; GST is not an income-tax rebate.", what_if_assumptions=[], reasons=[], missing=[])
    if not owner: return dict(eff, kind="cannot_estimate", income_tax_saving=None, reason="The document has no owner (employee), so no tax profile applies.")
    if not amt: return dict(eff, kind="cannot_estimate", income_tax_saving=None, reason="The bill amount could not be read from the document; nothing was guessed.")
    fy, fy_note = _demo_fy(fy)
    if fy_note: eff["what_if_assumptions"] = [fy_note]
    prof = ST.get_profile(owner, fy)
    if not prof: return dict(eff, kind="cannot_estimate", income_tax_saving=None, reason=f"{owner} has no saved salary profile for FY {fy}.")
    rules = E.load_rules(fy); regime = prof.get("chosen_regime") if prof.get("chosen_regime") in ("old", "new") else "old"; elig = min(amt, quota)
    base = E.compute_regime(prof, [], regime, rules, mode="projected")["total_tax"]; p2 = lower_income(prof, elig)
    withb = E.compute_regime(p2, [], regime, rules, mode="projected")["total_tax"]
    return dict(eff, kind="demo_estimate", amount_on_bill=amt, eligible_amount=elig, section="DEMO allowance", tax_without=base, tax_with=withb, income_tax_saving=base - withb, regime=regime, fy=fy,
                assumption="Employee's saved salary profile" + ("" if prof.get("_verified_by") else " (not verified by HR)"), rules_status=rules["verification"]["status"],
                reasons=[f"DEMO: a notional fixed allowance of ₹{quota:,.0f}/yr for '{q['label']}' bills (not an income-tax rule); ₹{elig:,.0f} of this ₹{amt:,.0f} bill counted."])

def demo_allowance(employee_id, fy, recs):
    """Total notional allowance for the employee's bills of this FY (per-label yearly cap shared by all that label's bills) and the per-bill table. Used by the live Tax Summary."""
    import service as S
    used, rows = {}, []
    for rec in sorted(recs, key=lambda r: r.get("uploaded_at") or ""):
        if rec.get("owner") != employee_id or rec.get("status") not in ("done", "done_without_fields"): continue
        label = rec.get("category") or ""; cap = demo_quota(label)
        if not cap: continue
        f_ = S.OUT / rec["result_dir"] / "ocr.txt"; txt = f_.read_text(encoding="utf-8") if f_.exists() else ""
        fields = ((rec.get("task_b_enhanced") or {}).get("fields")) or ((rec.get("task_b") or {}).get("fields")) or {}
        amt, bfy = _amount_fy(rec, fields, txt)
        bfy, _n = _demo_fy(bfy)
        if not amt or bfy != fy: rows.append((rec["doc_id"], label, None, "amount unreadable" if not amt else f"document is for FY {bfy}")); continue
        take = max(0.0, min(amt, cap - used.get(label, 0.0))); used[label] = used.get(label, 0.0) + take; rows.append((rec["doc_id"], label, take, f"bill ₹{amt:,.0f}, label quota ₹{cap:,.0f}/yr"))
    return sum(used.values()), rows

def bill_rebate(rec, fields, employee_id=None, ocr_text=None):
    """Quota + estimated rebate for one processed bill. Works for web uploads (owner = signed-in employee) and e-mailed bills (owner = authorised sender).
    The claim type comes from the document-profile rules, else from the classified label (so a second-stage label such as rent_receipt always maps to its section)."""
    label = rec.get("category") or (rec.get("task_a") or {}).get("label") or "non_expense"
    q = label_quota(label); rec2 = dict(rec); prof = dict(rec.get("profile") or {})
    if not prof.get("primary_claim_type") and q["claim_type"]: prof["primary_claim_type"] = q["claim_type"]; rec2["profile"] = prof
    dq = demo_quota(label) if not q["claim_type"] else None
    if dq and not rec.get("profile", {}).get("primary_claim_type"):
        q = dict(q, section="DEMO allowance", cap=dq, cap_unit="per year (demo)", claim_type="demo", quota_text=f"DEMO (temporary): fixed notional allowance of ₹{dq:,.0f} per year for '{label}' bills. Not a statutory deduction - " + q["quota_text"])
        return dict(quota=q, effect=_demo_effect(rec, fields or {}, employee_id or rec.get("owner"), dq, ocr_text))
    if not prof.get("primary_claim_type") and not q["claim_type"]:
        return dict(quota=q, effect=dict(kind="no_income_tax_deduction", income_tax_saving=0, reason=q["quota_text"], gst_on_bill=BE.gst_components(fields or {}), gst_note="Shown for information only; GST is not an income-tax rebate.", synthetic_profile=False, what_if_assumptions=[]))
    fields, fb, extra = literal_fallback(ocr_text, fields) if ocr_text else (fields or {}, [], {})
    eff = BE.estimate(rec2, fields, employee_id or rec.get("owner"), extra_details=extra)
    if eff.get("kind") == "cannot_estimate":
        from . import autofill as _AF
        d = (fields.get("invoice_date") or {}).get("value"); fy = _AF.DF.financial_year_of(d) if d else None
        if fy and fy not in E.available_years(): eff["reason"] = f"The document belongs to FY {fy}, which has no tax rules configured (config/tax_rules has {', '.join(E.available_years())}); no saving can be estimated."
    if fb and eff.get("kind") == "estimate": eff.setdefault("what_if_assumptions", []).extend("extraction fallback: " + n for n in fb)
    return dict(quota=q, effect=eff)

def submit_bills_to_hr(recs):
    """Every processed, tax-relevant bill that belongs to an employee and is not yet part of any claim becomes a SUBMITTED claim, so it shows under 'Needs approval' in HR Review
    (the employee can still edit it; HR approves / rejects / requests documents). Idempotent. Returns the new claim ids."""
    made = []; allc = ST.list_claims(); declared = {d for c in allc for d in (c.get("docs") or [])}
    for emp in sorted({r.get("owner") for r in recs if r.get("owner")}):
        todo = [r for r in recs if r.get("owner") == emp and r["doc_id"] not in declared]
        derived, _ = derive_bill_claims(emp, None, todo, ())
        for c in derived:
            cl = ST.add_claim(emp, c["fy"], c["type"], c["amount"], c["details"], subtype=c.get("subtype"), docs=c["docs"], doc_hashes=c["doc_hashes"],
                              source=dict(c["source"], auto_submitted=True), actor=emp, role="employee")
            ST.mark_submitted(cl["id"], emp); made.append(cl["id"])
    return made

def derive_bill_claims(employee_id, fy, recs, existing_claims=()):
    """Turn the employee's PROCESSED tax-relevant bills into projected claims so the Tax Summary moves with every bill. Returns (claims, skipped[(doc_id, label, reason)]).
    A bill already attached to a declared claim is not counted twice. Amount / FY come from the document (extraction + literal fallback); nothing is guessed."""
    from . import autofill as AF
    import service as S
    out, skipped = [], []; declared = {d for c in existing_claims for d in (c.get("docs") or [])}
    for rec in sorted(recs, key=lambda r: r.get("uploaded_at") or ""):
        if rec.get("owner") != employee_id or rec.get("status") not in ("done", "done_without_fields"): continue
        label = rec.get("category") or ""; q = label_quota(label)
        if not q["claim_type"]: continue
        if rec["doc_id"] in declared: skipped.append((rec["doc_id"], label, "already part of a declared claim")); continue
        try:
            f_ = S.OUT / rec["result_dir"] / "ocr.txt"; txt = f_.read_text(encoding="utf-8") if f_.exists() else ""
            fields = ((rec.get("task_b_enhanced") or {}).get("fields")) or ((rec.get("task_b") or {}).get("fields")) or {}
            fields, notes, extra = literal_fallback(txt, fields)
            r2 = dict(rec); pf = dict(rec.get("profile") or {})
            if not pf.get("primary_claim_type"): pf["primary_claim_type"] = q["claim_type"]
            r2["profile"] = pf; af = AF.autofill(r2, fields, employee_id)
            if not af.get("amount") or not af.get("fy"): skipped.append((rec["doc_id"], label, "amount or financial year not readable on the document")); continue
            if fy and af["fy"] != fy: skipped.append((rec["doc_id"], label, f"document is for FY {af['fy']}")); continue
            if af["fy"] not in E.available_years(): skipped.append((rec["doc_id"], label, f"no tax rules configured for FY {af['fy']}")); continue
            det = dict(extra, **af["details"], document_amount=af["amount"])
            if af["claim_type"] == "health_insurance" and not det.get("payment_mode"): det["payment_mode"] = "online"    # what-if, same assumption as the per-bill estimate
            out.append(dict(id="bill-" + rec["doc_id"], employee_id=employee_id, fy=af["fy"], type=af["claim_type"], subtype=extra.get("subtype"), amount=af["amount"], details=det, docs=[rec["doc_id"]],
                            doc_hashes=[rec.get("sha256")], source={"derived_from_processed_bill": True, "label": label}, hr_status="submitted", hr_note=None, hr_reviewer=None,
                            created_at=rec.get("uploaded_at") or "", updated_at=""))
        except Exception as e: skipped.append((rec["doc_id"], label, f"{type(e).__name__}"))
    return out, skipped
