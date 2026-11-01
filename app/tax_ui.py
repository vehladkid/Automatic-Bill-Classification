"""Streamlit pages for the tax declaration module: Tax Declaration (employee), HR Review (HR), Tax Summary (both).  ESTIMATE ONLY - see tax/engine.py."""
import os, json
import pandas as pd
import streamlit as st
import service as S
from tax import engine as E, store as ST

DETAIL_FIELDS = {   # claim type -> [(key, label, kind, options)]
    "hra_rent": [("month", "Rent month (YYYY-MM)", "text", None), ("landlord_name", "Landlord name", "text", None), ("landlord_pan", "Landlord PAN (needed if annual rent > ₹1,00,000)", "text", None)],
    "meal_voucher": [("meals", "Number of meals covered", "number", None)],
    "school_fee": [("child_name", "Child's name", "text", None), ("tuition_component", "Tuition-fee part of the receipt (₹)", "number", None), ("full_time", "Full-time course", "bool", None), ("institution_in_india", "Institution in India", "bool", None)],
    "invest_80c": [("subtype", "Instrument", "select", ["ppf", "elss", "life_insurance", "nsc", "tax_saver_fd", "sukanya_samriddhi", "stamp_duty_registration", "epf_employee"])],
    "home_loan_principal": [], "home_loan_interest": [("occupancy", "Property", "select", ["self", "let_out"])],
    "education_loan_interest": [("repayment_start_fy", "Repayment started in FY (e.g. 2024-25)", "text", None), ("lender_type", "Lender", "select", ["bank", "financial_institution", "other"]), ("principal_included", "Amount includes principal", "bool", None), ("principal_amount", "Principal part (₹)", "number", None)],
    "health_insurance": [("group", "For", "select", ["self_family", "parents"]), ("kind", "Kind", "select", ["premium", "preventive_checkup"]), ("beneficiary_senior", "Senior citizen (60+) covered", "bool", None), ("payment_mode", "Payment mode", "select", ["upi", "netbanking", "card", "cheque", "cash"])],
    "nps_employee": [], "ev_loan_interest": [("sanction_date", "Loan sanction date (YYYY-MM-DD)", "text", None), ("vehicle_is_electric", "Electric vehicle", "bool", None)],
    "ev_lease": [], "ev_reimbursement": [], "other": [("section", "Section claimed", "text", None)]}
LABEL = {"hra_rent": "HRA - rent paid (one claim per month)", "meal_voucher": "Meal vouchers (Sodexo / Pluxee)", "school_fee": "School / tuition fees", "invest_80c": "80C investment / premium",
         "home_loan_principal": "Home loan - principal", "home_loan_interest": "Home loan - interest", "education_loan_interest": "Education loan - interest", "health_insurance": "Health insurance / check-up (80D)",
         "nps_employee": "NPS - own contribution (80CCD(1B))", "ev_loan_interest": "EV loan - interest (80EEB)", "ev_lease": "EV lease (employer benefit)", "ev_reimbursement": "EV reimbursement", "other": "Other (HR-reviewed)"}
STATUS_ICON = {"allowed": "✅", "limited": "🟡", "rejected": "❌", "pending_review": "⏳"}
def inr(x): return f"₹{x:,.0f}"

def identity():
    """Sidebar role/identity. NO authentication: production must use SSO. Optional HR PIN via env HRBC_HR_PIN."""
    st.sidebar.markdown("---"); role = st.sidebar.radio("Acting as", ["Employee", "HR"], key="role", horizontal=True)
    eid = st.sidebar.text_input("Employee ID" if role == "Employee" else "HR user ID", key="actor_" + role, value=("E1" if role == "Employee" else "HR1")).strip()   # defaults: employee E1 / HR user HR1
    ok = bool(eid)
    if role == "HR":
        pin = os.environ.get("HRBC_HR_PIN")
        if os.environ.get("HRBC_ENV", "").lower() == "prod" and not pin: ok = False; st.sidebar.error("HRBC_ENV=prod: the HR role is disabled until HRBC_HR_PIN is set. Note: employee identity is still self-asserted (no real authentication).")
        if pin: ok = ok and st.sidebar.text_input("HR PIN", type="password", key="hrpin") == pin
        if not ok: st.sidebar.warning("Enter your HR ID" + (" and PIN" if pin else ""))
    try:
        from tax import quota as Q; st.sidebar.caption("Demo IDs: " + ", ".join(Q.default_employee_ids()) + " (employees) · HR1 (HR) - identity selector, not authentication")
    except Exception: pass
    return ("hr" if role == "HR" else "employee"), eid, ok

def rules_banner(rules):
    v = rules["verification"]
    st.caption(f"Estimate only - not an official payroll or TDS computation · rules FY {rules['financial_year']} ({v['status'].replace('_', ' ').lower()})")
    with st.expander("About these rules"): st.caption(f"{rules['legal_basis']}. Verification status **{v['status']}** (checked {v['checked_on']}). {v['note']}")

def render_bill_rebate(rec, fields, key=""):
    """Quota + estimated tax rebate for one classified bill (shared by the Results page and the Email Inbox page). ESTIMATE ONLY."""
    from tax import quota as Q
    st.subheader("Tax rebate (estimate)")
    try:
        f_ = S.OUT / rec["result_dir"] / "ocr.txt"; r = Q.bill_rebate(rec, fields, rec.get("owner"), ocr_text=f_.read_text(encoding="utf-8") if f_.exists() else None)
    except Exception as e: st.caption(f"Tax estimate unavailable ({type(e).__name__})."); return
    q, be = r["quota"], r["effect"]; cap = q["cap"]
    quota_txt = "No rebate" if not q["claim_type"] else (f"₹{cap:,.0f} {q['cap_unit']}" if cap else "Formula / see note")
    c1, c2, c3, c4 = st.columns(4); c1.metric("Bill label", q["label"]); c2.metric("Section", q["section"] or "none"); c3.metric("Annual quota", quota_txt)
    if be["kind"] in ("estimate", "demo_estimate"): c4.metric("Estimated yearly tax saving", f"₹{be['income_tax_saving']:,.0f}")
    elif be["kind"] == "no_income_tax_deduction": c4.metric("Estimated yearly tax saving", "₹0")
    else: c4.metric("Estimated yearly tax saving", "n/a")
    if be["kind"] == "no_income_tax_deduction": st.info("No income-tax rebate for this bill type.")
    elif be["kind"] == "cannot_estimate": st.warning(be["reason"] + (" Sign in as an Employee (sidebar) before uploading so the bill is linked to a salary profile." if "owner" in be["reason"] else ""))
    else:
        st.success(f"₹{be['eligible_amount']:,.0f} of this ₹{be['amount_on_bill']:,.0f} bill counts under {be['section'] or q['section']}: yearly tax ₹{be['tax_without']:,.0f} → ₹{be['tax_with']:,.0f} ({be['regime']} regime, FY {be['fy']}).")
        if be["kind"] == "demo_estimate": st.caption("⚠ Temporary demo allowance - not a statutory deduction.")
        if be["synthetic_profile"]: st.caption("⚠ No saved salary profile: a synthetic demo profile was used.")
    with st.expander("How this is calculated"):
        st.caption("Quota rule: " + q["quota_text"])
        if be["kind"] == "demo_estimate": st.caption("DEMO ALLOWANCE (temporary): this bill type has no statutory deduction, so a notional fixed yearly quota is applied. Switch it off with `demo_rebate.enabled` in config/bill_tax_quota.json before real use.")
        if be["kind"] in ("estimate", "demo_estimate"):
            st.caption(f"{be['assumption']}. Rules status: {be['rules_status']} (not verified against the official text). Estimate only - HR/payroll decides.")
            for w_ in be.get("what_if_assumptions", []): st.caption("What-if assumption: " + w_)
            for r_ in be["reasons"]: st.caption("• " + r_)
            for m_ in be["missing"]: st.caption("Missing: " + m_)
            if be.get("not_eligible_amount"): st.caption(f"Not eligible / over the quota: ₹{be['not_eligible_amount']:,.0f}")
        if be.get("gst_on_bill"): st.caption("GST read from the bill: " + ", ".join(f"{k} ₹{v:,.2f}" for k, v in be["gst_on_bill"].items()) + ". " + be["gst_note"])

def pick_fy():
    yrs = E.available_years(); return st.selectbox("Financial year", yrs, index=len(yrs) - 1, key="fy")

# ------------------------------------------------------------------ employee
def page_declaration(role, actor, ok, recs):
    st.title("Tax Declaration")
    if role != "employee" or not ok: st.info("Choose **Employee** in the sidebar and enter your employee ID."); return
    fy = pick_fy(); rules = E.load_rules(fy); rules_banner(rules)
    prof = ST.get_profile(actor, fy) or {}
    with st.expander("1. Salary profile (CTC and components)", expanded=not prof):
        c = st.columns(3); g = lambda k, d=0.0: float(prof.get(k, d))
        name = c[0].text_input("Name", prof.get("name", "")); ctc = c[1].number_input("Annual CTC (₹)", 0.0, step=10000.0, value=g("ctc")); city = c[2].text_input("City of residence (for metro HRA %)", prof.get("city", ""))
        c = st.columns(4); basic = c[0].number_input("Basic (₹/yr)", 0.0, step=10000.0, value=g("basic")); da = c[1].number_input("DA (₹/yr)", 0.0, step=1000.0, value=g("da")); hra = c[2].number_input("HRA received (₹/yr)", 0.0, step=10000.0, value=g("hra_received")); spl = c[3].number_input("Special/other allowances (₹/yr)", 0.0, step=10000.0, value=g("special_allowance"))
        c = st.columns(4); bonus = c[0].number_input("Bonus (₹/yr)", 0.0, step=10000.0, value=g("bonus")); meal = c[1].number_input("Employer meal vouchers (₹/yr)", 0.0, step=1000.0, value=g("meal_voucher_annual")); epf = c[2].number_input("Employee EPF (₹/yr)", 0.0, step=1000.0, value=g("employee_pf")); nps = c[3].number_input("Employer NPS (₹/yr)", 0.0, step=1000.0, value=g("employer_nps"))
        c = st.columns(4); pt = c[0].number_input("Professional tax (₹/yr)", 0.0, step=100.0, value=g("professional_tax")); oi = c[1].number_input("Other income (₹/yr)", 0.0, step=1000.0, value=g("other_income")); age = c[2].number_input("Age", 18, 100, int(g("age", 30))); tds = c[3].number_input("TDS deducted so far (₹)", 0.0, step=1000.0, value=g("tds_deducted_to_date"))
        regime = st.radio("Regime you intend to choose", ["undecided", "old", "new"], index=["undecided", "old", "new"].index(prof.get("chosen_regime", "undecided")), horizontal=True)
        if ctc and (basic + da + hra + spl + bonus + meal + nps) > ctc * 1.001: st.warning("Components add up to more than CTC - please check.")
        if st.button("Save salary profile"):
            ST.save_profile(actor, fy, name, dict(ctc=ctc, city=city, basic=basic, da=da, hra_received=hra, special_allowance=spl, bonus=bonus, meal_voucher_annual=meal, employee_pf=epf, employer_nps=nps, professional_tax=pt, other_income=oi, age=age, tds_deducted_to_date=tds, chosen_regime=regime), actor); st.rerun()
    # ---- add claim
    pre = st.session_state.pop("prefill", None)
    st.subheader("2. Add a declaration item")
    if pre: st.info(f"Pre-filled from bill {pre.get('doc_id')}: please check every value. This is a suggestion, not an eligibility decision.")
    types = list(LABEL); ctype = st.selectbox("Type", types, index=types.index(pre["claim_type"]) if pre and pre["claim_type"] in types else 0, format_func=LABEL.get)
    c1, c2 = st.columns(2); amount = c1.number_input("Amount claimed (₹)", 0.0, step=500.0, value=float(pre.get("amount") or 0) if pre else 0.0)
    linked = c2.multiselect("Supporting bills (processed in this app)", [r["doc_id"] for r in recs], default=[pre["doc_id"]] if pre else [], format_func=lambda i: next((f"{r['original_filename']} · {r.get('category')}" for r in recs if r["doc_id"] == i), i))
    details = {}
    for key, label, kind, opts in DETAIL_FIELDS.get(ctype, []):
        dflt = (pre or {}).get("details", {}).get(key)
        if kind == "text": details[key] = st.text_input(label, dflt or "", key=f"d_{ctype}_{key}")
        elif kind == "number": details[key] = st.number_input(label, 0.0, value=float(dflt or 0), key=f"d_{ctype}_{key}")
        elif kind == "bool": details[key] = st.checkbox(label, value=bool(dflt) if dflt is not None else key in ("full_time", "institution_in_india", "vehicle_is_electric"), key=f"d_{ctype}_{key}")
        else: details[key] = st.selectbox(label, opts, key=f"d_{ctype}_{key}")
    if pre:
        for cv in pre.get("caveats", []): st.caption("• " + cv)
    if st.button("Add to my declaration (draft)", type="primary"):
        by = {r["doc_id"]: r for r in recs}
        try:
            sub = details.pop("subtype", None) if ctype == "invest_80c" else None
            vend = next((((by[i].get("task_b_enhanced") or by[i].get("task_b") or {}).get("fields", {}).get("merchant_name") or {}).get("value") for i in linked), None)
            dn = next((((by[i].get("task_b_enhanced") or by[i].get("task_b") or {}).get("fields", {}).get("invoice_number") or {}).get("value") for i in linked), None)
            dd = next((((by[i].get("task_b_enhanced") or by[i].get("task_b") or {}).get("fields", {}).get("invoice_date") or {}).get("value") for i in linked), None)
            details.update({k: v for k, v in dict(vendor=vend, doc_number=dn, doc_date=dd).items() if v})
            cl = ST.add_claim(actor, fy, ctype, amount, details, sub, linked, [by[i]["sha256"] for i in linked if i in by], source={"suggested": bool(pre)}, actor=actor)
            dup = E.find_duplicates(ST.list_claims(actor, fy)); st.success(f"Added draft claim {cl['id']}.")
            if cl["id"] in dup: st.warning(f"This looks like a duplicate of claim {dup[cl['id']]} and will be counted only once.")
        except (ST.StoreError, ST.PermissionDenied) as e: st.error(str(e))
    # ---- my declaration
    st.subheader("3. My declaration")
    claims = ST.list_claims(actor, fy)
    if not claims: st.info("No items yet."); return
    if not prof: st.warning("Save your salary profile to see the tax effect."); 
    res = E.compute_regime(prof, claims, "old", rules, "projected")["claim_results"] if prof else {}
    st.dataframe(pd.DataFrame([{"ID": c["id"], "Type": LABEL[c["type"]], "Claimed": c["amount"], "Status": c["hr_status"], "HR note": c.get("hr_note") or "", "Docs": len(c["docs"]),
                                "Eligible (old)": res.get(c["id"], {}).get("eligible"), "Why": " ".join(res.get(c["id"], {}).get("reasons", []))[:220]} for c in claims]), hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    if a.button("Submit all draft items to HR"): st.success(f"Submitted: {ST.submit(actor, fy, actor) or 'nothing to submit'}"); st.rerun()
    dr = [c["id"] for c in claims if c["hr_status"] == "draft"]
    if dr:
        d = b.selectbox("Delete a draft", dr)
        if b.button("Delete draft"): ST.delete_draft(d, actor); st.rerun()

# ------------------------------------------------------------------ HR
def page_hr(role, actor, ok, recs):
    st.title("HR Review")
    if role != "hr" or not ok: st.info("Choose **HR** in the sidebar and sign in with your HR ID."); return
    rules_banner(E.load_rules(E.available_years()[-1]))
    try:                                                         # every tax-relevant bill of an employee (upload or e-mail) goes to HR as a submitted claim; idempotent
        from tax import quota as Q; Q.submit_bills_to_hr(recs)
    except Exception as e: st.caption(f"Could not queue bills for review ({type(e).__name__}).")
    allc = ST.list_claims(); cnt = {k: sum(1 for c in allc if c["hr_status"] == k) for k in ("submitted", "needs_documents", "approved", "rejected")}   # all financial years
    in_claim = {d: c["hr_status"] for c in allc for d in (c.get("docs") or [])}
    m = st.columns(5); m[0].metric("Processed bills", len(recs)); m[1].metric("Needs approval", cnt["submitted"]); m[2].metric("Approved", cnt["approved"]); m[3].metric("Rejected", cnt["rejected"]); m[4].metric("Documents requested", cnt["needs_documents"])
    try:
        from tax import hr_folders; hr_folders.sync(); st.caption(f"Also saved on disk under `{hr_folders.out_dir() / 'hr_review'}` (processed/, pending_approval/, approved/, rejected/, needs_documents/, draft/).")
    except Exception: pass
    with st.expander(f"All processed bills ({len(recs)})", expanded=True):
        if recs: st.dataframe(pd.DataFrame([{"Bill": r["original_filename"], "Employee": r.get("owner") or "-", "Label": r.get("category") or "(unclassified)", "HR status": {"submitted": "needs approval", "needs_documents": "documents requested"}.get(in_claim.get(r["doc_id"]), in_claim.get(r["doc_id"]) or "no tax claim (no deduction, no owner, or amount/date unreadable)"), "Uploaded": r["uploaded_at"], "Doc id": r["doc_id"]} for r in recs]), hide_index=True, use_container_width=True)
        else: st.caption("No bills processed yet.")
    q = st.radio("Show", ["submitted", "needs_documents", "approved", "rejected"], horizontal=True, format_func=lambda k: {"submitted": "Needs approval", "needs_documents": "Documents requested", "approved": "Approved", "rejected": "Rejected"}[k])
    claims = ST.list_claims(None, None, q)
    if not claims: st.info("Nothing in this list."); return
    for c in claims:
        fy = c["fy"]; rules = E.load_rules(fy); prof = ST.get_profile(c["employee_id"], fy) or {}; allc = ST.list_claims(c["employee_id"], fy)
        r = {rg: E.compute_regime(prof, allc, rg, rules, "projected")["claim_results"].get(c["id"]) for rg in E.REGIMES} if prof else {}
        with st.expander(f"{c['employee_id']} · {LABEL[c['type']]} · {inr(c['amount'])} · {c['hr_status']} · {c['id']}"):
            st.json(c["details"], expanded=False)
            for i in c["docs"]:
                rec = S.load_record(i)
                if rec: st.caption(f"Document: {rec['original_filename']} · {rec.get('category')} · {i}"); 
            for rg, x in r.items():
                if x: st.markdown(f"**{rg.title()} regime:** {STATUS_ICON.get(x['status'], '')} {x['status']} · eligible {inr(x['eligible'])} of {inr(x['claimed'])} — " + " ".join(x["reasons"]) + (f"  \n*Missing:* {'; '.join(x['missing'])}" if x["missing"] else ""))
            if not prof: st.warning("Employee has no salary profile, so tax effect cannot be shown.")
            note = st.text_input("Reason / note (required to reject or request documents)", key=f"n_{c['id']}"); nodoc = st.checkbox("Approve without a linked document (needs a written reason)", key=f"nd_{c['id']}") if not c["docs"] else False
            b = st.columns(3)
            for col, dec, lab in zip(b, ("approved", "rejected", "needs_documents"), ("Approve", "Reject", "Request documents")):
                if col.button(lab, key=f"{dec}_{c['id']}"):
                    try: ST.hr_decide(c["id"], dec, note, actor, "hr", allow_without_docs=nodoc); st.rerun()
                    except (ST.StoreError, ST.PermissionDenied) as e: st.error(str(e))
            with st.popover("Audit trail"): st.dataframe(pd.DataFrame(ST.audit_log(c["id"]))[["ts", "actor", "role", "action", "detail"]], hide_index=True)

# ------------------------------------------------------------------ summary
def page_summary(role, actor, ok, recs):
    st.title("Tax Summary")
    if not ok: st.info("Enter your ID in the sidebar."); return
    fy = pick_fy(); rules = E.load_rules(fy); rules_banner(rules)
    emp = actor
    if role == "hr":
        emps = ST.list_employees(); 
        if not emps: st.info("No employee profiles yet."); return
        emp = st.selectbox("Employee", [e["employee_id"] for e in emps], format_func=lambda i: f"{i} · {next(e['name'] for e in emps if e['employee_id'] == i)}")
    prof = ST.get_profile(emp, fy)
    if not prof: st.info("No salary profile saved for this employee/FY."); return
    claims = ST.list_claims(emp, fy)
    view = st.radio("Count", ["live", "projected", "approved"], format_func=lambda m: {"live": "All processed bills + claims (live, updates with every bill)", "projected": "Declared claims incl. submitted-but-unreviewed", "approved": "HR-approved claims only (verified)"}[m], horizontal=True)
    bill_claims, skipped, demo_total, demo_rows = [], [], 0, []
    if view == "live":
        from tax import quota as Q
        bill_claims, skipped = Q.derive_bill_claims(emp, fy, recs, claims)
        claims = [c for c in claims if c["hr_status"] != "draft"] + bill_claims
        demo_total, demo_rows = Q.demo_allowance(emp, fy, recs)
        if demo_total: prof = Q.lower_income(prof, demo_total)          # DEMO: notional allowance on no-deduction bills lowers taxable salary
    mode = "projected" if view in ("live", "projected") else "approved"
    cmp_ = E.compare(prof, claims, rules, mode); o, n = cmp_["old"], cmp_["new"]
    c = st.columns(3); c[0].metric("Old regime tax", inr(o["total_tax"])); c[1].metric("New regime tax", inr(n["total_tax"])); c[2].metric("Lower tax under", cmp_["lower_tax_regime"].upper() + " regime", f"saves {inr(cmp_['saving'])}")
    if view == "live":
        by_doc = {r["doc_id"]: r for r in recs}; used = []
        for c in bill_claims:                                    # statutory: bills turned into claims (HRA, 80C, 80D, 24(b) ...)
            for rg, x in (("old", o), ("new", n)):
                v = x["claim_results"].get(c["id"])
                if v: used.append({"Bill": by_doc[c["docs"][0]]["original_filename"], "Label": c["source"].get("label"), "Basis": f"Statutory - {v.get('section') or LABEL.get(c['type'], c['type'])}", "Regime": rg.title(), "Bill amount (₹)": c["amount"], "Counted (₹)": v["eligible"], "Note": " ".join(v["reasons"])[:140]})
        for doc_id, label, take, note in (demo_rows if demo_total else []):    # DEMO allowance bills
            if take: used.append({"Bill": by_doc[doc_id]["original_filename"], "Label": label, "Basis": "DEMO allowance (not a tax rule)", "Regime": "Both", "Bill amount (₹)": None, "Counted (₹)": take, "Note": note})
        st.subheader("Bills used to give the rebate")
        if used: st.dataframe(pd.DataFrame(used), hide_index=True, use_container_width=True)
        else: st.caption("No processed bill for this employee / financial year contributes a rebate yet.")
        n_bills = sum(1 for r in recs if r.get("owner") == emp)
        st.caption(f"{n_bills} bill(s) for {emp} · projection, not yet HR-approved" + (f" · ₹{demo_total:,.0f} is a temporary DEMO allowance (no statutory deduction), not a tax rule" if demo_total else "")
                   + (" · some bills belong to another financial year: change the year above" if any(str(x[2]).startswith("document is for FY") for x in skipped) else ""))
        if skipped:
            with st.expander(f"Bills not counted ({len(skipped)})"): st.dataframe(pd.DataFrame(skipped, columns=["Doc id", "Label", "Reason"]), hide_index=True, use_container_width=True)
    rows = [("Gross salary", "gross_salary"), ("HRA exemption", ("exemptions", "hra")), ("Meal voucher exemption", ("exemptions", "meal_vouchers")), ("Professional tax", ("exemptions", "professional_tax")), ("Standard deduction", ("exemptions", "standard_deduction")),
            ("Salary income", "salary_income"), ("Home-loan interest (24b)", "house_property_loss"), ("Other income", "other_income"), ("80C", ("chapter_vi_a", "80C")), ("80CCD(1B)", ("chapter_vi_a", "80CCD1B")), ("80CCD(2) employer NPS", ("chapter_vi_a", "80CCD2")),
            ("80D", ("chapter_vi_a", "80D")), ("80E", ("chapter_vi_a", "80E")), ("80EEB", ("chapter_vi_a", "80EEB")), ("Other (HR override)", ("chapter_vi_a", "other")), ("Taxable income", "taxable_income"), ("Slab tax", "slab_tax"), ("87A rebate", "rebate_87a"),
            ("Surcharge", "surcharge"), ("Cess", "cess"), ("Total tax", "total_tax"), ("TDS so far", "tds_deducted_to_date"), ("Balance payable / (refund)", "balance_payable")]
    pick = lambda d, k: d[k[0]][k[1]] if isinstance(k, tuple) else d[k]
    with st.expander("Full tax breakdown (old vs new regime)"):
        st.dataframe(pd.DataFrame([{"Item": l, "Old regime": pick(o, k), "New regime": pick(n, k)} for l, k in rows]), hide_index=True, use_container_width=True)
        for x in (o, n):
            for t in x.get("notes", []): st.caption(f"{x['regime']}: {t}")
        if o["hra_detail"]: st.caption(f"HRA working: {o['hra_detail']}")
    tab_o, tab_n = st.tabs(["Claims - old regime", "Claims - new regime"])
    for tab, x in ((tab_o, o), (tab_n, n)):
        with tab:
            by = {c["id"]: c for c in claims}
            if not x["claim_results"]: st.caption("No declared or bill-derived claims (the demo allowance bills above are not claims).")
            else: st.dataframe(pd.DataFrame([{"": STATUS_ICON.get(v["status"], ""), "Claim": by[k]["id"], "Type": LABEL[v["type"]], "Claimed": v["claimed"], "On document": v.get("document_amount"), "Eligible": v["eligible"], "Rejected": v["rejected"], "Final approved": v.get("approved_amount"), "Status": v["status"], "HR": v["hr_status"],
                                        "Why": " ".join(v["reasons"]), "Missing documents": "; ".join(v["missing"])} for k, v in x["claim_results"].items()]), hide_index=True, use_container_width=True)
            st.write(f"Claimed {inr(x['claimed_total'])} · eligible {inr(x['eligible_total'])} · rejected {inr(x['rejected_total'])} · pending HR: {sum(1 for c in claims if c['hr_status'] in ('submitted', 'draft'))} · missing documents: {len(x['missing_documents'])}")
    st.caption(o["disclaimer"] + " Surcharge, 87A and cess follow the rules file; other income types (capital gains, perquisites beyond those entered) are not modelled.")
