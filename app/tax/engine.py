"""Employee tax estimate engine (India). ESTIMATE ONLY - not an official payroll or statutory computation.
All rates, caps and eligibility limits come from config/tax_rules/fy<YYYY>-<YY>.json (loaded, never hardcoded here). Pure functions, no I/O besides load_rules.

Pipeline: profile + claims --evaluate_claims(regime)--> per-claim {claimed, eligible, rejected, status, reasons, missing}
          --compute_regime--> taxable income, tax, surcharge, cess.  compare() runs both regimes.
Claim 'mode': "approved" counts only HR-approved claims (the figure that is safe to quote); "projected" also counts submitted-but-unreviewed claims."""
import json, hashlib, datetime as dt
from pathlib import Path

RULES_DIR = Path(__file__).resolve().parents[2] / "config" / "tax_rules"
REGIMES = ("old", "new")
SECTION_OF = {"hra_rent": "HRA", "meal_voucher": "meal_vouchers", "school_fee": "80C", "invest_80c": "80C", "home_loan_principal": "80C",
              "home_loan_interest": "24b", "education_loan_interest": "80E", "health_insurance": "80D", "nps_employee": "80CCD1B",
              "ev_loan_interest": "80EEB", "ev_lease": "review", "ev_reimbursement": "review", "other": "other"}
COUNTED = {"approved": {"approved"}, "projected": {"approved", "submitted"}}

class RulesError(Exception): pass

def available_years(): return sorted(p.stem.replace("fy", "") for p in RULES_DIR.glob("fy*.json"))

def load_rules(fy):
    f = RULES_DIR / f"fy{fy}.json"
    if not f.exists(): raise RulesError(f"No tax rules configured for FY {fy} (have: {', '.join(available_years()) or 'none'}). Add config/tax_rules/fy{fy}.json - the app will not guess.")
    r = json.loads(f.read_text())
    if r.get("financial_year") != fy: raise RulesError(f"{f.name} declares FY {r.get('financial_year')}, expected {fy}")
    return r

def fy_bounds(fy):
    y = int(fy[:4]); return dt.date(y, 4, 1), dt.date(y + 1, 3, 31)

def n(x):
    try: return max(0.0, float(x or 0))
    except (TypeError, ValueError): return 0.0

def claim_key(c):
    """Duplicate identity of a claim: same document bytes, or same (type, vendor, doc number, date, amount)."""
    d = c.get("details") or {}
    if c["type"] == "hra_rent" and d.get("month"): return ("rent", c.get("employee_id"), c.get("fy"), d["month"])
    ident = (str(d.get("vendor") or "").strip().casefold(), str(d.get("doc_number") or "").strip().casefold(), str(d.get("doc_date") or ""), round(n(c.get("amount")), 2))
    return (c["type"], c.get("subtype") or "", c.get("employee_id"), c.get("fy")) + ident if any(ident[:3]) else None

def find_duplicates(claims):
    """claim id -> id of the earlier claim it duplicates (same document sha256 or same claim_key)."""
    seen_k, seen_h, dup = {}, {}, {}
    for c in sorted(claims, key=lambda x: (x.get("created_at") or "", str(x["id"]))):
        k = claim_key(c); hs = [h for h in (c.get("doc_hashes") or []) if h]
        first = seen_k.get(k) if k else None
        for h in hs: first = first or seen_h.get(h)
        if first and first != c["id"]: dup[c["id"]] = first; continue
        if k: seen_k[k] = c["id"]
        for h in hs: seen_h[h] = c["id"]
    return dup

# ---------------------------------------------------------------- tax arithmetic
def slab_tax(income, slabs):
    tax, lo = 0.0, 0.0
    for hi, rate in slabs:
        top = income if hi is None else min(income, hi)
        if top > lo: tax += (top - lo) * rate
        if hi is None or income <= hi: break
        lo = hi
    return tax

def _slabs(rules, regime, age):
    r = rules["regimes"][regime]
    if regime == "old" and age >= 80 and "80" in r.get("senior_slabs", {}): return r["senior_slabs"]["80"]
    if regime == "old" and age >= 60 and "60" in r.get("senior_slabs", {}): return r["senior_slabs"]["60"]
    return r["slabs"]

def tax_with_rebate_surcharge_cess(income, regime, rules, age=30):
    """income = total taxable income (already rounded). Returns dict of components. Marginal relief applied for 87A (new regime flag) and surcharge."""
    r = rules["regimes"][regime]; slabs = _slabs(rules, regime, age); base = slab_tax(income, slabs); notes = []
    rb = r["rebate_87a"]; rebate = 0.0
    if income <= rb["income_limit"]: rebate = min(base, rb["max_rebate"])
    elif rb.get("marginal_relief") and base > income - rb["income_limit"]:
        rebate = base - (income - rb["income_limit"]); notes.append("87A marginal relief applied")
    after = base - rebate
    sur_tbl = r["surcharge"]; rate = 0.0; thr = None; prev = 0.0
    for hi, rt in sur_tbl:
        if hi is None or income <= hi: rate = rt; break
        thr, prev = hi, rt
    sur = after * rate
    if rate > 0:   # marginal relief at the lower edge of the bracket: extra tax+surcharge may not exceed the extra income over the threshold
        edges = [h for h, _ in sur_tbl if h is not None and h < income]
        T = max(edges) if edges else None
        if T is not None:
            low_rate = next((rt for h, rt in sur_tbl if h == T), 0.0)
            t_at = slab_tax(T, slabs); t_at -= min(t_at, rb["max_rebate"]) if T <= rb["income_limit"] else 0
            cap = t_at * (1 + low_rate) + (income - T)
            if after + sur > cap: sur = max(0.0, cap - after); notes.append("surcharge marginal relief applied")
    cess = (after + sur) * rules["cess_rate"]
    return dict(slab_tax=round(base), rebate_87a=round(rebate), surcharge=round(sur), cess=round(cess), total_tax=round(after + sur + cess), notes=notes)

# ---------------------------------------------------------------- HRA
def hra_exemption(profile, rent_paid, months, rules):
    h = rules["exemptions"]["hra"]; basic_da = (n(profile.get("basic")) + n(profile.get("da"))) * months / 12.0
    received = n(profile.get("hra_received")) * months / 12.0
    metro = str(profile.get("city") or "").strip().casefold() in {c.casefold() for c in h["metro_cities"]}
    pct = h["basic_pct_metro"] if metro else h["basic_pct_non_metro"]
    a, b, c = received, rent_paid - h["rent_minus_basic_pct"] * basic_da, pct * basic_da
    return dict(exempt=round(max(0.0, min(a, b, c))), actual_hra_received=round(a), rent_minus_10pct_basic=round(b), pct_of_basic=round(c), metro=metro, pct=pct, months=months)

# ---------------------------------------------------------------- claim evaluation
def _res(c, regime, section, eligible=0.0, status=None, reasons=(), missing=()):
    """Four different amounts are kept apart: claimed (employee), document (read from the supporting bill), eligible (rules), approved (HR-approved eligible; None until approved)."""
    claimed = n(c.get("amount")); eligible = max(0.0, min(eligible, claimed)) if c["type"] != "other" else max(0.0, eligible); reasons = list(reasons)
    doc_amt = (c.get("details") or {}).get("document_amount"); doc_amt = n(doc_amt) if doc_amt not in (None, "") else None
    if doc_amt is not None and claimed > doc_amt + 0.5 and c["type"] != "other":
        reasons.append(f"Claimed ₹{claimed:,.0f} exceeds the ₹{doc_amt:,.0f} shown on the supporting document; eligible amount is limited to the document amount."); eligible = min(eligible, doc_amt)
    status = status or ("allowed" if eligible >= claimed and claimed > 0 else "limited" if eligible > 0 else "rejected")
    return dict(claim_id=c["id"], type=c["type"], section=section, regime=regime, claimed=round(claimed, 2), eligible=round(eligible, 2), rejected=round(max(0.0, claimed - eligible), 2),
                status=status, reasons=list(reasons), missing=list(missing), hr_status=c.get("hr_status"),
                document_amount=doc_amt, approved_amount=(round(eligible, 2) if c.get("hr_status") == "approved" and eligible > 0 else (0.0 if c.get("hr_status") == "approved" else None)))

def evaluate_claims(profile, claims, regime, rules, mode="approved"):
    """Returns (results_by_claim_id, totals). Every claim gets an explained result, even ones excluded from the figures."""
    fy = rules["financial_year"]; counted = COUNTED[mode]; out = {}; dup = find_duplicates(claims)
    d = rules["deductions"]; ex = rules["exemptions"]; allowed = set(rules["regimes"][regime]["allowed_deductions"]); lo, hi = fy_bounds(fy)
    pools = {"80C": 0.0, "80CCD1B": 0.0, "80D_self": 0.0, "80D_par": 0.0, "80D_chk": 0.0, "24b": 0.0, "80E": 0.0, "80EEB": 0.0, "other": 0.0}
    pay_epf = n(profile.get("employee_pf")); c80 = d["80C"]["cap"] if regime in d["80C"]["regimes"] else 0
    if "80C" in allowed: pools["80C"] = min(pay_epf, c80)       # payroll EPF is counted first, from the profile
    rent_claims, meal_claims, children = [], [], []
    ordered = sorted(claims, key=lambda x: (x.get("created_at") or "", str(x["id"])))
    for c in ordered:
        sec = SECTION_OF.get(c["type"], "other"); det = c.get("details") or {}; amt = n(c.get("amount")); miss = []; why = []
        if c["id"] in dup: out[c["id"]] = _res(c, regime, sec, 0, "rejected", [f"Duplicate of claim {dup[c['id']]} (same document or same bill/receipt); counted once there."]); continue
        if c.get("fy") not in (None, fy): out[c["id"]] = _res(c, regime, sec, 0, "rejected", [f"Claim is for FY {c.get('fy')}, not {fy}."]); continue
        if c.get("hr_status") == "rejected": out[c["id"]] = _res(c, regime, sec, 0, "rejected", [f"Rejected by HR: {c.get('hr_note') or 'no reason recorded'}"]); continue
        if amt <= 0: out[c["id"]] = _res(c, regime, sec, 0, "rejected", ["Claimed amount is zero or missing."]); continue
        if not c.get("docs"): miss.append("supporting document (receipt / certificate)")
        sec_key = {"HRA": "HRA", "meal_vouchers": "meal_vouchers", "review": "review", "other": "other"}.get(sec, sec)
        if sec_key in ("80C", "24b", "80E", "80D", "80CCD1B", "80EEB") and sec_key not in allowed:
            out[c["id"]] = _res(c, regime, sec, 0, "rejected", [f"Section {sec_key} deductions are not available under the {regime} tax regime."], miss); continue
        if sec_key == "HRA" and regime not in ex["hra"]["regimes"]:
            out[c["id"]] = _res(c, regime, sec, 0, "rejected", [f"HRA exemption is not available under the {regime} tax regime."], miss); continue
        if c["type"] == "hra_rent":
            if not det.get("month") or not str(det["month"])[:7] <= hi.strftime("%Y-%m") or not str(det["month"])[:7] >= lo.strftime("%Y-%m"):
                out[c["id"]] = _res(c, regime, sec, 0, "rejected", [f"Rent month '{det.get('month')}' is missing or outside FY {fy}."], miss); continue
            if not det.get("landlord_name"): miss.append("landlord name")
            rent_claims.append((c, miss)); continue
        if c["type"] == "meal_voucher":
            if not n(det.get("meals")): miss.append("number of meals covered by the vouchers")
            meal_claims.append((c, miss)); continue
        if c["type"] in ("ev_lease", "ev_reimbursement"):
            out[c["id"]] = _res(c, regime, sec, 0, "pending_review", [rules["review_only"]["ev_lease_or_reimbursement"]], miss); continue
        if c["type"] == "other":
            ov = n(det.get("hr_override_amount")); ok = c.get("hr_status") == "approved" and ov > 0 and det.get("hr_override_reason")
            if ok and regime == "old": pools["other"] += ov; out[c["id"]] = _res(c, regime, "other", ov, "limited" if ov < amt else "allowed", [f"HR override: {det['hr_override_reason']}"], miss)
            elif ok: out[c["id"]] = _res(c, regime, "other", 0, "rejected", ["Override recorded but other deductions are not allowed under the new regime."], miss)
            else: out[c["id"]] = _res(c, regime, "other", 0, "pending_review", [rules["review_only"]["other"]], miss)
            out[c["id"]]["counted"] = True
            continue
        # ---- capped deduction families (processed in claim order so caps bite deterministically)
        elig = amt
        if c["type"] == "school_fee":
            tf = n(det.get("tuition_component")); child = str(det.get("child_name") or "").strip().casefold()
            if not child: miss.append("child's name")
            if child and child not in children: children.append(child)
            if child and children.index(child) >= d["80C"]["tuition_fee"]["max_children"]: elig = 0; why.append(f"Tuition fee is deductible for at most {d['80C']['tuition_fee']['max_children']} children; this is child #{children.index(child) + 1}.")
            elif not tf: elig = 0; miss.append("tuition-fee component (receipt breakup)"); why.append("Only the tuition-fee component qualifies; the receipt breakup was not provided, so nothing is allowed yet.")
            else:
                elig = min(tf, amt)
                if elig < amt: why.append(f"Only the tuition component (₹{tf:,.0f}) of the ₹{amt:,.0f} receipt qualifies; development/transport/hostel/other fees are excluded.")
            if det.get("institution_in_india") is False: elig = 0; why.append("Education must be at a school/college/university in India.")
            if det.get("full_time") is False: elig = 0; why.append("Only full-time education qualifies.")
        elif c["type"] == "invest_80c":
            sub = c.get("subtype") or det.get("subtype")
            if sub not in d["80C"]["subtypes"]: elig = 0; why.append(f"'{sub}' is not a recognised 80C instrument in the rules file.")
            elif sub == "epf_employee": elig = 0; why.append("Employee EPF is taken from payroll (profile) and cannot be claimed again.")
        elif c["type"] == "home_loan_principal":
            if not det.get("lender_certificate_attached") and not c.get("docs"): miss.append("lender principal-repayment certificate")
        elif c["type"] == "home_loan_interest":
            if det.get("occupancy", "self") != "self": elig = 0; why.append("Only self-occupied property interest is computed; let-out property needs manual HR/tax review.")
        elif c["type"] == "education_loan_interest":
            if det.get("principal_included"): elig = max(0.0, amt - n(det.get("principal_amount"))); why.append("Principal is not deductible under 80E; only the interest portion qualifies.")
            ry = det.get("repayment_start_fy")
            if not ry: miss.append("year in which repayment started");
            elif int(fy[:4]) - int(str(ry)[:4]) >= d["80E"]["max_years"]: elig = 0; why.append(f"80E interest is allowed for {d['80E']['max_years']} years from the start of repayment; that window has ended.")
            if det.get("lender_type") not in ("bank", "financial_institution", "approved_charitable"): miss.append("confirmation lender is a bank / financial institution");
        elif c["type"] == "ev_loan_interest":
            sd = str(det.get("sanction_date") or "")
            w0, w1 = d["80EEB"]["loan_sanction_window"]
            if not sd: miss.append("loan sanction date"); elig = 0
            elif not (w0 <= sd <= w1): elig = 0; why.append(f"80EEB applies only to loans sanctioned between {w0} and {w1}; this loan was sanctioned on {sd}.")
            if det.get("vehicle_is_electric") is False: elig = 0; why.append("Vehicle is not an electric vehicle.")
        elif c["type"] == "health_insurance":
            kind = det.get("kind", "premium"); mode_ = str(det.get("payment_mode") or "").lower()
            if kind == "premium" and mode_ in ("cash", ""):
                elig = 0
                if mode_ == "cash": why.append("80D premium paid in cash is not allowed (only preventive health check-up may be paid in cash).")
                else: miss.append("payment mode of the premium (must be non-cash)")
        elif c["type"] == "nps_employee": pass
        # ---- apply caps
        pk, cap, label = {"school_fee": ("80C", c80, "80C"), "invest_80c": ("80C", c80, "80C"), "home_loan_principal": ("80C", c80, "80C"),
                          "home_loan_interest": ("24b", d["24b"]["self_occupied_cap"], "24(b)"), "education_loan_interest": ("80E", d["80E"]["cap"], "80E"),
                          "nps_employee": ("80CCD1B", d["80CCD1B"]["cap"], "80CCD(1B)"), "ev_loan_interest": ("80EEB", d["80EEB"]["cap"], "80EEB")}.get(c["type"], (None, None, None))
        if c["type"] == "health_insurance":
            grp = det.get("group", "self_family"); sen = bool(det.get("beneficiary_senior")); kind = det.get("kind", "premium")
            gcap = d["80D"]["self_family_cap_senior" if sen else "self_family_cap"] if grp == "self_family" else d["80D"]["parents_cap_senior" if sen else "parents_cap"]
            pk = "80D_self" if grp == "self_family" else "80D_par"
            if kind == "preventive_checkup":
                room = d["80D"]["preventive_checkup_cap"] - pools["80D_chk"]; take = max(0.0, min(elig, room, gcap - pools[pk]))
                if take < elig: why.append(f"Preventive check-up is limited to ₹{d['80D']['preventive_checkup_cap']:,} (within the {grp} limit).")
                pools["80D_chk"] += take; pools[pk] += take; elig = take
            else:
                take = max(0.0, min(elig, gcap - pools[pk]))
                if take < elig: why.append(f"80D {grp.replace('_', '/')} limit of ₹{gcap:,} reached.")
                pools[pk] += take; elig = take
        elif pk:
            if cap is None: take = elig
            else: take = max(0.0, min(elig, cap - pools[pk]));
            if cap is not None and take < elig and elig > 0: why.append(f"{label} limit of ₹{cap:,.0f} reached (already used: ₹{pools[pk]:,.0f}).")
            if mode == "approved" and c.get("hr_status") != "approved": pass
            pools[pk] += take if c.get("hr_status") in counted else 0; elig = take
        if c.get("hr_status") not in counted:   # shown with its own explanation but excluded from totals
            r = _res(c, regime, sec, elig, "pending_review" if elig > 0 or miss else "rejected", why + [f"Awaiting HR approval (status: {c.get('hr_status')}); not included in the '{mode}' figure."], miss)
            r["counted"] = False; out[c["id"]] = r; continue
        if miss: why.append("Missing: " + "; ".join(miss) + " - HR cannot verify until provided.")
        if elig > 0 and not why: why.append(f"Eligible under {label or sec}.")
        r = _res(c, regime, sec, elig, None, why, miss); r["counted"] = True; out[c["id"]] = r
    # ---- aggregated: HRA
    hra_info = None
    live = [(c, m) for c, m in rent_claims if c.get("hr_status") in counted]
    for c, m in rent_claims:
        if (c, m) not in live: r = _res(c, regime, "HRA", 0, "pending_review", [f"Awaiting HR approval (status: {c.get('hr_status')}); not included in the '{mode}' figure."], m); r["counted"] = False; out[c["id"]] = r
    if live:
        rent = sum(n(c["amount"]) for c, _ in live); months = len({str((c.get("details") or {})["month"])[:7] for c, _ in live})
        hra_info = hra_exemption(profile, rent, months, rules); pan_need = rent > ex["hra"]["landlord_pan_required_above_annual_rent"]
        if n(profile.get("hra_received")) <= 0: hra_info["exempt"] = 0
        for c, m in live:
            det = c.get("details") or {}; mm = list(m)
            if pan_need and not det.get("landlord_pan"): mm.append(f"landlord PAN (total rent ₹{rent:,.0f} exceeds ₹{ex['hra']['landlord_pan_required_above_annual_rent']:,})")
            share = hra_info["exempt"] * n(c["amount"]) / rent if rent else 0; why = []
            if n(profile.get("hra_received")) <= 0: why.append("Salary has no HRA component, so no HRA exemption (80GG is not modelled).")
            else:
                why.append(f"HRA exemption for {months} month(s) = least of: HRA received ₹{hra_info['actual_hra_received']:,}, rent − 10% basic ₹{hra_info['rent_minus_10pct_basic']:,}, {int(hra_info['pct'] * 100)}% of basic ({'metro' if hra_info['metro'] else 'non-metro'}) ₹{hra_info['pct_of_basic']:,}. Share of this month: ₹{share:,.0f}.")
            if mm: why.append("Missing: " + "; ".join(mm))
            r = _res(c, regime, "HRA", share, None, why, mm); r["counted"] = True
            if mm and any("PAN" in x for x in mm): r["eligible"] = 0.0; r["rejected"] = r["claimed"]; r["status"] = "rejected"; r["reasons"].append("Not allowed until landlord PAN is provided.")
            out[c["id"]] = r
        if any(out[c["id"]]["eligible"] == 0 and "PAN" in " ".join(out[c["id"]]["missing"]) for c, _ in live):   # recompute with only PAN-complete months
            ok = [c for c, _ in live if out[c["id"]]["status"] != "rejected"]
            rent2 = sum(n(c["amount"]) for c in ok); hra_info = hra_exemption(profile, rent2, len(ok), rules) if ok else dict(hra_info, exempt=0)
            for c in ok:
                out[c["id"]]["eligible"] = round(hra_info["exempt"] * n(c["amount"]) / rent2, 2) if rent2 else 0.0; out[c["id"]]["rejected"] = round(out[c["id"]]["claimed"] - out[c["id"]]["eligible"], 2)
    # ---- aggregated: meal vouchers
    meal_total = 0.0; cap_meals = ex["meal_vouchers"]["per_meal"]; provided = n(profile.get("meal_voucher_annual"))
    live_m = [(c, m) for c, m in meal_claims if c.get("hr_status") in counted]
    for c, m in meal_claims:
        if (c, m) not in live_m: r = _res(c, regime, "meal_vouchers", 0, "pending_review", [f"Awaiting HR approval (status: {c.get('hr_status')})."], m); r["counted"] = False; out[c["id"]] = r
    room = provided
    for c, m in live_m:
        det = c.get("details") or {}; meals = n(det.get("meals")); per_cap = cap_meals * meals; why = []
        if provided <= 0: e = 0.0; why.append("No employer meal-voucher amount in the salary profile; the exemption applies only to employer-provided meals.")
        elif not meals: e = 0.0; why.append("Number of meals not stated, so the per-meal cap cannot be applied; nothing allowed.")
        else:
            e = min(n(c["amount"]), per_cap, room); room -= e
            why.append(f"Exempt up to ₹{cap_meals} per meal × {int(meals)} meals = ₹{per_cap:,.0f}.")
            if e < min(n(c["amount"]), per_cap): why.append(f"Limited to the voucher value the employer provides in salary (₹{provided:,.0f}).")
        if m: why.append("Missing: " + "; ".join(m))
        r = _res(c, regime, "meal_vouchers", e, None, why, m); r["counted"] = True; out[c["id"]] = r
    meal_total = round(sum(out[c["id"]]["eligible"] for c, _ in live_m), 2)
    totals = dict(hra=hra_info, hra_exempt=(hra_info or {}).get("exempt", 0), meal_exempt=meal_total, pools=pools)
    return out, totals

# ---------------------------------------------------------------- regime computation
def compute_regime(profile, claims, regime, rules, mode="approved"):
    res, tot = evaluate_claims(profile, claims, regime, rules, mode); r = rules["regimes"][regime]; allowed = set(r["allowed_deductions"]); age = int(n(profile.get("age")) or 30)
    gross = sum(n(profile.get(k)) for k in ("basic", "da", "hra_received", "special_allowance", "bonus", "meal_voucher_annual", "other_taxable", "employer_nps"))
    steps = [("Gross salary", gross)]
    hra = tot["hra_exempt"] if "hra" in allowed else 0; meal = tot["meal_exempt"] if "meal_vouchers" in allowed else 0
    pt = min(n(profile.get("professional_tax")), rules["exemptions"]["professional_tax"]["max"]) if "professional_tax" in allowed else 0
    std = r["standard_deduction"] if "standard_deduction" in allowed else 0
    salary_income = max(0.0, gross - hra - meal - pt - std)
    p = tot["pools"]; hp_loss = p["24b"] if "24b" in allowed else 0
    ccd2_cap = (rules["deductions"]["80CCD2"]["pct_of_basic_da_old" if regime == "old" else "pct_of_basic_da_new"]) * (n(profile.get("basic")) + n(profile.get("da")))
    ccd2 = min(n(profile.get("employer_nps")), ccd2_cap) if "80CCD2" in allowed else 0
    d80d = min(p["80D_self"] + p["80D_par"], rules["deductions"]["80D"]["overall_cap"]) if "80D" in allowed else 0
    ch6 = dict(**{"80C": p["80C"] if "80C" in allowed else 0, "80CCD1B": p["80CCD1B"] if "80CCD1B" in allowed else 0, "80CCD2": ccd2, "80D": d80d,
                  "80E": p["80E"] if "80E" in allowed else 0, "80EEB": p["80EEB"] if "80EEB" in allowed else 0, "other": p["other"] if regime == "old" else 0})
    gti = max(0.0, salary_income - hp_loss + n(profile.get("other_income")))
    ded_total = min(sum(ch6.values()), gti); taxable = max(0.0, gti - ded_total); taxable = int(round(taxable / 10.0)) * 10
    t = tax_with_rebate_surcharge_cess(taxable, regime, rules, age); tds = n(profile.get("tds_deducted_to_date"))
    claimed = sum(v["claimed"] for v in res.values()); elig = sum(v["eligible"] for v in res.values() if v.get("counted", True))
    return dict(regime=regime, mode=mode, fy=rules["financial_year"], gross_salary=round(gross), exemptions=dict(hra=round(hra), meal_vouchers=round(meal), professional_tax=round(pt), standard_deduction=round(std)),
                salary_income=round(salary_income), house_property_loss=round(hp_loss), other_income=round(n(profile.get("other_income"))), gross_total_income=round(gti),
                chapter_vi_a=dict((k, round(v)) for k, v in ch6.items()), total_deductions=round(ded_total), taxable_income=taxable, **t,
                tds_deducted_to_date=round(tds), balance_payable=round(t["total_tax"] - tds), claimed_total=round(claimed), eligible_total=round(elig),
                rejected_total=round(sum(v["rejected"] for v in res.values() if v.get("counted", True))),
                missing_documents=sorted({(v["claim_id"], m) for v in res.values() for m in v["missing"]}), claim_results=res, hra_detail=tot["hra"],
                disclaimer="ESTIMATE ONLY. Not an official payroll/TDS computation. Rules file verification: " + rules["verification"]["status"])

def compare(profile, claims, rules, mode="approved"):
    o, nw = compute_regime(profile, claims, "old", rules, mode), compute_regime(profile, claims, "new", rules, mode)
    better = "new" if nw["total_tax"] <= o["total_tax"] else "old"
    return dict(old=o, new=nw, lower_tax_regime=better, saving=abs(o["total_tax"] - nw["total_tax"]), mode=mode)
