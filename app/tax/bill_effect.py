"""Per-bill income-tax effect (ESTIMATE ONLY). For a processed bill: which income-tax section (if any) it can support, the eligible amount and the change in the employee's yearly tax
(tax without this bill minus tax with it), computed by app/tax/engine.py from config/tax_rules (status SECONDARY_SOURCES_ONLY: not verified against the official gazette).
Honest scope: GST is an indirect tax charged on the bill; an employee cannot claim it back as an income-tax rebate. Ordinary expense bills (food, travel, hotel, fuel ...) give NO income-tax deduction
to the employee (reimbursement is employer policy). Only the tax-document types in config/taxonomy.json (rent -> HRA, school fee / investments -> 80C, health insurance -> 80D, loan interest, ...) can reduce tax.
If the employee has no saved salary profile, a SYNTHETIC DEMO profile is used and every output is labelled synthetic: it shows how the calculation works, not anyone's real saving."""
from . import engine as E, store as ST, autofill as AF
SYNTHETIC_PROFILE = dict(basic=720000, da=0, hra_received=288000, special_allowance=192000, city="metro", age=30, chosen_regime="old", employee_pf=0, professional_tax=2400,
                         note="SYNTHETIC DEMO PROFILE (assumed gross salary 12,00,000, metro, old regime) - not real employee data")
NO_DEDUCTION = "This expense type gives the employee no income-tax deduction. Reimbursement or taxability is decided by employer policy/payroll, and GST on the bill is not claimable by an individual as an income-tax rebate."

def gst_components(fields):
    return {k: fields[k]["value"] for k in ("cgst", "sgst", "igst", "tax_total") if fields.get(k) and isinstance(fields[k].get("value"), (int, float))}

def estimate(rec, fields, employee_id=None, extra_details=None):
    pf = rec.get("profile") or {}; ct = pf.get("primary_claim_type"); gst = gst_components(fields or {}); cat = rec.get("category")
    out = dict(estimate_only=True, expense_category=cat, claim_type=ct, gst_on_bill=gst, gst_note="Shown for information only; GST is not an income-tax rebate.", rules_status=None, synthetic_profile=False)
    if not ct:
        out.update(kind="no_income_tax_deduction", income_tax_saving=0, reason=NO_DEDUCTION); return out
    owner = employee_id or rec.get("owner")
    if not owner: out.update(kind="cannot_estimate", income_tax_saving=None, reason="The document has no owner (employee), so no tax profile applies."); return out
    af = AF.autofill(rec, fields or {}, owner); out["autofill_status"] = af["status"]
    if not af.get("amount") or not af.get("fy"):
        out.update(kind="cannot_estimate", income_tax_saving=None, reason="Amount or financial year could not be read from the document; nothing was guessed.", missing=af["missing"], notes=af["notes"]); return out
    fy = af["fy"]; rules = E.load_rules(fy); out["rules_status"] = rules["verification"]["status"]; out["fy"] = fy
    prof = ST.get_profile(owner, fy)
    if prof: others = [c for c in ST.list_claims(owner, fy) if c.get("hr_status") in ("approved", "submitted") or c.get("status") in ("approved", "submitted")]
    else: prof, others, out["synthetic_profile"] = dict(SYNTHETIC_PROFILE), [], True
    regime = prof.get("chosen_regime") if prof.get("chosen_regime") in ("old", "new") else "old"; out["regime"] = regime
    det = dict(extra_details or {}, **dict(af["details"], document_amount=af["amount"])); out["what_if_assumptions"] = []
    if ct == "health_insurance" and not det.get("payment_mode"):         # a receipt rarely shows how it was paid; the saving is shown as a WHAT-IF and says so
        det["payment_mode"] = "online"; out["what_if_assumptions"].append("premium assumed paid by a non-cash mode (80D allows only non-cash premiums); beneficiary assumed self/family")
    this = dict(id="this-bill", type=ct, fy=fy, amount=af["amount"], docs=[rec["doc_id"]], details=det, hr_status="approved", created_at="9999")
    base = E.compute_regime(prof, others, regime, rules, mode="projected"); withb = E.compute_regime(prof, others + [this], regime, rules, mode="projected")
    r = withb["claim_results"].get("this-bill", {})
    out.update(kind="estimate", amount_on_bill=af["amount"], section=r.get("section"), eligible_amount=r.get("eligible"), not_eligible_amount=r.get("rejected"), status=r.get("status"), reasons=r.get("reasons", []),
               missing=sorted(set(af["missing"]) | set(r.get("missing", []))), tax_without=base["total_tax"], tax_with=withb["total_tax"], income_tax_saving=base["total_tax"] - withb["total_tax"],
               assumption=("SYNTHETIC DEMO PROFILE: " + SYNTHETIC_PROFILE["note"]) if out["synthetic_profile"] else "Employee's saved salary profile" + ("" if prof.get("_verified_by") else " (not verified by HR)"))
    return out
