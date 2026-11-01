"""Structured document profile: keeps document type, expense category, benefit category, candidate tax sections, validity/completeness, fields and evidence SEPARATE.
Classification is not tax eligibility: 'tax_section_candidates' lists sections the document could be evidence for; eligibility is decided only by app/tax/engine.py + HR.
Existing Task A prediction is passed through untouched (backward compatible). New labels are rules_only (see config/taxonomy.json) - not trained model classes."""
import json
from pathlib import Path
from tax.suggest import suggest
import doc_fields as DF
CFG = json.loads((Path(__file__).resolve().parents[1] / "config" / "taxonomy.json").read_text())
LEVEL_RANK = {"high": 3, "medium": 2, "low": 1}
VERSION = "profile-v2-policy"
GENERIC = ("expense_bill", "non_expense_document", "other", "unknown")
LEVELS = LEVEL_RANK

def _have(profile_key, fields, ext):
    """map taxonomy required-field names to what we have (V1/enhanced fields + extension fields)"""
    f = lambda k: (fields.get(k) or {}).get("value"); e = lambda k: (ext.get(k) or {}).get("value")
    has = {"amount": f("total_amount") or e("rent_amount") or e("premium_amount"), "doc_number": f("invoice_number"), "doc_date": f("invoice_date"),
           "provider": f("merchant_name") or e("provider"), "institution": e("institution") or f("merchant_name"), "insurer": e("insurer") or f("merchant_name"),
           "lender": e("lender") or f("merchant_name"), "landlord_name": e("landlord_name"), "tenant_name": e("tenant_name") or f("employee_name"), "student_name": e("student_name"),
           "policy_number": e("policy_number"), "interest_amount": e("interest_amount"), "loan_type": e("loan_type"), "period": e("period") or e("rental_period"),
           "rental_period_or_date": e("rental_period") or f("invoice_date"), "sanction_date": e("sanction_date")}
    return has.get

def decide(cands, llm_class, task_a):
    """EXPLICIT decision policy (versioned). Returns (primary_claim_type | None, status, reasons[], llm_used).
    Rules: (1) the LLM opinion counts only if the OCR words it quotes as evidence really are in the document; (2) the LLM can confirm or resolve a tie, but never creates
    a tax-document label on its own and never overrides rule evidence (a disagreement => needs_review); (3) weak/tied rule evidence => needs_review unless a grounded LLM
    opinion matches exactly one tied candidate; (4) the trained Task A expense category is never changed here; low Task A confidence only adds a review reason."""
    reasons = []; lk = None
    if llm_class:
        if llm_class.get("grounded"): lk = llm_class["kind"]
        else: reasons.append("LLM classification ignored: its quoted evidence was not found in the OCR text")
    top = cands[0] if cands else None
    strong = [c for c in cands if c["evidence_level"] in ("high", "medium")]
    tie = len(cands) > 1 and LEVEL_RANK[cands[1]["evidence_level"]] == LEVEL_RANK[top["evidence_level"]] and len(cands[1]["matched"]) >= len(top["matched"]) and CFG["profiles"][top["claim_type"]]["document_type"] != CFG["profiles"][cands[1]["claim_type"]]["document_type"]
    primary, status, used = None, "classified", False
    if not cands:
        if lk in CFG["profiles"]: status = "needs_review"; reasons.append(f"only the LLM suggests '{lk}'; no rule evidence - an LLM alone cannot label a tax document"); used = True
    elif top["evidence_level"] == "low" or tie:
        status = "needs_review"; reasons.append("weak or tied rule evidence")
        tied = [c["claim_type"] for c in cands if LEVEL_RANK[c["evidence_level"]] == LEVEL_RANK[top["evidence_level"]]]
        if lk in tied and tied.count(lk) == 1 and top["evidence_level"] != "low": primary, status, used = lk, "classified", True; reasons.append(f"tie resolved by grounded LLM opinion '{lk}'")
    else:
        primary = top["claim_type"]
        if lk is not None and lk != primary:
            status, primary, used = "needs_review", None, True
            reasons.append(f"LLM says '{lk}' but rules say '{top['claim_type']}' - contradiction")
        elif lk == primary: used = True; reasons.append("rules and LLM agree")
    ta = task_a or {}
    if ta.get("route") not in (None, "auto"): reasons.append("expense-category confidence below the routing threshold")
    return primary, status, reasons, used

def classify(ocr_text, fields=None, task_a=None, ext_fields=None, llm_class=None):
    """fields: Task B (or enhanced) dict {name:{'value'}}; task_a: {'label','confidence','route'}; ext_fields: LLM-validated extension fields (optional).
    Returns the profile dict (JSON-serialisable)."""
    fields = fields or {}; ext = dict(DF.extract_lite(ocr_text)); ext.update(ext_fields or {})
    sg = suggest(ocr_text, fields, (task_a or {}).get("label"))
    cands = [s for s in sg if CFG["profiles"].get(s["claim_type"])]
    cands.sort(key=lambda s: (-LEVEL_RANK[s["evidence_level"]], -len(s["matched"])))
    strong = [s for s in cands if s["evidence_level"] in ("high", "medium")]
    primary, status, reasons, llm_used = decide(cands, llm_class, task_a); secondary = []
    if not cands:
        doc_type = "needs_review" if status == "needs_review" else ("expense_bill" if task_a and task_a.get("label") not in (None, "non_expense") else ("non_expense_document" if task_a and task_a.get("label") == "non_expense" else "unknown"))
        prof = dict(document_type=doc_type, benefit_category="none", tax_section_candidates=[])
    else:
        top = cands[0]; secondary = [s_["claim_type"] for s_ in cands if s_["claim_type"] != primary and s_["evidence_level"] in ("high", "medium")]
        base = CFG["profiles"][(primary or top["claim_type"])]
        prof = dict(document_type="needs_review" if primary is None else base["document_type"], benefit_category="none" if primary is None else base["benefit_category"], tax_section_candidates=[] if primary is None else base["tax_section_candidates"])
    # completeness / validity (only for a confidently classified benefit document)
    miss, ok = [], None
    if primary:
        have = _have(primary, fields, ext)
        miss = [r for r in CFG["profiles"][primary]["required_fields"] if not have(r)]
        ok = "complete" if not miss else "incomplete"
    prof.update(
        profile_version=VERSION, status=status, primary_claim_type=primary, secondary_claim_types=secondary, candidates=[dict(claim_type=s["claim_type"], evidence_level=s["evidence_level"], matched=s["matched"]) for s in cands],
        expense_category=(task_a or {}).get("label"), expense_category_confidence=(task_a or {}).get("confidence"), expense_category_route=(task_a or {}).get("route"),
        validity=dict(completeness=ok, missing_fields=miss, financial_year=DF.financial_year_of((fields.get("invoice_date") or {}).get("value")),
                      note="Completeness is about document fields, not tax eligibility." if primary else "No employee-benefit document type detected."),
        model_support=dict(expense_category="trained (Task A, calibrated)", document_type_and_benefit="rules_only - NOT a trained classifier", calibrated=False,
                           data_needed=(CFG["profiles"][primary]["data_needed"] if primary else None)),
        classification_confidence=dict(kind="rule_evidence_level", level=(cands[0]["evidence_level"] if cands else "none"), calibrated=False),
        ext_fields=ext, evidence=dict(matched={s["claim_type"]: s["matched"] for s in cands}, caveats={s["claim_type"]: s["caveats"] for s in cands}),
        outcome=("needs_review" if status == "needs_review" else ("unknown" if prof["document_type"] == "unknown" else "classified")),
        decision=dict(policy=VERSION, llm_opinion=(llm_class or {}).get("kind"), llm_opinion_grounded=(llm_class or {}).get("grounded"), llm_influenced_outcome=llm_used, reasons=reasons),
        claim_eligibility=("pending_verification" if primary else "not_applicable"), auto_claim_allowed=False)
    return prof
