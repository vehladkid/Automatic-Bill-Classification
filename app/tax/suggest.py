"""Suggest which tax-claim type (if any) a processed bill might support. Evidence-based and conservative: a suggestion lists matched phrases, caveats and the
documents HR will still need. It NEVER states that an amount is deductible. A bill with no evidence yields no suggestion (e.g. a fuel or restaurant bill)."""
import json, re
from pathlib import Path
CFG = Path(__file__).resolve().parents[2] / "config" / "tax_suggest.json"
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
REQUIRED_DOCS = {"hra_rent": ["rent receipt", "landlord PAN if annual rent > ₹1,00,000", "rent agreement (recommended)"], "meal_voucher": ["voucher/meal-card statement showing meals"],
                 "school_fee": ["fee receipt with tuition-fee breakup", "child's name"], "health_insurance": ["premium receipt", "proof of non-cash payment"],
                 "invest_80c": ["investment/premium receipt or statement"], "home_loan_interest": ["lender's provisional/annual interest certificate"],
                 "education_loan_interest": ["lender's interest certificate"], "ev_loan_interest": ["loan sanction letter + interest certificate"], "ev_lease": ["lease agreement + employer letter"]}

def _hits(text, words):
    """whole-word / whole-phrase match (so 'ppf' does not fire inside another word); text is lower-cased with single spaces"""
    return [w for w in words if re.search(r"(?<![a-z0-9])" + re.escape(w.strip()) + r"(?![a-z0-9])", text)]

def suggest(ocr_text, fields=None, category=None, cfg=None):
    """ocr_text: str; fields: Task B / enhanced fields dict {name: {'value':..}}; category: Task A label. Returns a list sorted by evidence (may be empty)."""
    cfg = cfg or json.loads(CFG.read_text())["candidates"]; t = " " + re.sub(r"\s+", " ", str(ocr_text or "").casefold()) + " "; fields = fields or {}
    val = lambda k: (fields.get(k) or {}).get("value")
    out = []
    for ctype, c in cfg.items():
        s, w, neg = _hits(t, c.get("strong", [])), _hits(t, c.get("weak", [])), _hits(t, c.get("negative", []))
        if not s and len(w) < 2: continue
        level = "high" if len(s) >= 2 else "medium" if s else "low"
        caveats = list(c.get("caveats", []))
        if neg: caveats.insert(0, "Bill also mentions non-tuition items (" + ", ".join(neg) + "): only the tuition-fee portion can qualify.")
        if ctype == "health_insurance" and category == "medical" and not _hits(t, ["premium", "policy"]): continue     # hospital/pharmacy bill, not a premium receipt
        pans = sorted(set(PAN_RE.findall(str(ocr_text or "").upper()))) if ctype == "hra_rent" else []
        out.append(dict(claim_type=ctype, evidence_level=level, matched=s + [x for x in w if x not in s], amount=val("total_amount"), doc_date=val("invoice_date"), vendor=val("merchant_name"),
                        doc_number=val("invoice_number"), pan_candidates=pans, caveats=caveats, required_documents=REQUIRED_DOCS.get(ctype, []), non_qualifying_items_seen=neg,
                        note="Suggestion only. Not an eligibility decision; the employee must confirm and HR must verify."))
    return sorted(out, key=lambda x: ({"high": 0, "medium": 1, "low": 2}[x["evidence_level"]], -len(x["matched"])))
