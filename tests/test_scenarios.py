"""The 14 required scenarios (synthetic, anonymised OCR text - no real bills) + decision policy + LLM service + amount stages. Fake LLM, no model needed."""
import os, tempfile, unittest, _path
from llm import enhance as EN, config as C, validate as V
from llm.client import LLMUnavailable, LLMBadOutput
from llm.service import LLMService, validate_schema
from doc_profile import classify, decide
from tax import autofill as AF, store as ST, engine as E
S = V.S
def seg(t, y=0, c=0.95): return dict(text=t, bbox=[10, y, 400, y + 20], conf=c)
def segs_of(*lines, conf=0.95): return [seg(t, i * 25, conf) for i, t in enumerate(lines)]
class Fake:
    def __init__(self, js=None, err=None, errs=None): self.js, self.err, self.errs, self.calls = js, err, list(errs or []), 0; self.s = C.settings({}) | {"mode": "always"}; self.last_stats = {}
    def chat_json(self, system, user, image=None):
        self.calls += 1
        if self.errs: raise self.errs.pop(0)
        if self.err: raise self.err
        return self.js, str(self.js), 0.1
def llm(segs, js, v1=None, policy="suggest", **kw):
    return EN.enhance(segs, (500, 600), v1 or {}, None, client=Fake(js, **kw), cfg=C.settings({}) | {"mode": "always", "cache": False, "llm_only": policy})
TA = lambda l="miscellaneous", c=0.9, r="auto": {"label": l, "confidence": c, "route": r}
F = lambda **kw: {k: {"value": v} for k, v in kw.items()}
GST = next(b + c for b in ["27AAPFU0939F1Z"] for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" if S.gstin_checksum_ok(b + c))
def fl(v): return {"value": v, "evidence": str(v)}

class Scenarios(unittest.TestCase):
    def test_01_typical_gst_invoice(self):
        s = segs_of("ACME STORES PVT LTD", f"GSTIN {GST}", "Invoice No: INV-2041", "Date 12/03/2026", "Sub Total 1,000.00", "CGST 90.00", "SGST 90.00", "Total 1,180.00")
        r = llm(s, {"fields": {"merchant_name": fl("ACME STORES PVT LTD"), "gstin": fl(GST), "invoice_date": fl("12/03/2026"), "subtotal": fl(1000), "cgst": fl(90), "sgst": fl(90), "total_amount": fl(1180), "tax_total": fl(180)}}, policy="add")
        f = r["fields"]; self.assertEqual(f["total_amount"]["value"], 1180.0); self.assertEqual(f["gstin"]["value"], GST); self.assertEqual(f["tax_total"]["value"], 180.0) if "tax_total" in f else None
        self.assertTrue(any("arithmetic ok" in e for e in f["total_amount"]["evidence"]))
        self.assertEqual(classify("\n".join(x["text"] for x in s), {}, TA("miscellaneous"))["document_type"], "expense_bill")
        d = llm(s, {"fields": {"total_amount": fl(1180)}}); self.assertNotIn("total_amount", d["fields"]); self.assertEqual(d["suggestions"]["total_amount"]["value"], 1180.0)     # default policy: reviewable suggestion, not a field
    def test_02_indian_comma_amounts(self):
        s = segs_of("Rent Receipt", "Amount Rs. ₹1,25,000.00 only"); self.assertEqual(S.parse_amount("₹1,25,000.00"), 125000.0)
        r = llm(s, {"fields": {"total_amount": fl("1,25,000.00")}}, policy="add"); self.assertEqual(r["fields"]["total_amount"]["value"], 125000.0)
    def test_03_rent_receipt_hra(self):
        t = "RENT RECEIPT\nReceived from Mr. Ravi Kumar rent for the month of April 2026 Rs 25000\nLandlord Name: Mohan Lal\nPAN ABCDE1234F"
        p = classify(t, F(total_amount=25000.0, invoice_date="2026-04-30"), TA()); self.assertEqual((p["document_type"], p["benefit_category"], p["tax_section_candidates"]), ("rent_receipt", "hra_rent", ["HRA"]))
        self.assertEqual(p["claim_eligibility"], "pending_verification"); self.assertFalse(p["auto_claim_allowed"]); self.assertEqual(p["ext_fields"]["rental_period"]["value"], "April 2026")
    def test_04_school_fee_receipt(self):
        p = classify("School Fee Receipt Tuition Fee 40000 Transport Fee 5000 Student Name: Aarav Sharma School Name: Sunrise Public School receipt no 5 academic year", F(total_amount=45000.0, invoice_date="2026-06-01", invoice_number="5"), TA())
        self.assertEqual((p["document_type"], p["benefit_category"]), ("school_fee_receipt", "education_fee")); self.assertTrue(p["evidence"]["caveats"]["school_fee"]); self.assertEqual(p["ext_fields"]["student_name"]["value"], "Aarav Sharma")
    def test_05_insurance_receipt_80d(self):
        p = classify("Star Health Insurance premium receipt Policy No: SH/2026/778899 sum insured", F(total_amount=18000.0, invoice_date="2026-05-01"), TA()); self.assertEqual((p["document_type"], p["tax_section_candidates"]), ("insurance_premium_receipt", ["80D"]))
        self.assertEqual(p["ext_fields"]["policy_number"]["value"], "SH/2026/778899")
    def test_06_80c_document(self):
        p = classify("PPF account statement national savings certificate tax saver ELSS", F(total_amount=50000.0, invoice_date="2026-06-01"), TA()); self.assertEqual((p["document_type"], p["tax_section_candidates"]), ("investment_proof_80c", ["80C"]))
    def test_07_loan_interest_certificate(self):
        p = classify("Home loan interest certificate housing loan principal repaid 120000 interest paid 180000 Lender: Example Bank", F(invoice_date="2026-03-31"), TA()); self.assertEqual((p["document_type"], p["benefit_category"]), ("loan_interest_certificate", "loan")); self.assertIn("24(b)", p["tax_section_candidates"])
    def test_08_poor_ocr_missing_fields_stay_missing(self):
        s = segs_of("AC#E ST0R", "T0tal ?", conf=0.35); r = llm(s, {"fields": {"total_amount": {"value": None}, "invoice_date": None, "merchant_name": fl("ACME STORES")}}); self.assertEqual(r["fields"], {}) ; self.assertTrue(any(x["field"] == "merchant_name" for x in r["rejected"]))
        self.assertIn("low OCR confidence (mean 0.35)", r["reasons"])
    def test_09_conflicting_totals_flagged(self):
        s = segs_of("Sub Total 1,000.00", "CGST 90.00", "SGST 90.00", "Total 1,500.00"); r = llm(s, {"fields": {"subtotal": fl(1000), "cgst": fl(90), "sgst": fl(90), "total_amount": fl(1500)}}, policy="add")
        self.assertTrue(r["fields"]["total_amount"]["needs_review"]); self.assertTrue(any("MISMATCH" in e for e in r["fields"]["total_amount"]["evidence"]))
    def test_10_unrelated_document(self):
        p = classify("Dear Sir, kindly find enclosed the minutes of the committee meeting held on Monday.", {}, TA("non_expense", 0.9)); self.assertEqual((p["document_type"], p["benefit_category"], p["claim_eligibility"]), ("non_expense_document", "none", "not_applicable"))
    def test_11_malformed_llm_output(self):
        r = llm(segs_of("x"), "```not json```"); self.assertEqual(r["status"], "bad_output"); self.assertIsNone(r["fields"])
        with self.assertRaises(LLMBadOutput): validate_schema({"foo": 1}); 
        with self.assertRaises(LLMBadOutput): validate_schema([1, 2])
    def test_12_timeout_or_unavailable_falls_back(self):
        r = llm(segs_of("x"), None, err=LLMUnavailable("timeout")); self.assertEqual((r["status"], r["fields"]), ("unavailable", None))
    def test_13_low_confidence_classification_needs_review(self):
        p = classify("Pluxee meal card statement", {}, TA("food", 0.3, "review")); self.assertTrue(any("below the routing threshold" in x for x in p["decision"]["reasons"]))
    def test_14_unsupported_tax_claim_not_auto_eligible(self):
        p = classify("ordinary lunch bill", {}, TA("food"), llm_class={"kind": "ev_lease", "evidence": "ordinary lunch", "grounded": True}); self.assertEqual((p["document_type"], p["status"]), ("needs_review", "needs_review")); self.assertFalse(p["auto_claim_allowed"]); self.assertEqual(p["benefit_category"], "none")
        r = E.load_rules("2026-27"); cl = dict(id="1", employee_id="e", fy="2026-27", type="ev_loan_interest", amount=50000, details={"sanction_date": "2025-06-01"}, docs=["d"], doc_hashes=[], hr_status="approved", created_at="1")
        self.assertEqual(E.compute_regime(dict(basic=1000000), [cl], "old", r)["claim_results"]["1"]["eligible"], 0)

class Policy(unittest.TestCase):
    C = lambda self, *c: [dict(claim_type=t, evidence_level=l, matched=["a"] * n) for t, l, n in c]
    def test_agreement(self): self.assertEqual(decide(self.C(("hra_rent", "high", 2)), {"kind": "hra_rent", "grounded": True}, TA())[:2], ("hra_rent", "classified"))
    def test_contradiction_goes_to_review_not_to_either_label(self):
        p, st, why, used = decide(self.C(("hra_rent", "high", 2)), {"kind": "school_fee", "grounded": True}, TA()); self.assertEqual((p, st, used), (None, "needs_review", True)); self.assertTrue(any("contradiction" in w for w in why))
    def test_llm_alone_cannot_create_a_tax_label(self): self.assertEqual(decide([], {"kind": "invest_80c", "grounded": True}, TA())[:2], (None, "needs_review"))
    def test_ungrounded_llm_opinion_ignored(self):
        p, st, why, used = decide(self.C(("hra_rent", "high", 2)), {"kind": "school_fee", "grounded": False}, TA()); self.assertEqual((p, st, used), ("hra_rent", "classified", False)); self.assertTrue(any("ignored" in w for w in why))
    def test_tie_resolved_only_by_grounded_matching_llm(self):
        c = self.C(("school_fee", "medium", 1), ("meal_voucher", "medium", 1)); self.assertEqual(decide(c, None, TA())[:2], (None, "needs_review"))
        self.assertEqual(decide(c, {"kind": "meal_voucher", "grounded": True}, TA())[:2], ("meal_voucher", "classified")); self.assertEqual(decide(c, {"kind": "invest_80c", "grounded": True}, TA())[:2], (None, "needs_review"))
    def test_llm_generic_bill_vs_strong_rule_is_a_contradiction(self): self.assertEqual(decide(self.C(("health_insurance", "high", 2)), {"kind": "expense_bill", "grounded": True}, TA())[:2], (None, "needs_review"))
    def test_no_llm_rules_only(self): self.assertEqual(decide(self.C(("health_insurance", "high", 2)), None, TA())[:2], ("health_insurance", "classified"))
    def test_task_a_category_never_changed(self):
        p = classify("Star Health premium receipt policy number HDF123456 health insurance", {}, TA("medical", 0.8), llm_class={"kind": "health_insurance", "evidence": "premium receipt", "grounded": True}); self.assertEqual(p["expense_category"], "medical")
    def test_enhance_attaches_grounded_flag_for_quoted_evidence(self):
        s = segs_of("Star Health premium receipt", "Policy No SH1234567"); r = llm(s, {"fields": {}, "classification": {"document_kind": "health_insurance", "evidence": "Star Health premium receipt"}}); self.assertTrue(r["llm_classification"]["grounded"])
        r = llm(s, {"fields": {}, "classification": {"document_kind": "health_insurance", "evidence": "totally invented quote here"}}); self.assertFalse(r["llm_classification"]["grounded"])
        r = llm(s, {"fields": {}, "classification": {"document_kind": "made_up_kind", "evidence": "Star Health premium receipt"}}); self.assertIsNone(r["llm_classification"])

class Service(unittest.TestCase):
    def setUp(self): self.d = tempfile.mkdtemp()
    def test_cache_prevents_duplicate_inference(self):
        f = Fake({"fields": {"total_amount": fl(5)}}); sv = LLMService(f, cache_dir=self.d, use_cache=True); a = sv.chat_json("s", "u"); b = sv.chat_json("s", "u")
        self.assertEqual((f.calls, a[0], b[0], sv.last_cached), (1, b[0], a[0], True)); sv.chat_json("s", "different"); self.assertEqual(f.calls, 2)
    def test_cache_never_stores_prompt_text(self):
        sv = LLMService(Fake({"fields": {}}), cache_dir=self.d, use_cache=True); sv.chat_json("SECRETSYS", "SECRETUSER 9999"); txt = "".join(open(os.path.join(self.d, x)).read() for x in os.listdir(self.d)); self.assertNotIn("SECRET", txt)
    def test_retry_only_for_transport_failures_and_bounded(self):
        f = Fake({"fields": {}}, errs=[LLMUnavailable("a")]); self.assertEqual(LLMService(f, use_cache=False, retries=1).chat_json("s", "u")[0], {"fields": {}}); self.assertEqual(f.calls, 2)
        f = Fake(err=LLMUnavailable("down")); 
        with self.assertRaises(LLMUnavailable): LLMService(f, use_cache=False, retries=1).chat_json("s", "u")
        self.assertEqual(f.calls, 2)
        f = Fake({"bad": 1})
        with self.assertRaises(LLMBadOutput): LLMService(f, use_cache=False, retries=3).chat_json("s", "u")
        self.assertEqual(f.calls, 1)            # a deterministic bad answer is not retried
    def test_busy_slot_times_out_cleanly(self):
        from llm import service as SVC
        SVC._SEM.acquire()
        try:
            with self.assertRaises(LLMUnavailable): LLMService(Fake({"fields": {}}), use_cache=False, wait_timeout=0.1).chat_json("s", "u")
        finally: SVC._SEM.release()

class VisionFallback(unittest.TestCase):
    class Flaky:
        def __init__(self): self.calls = []; self.s = C.settings({}) | {"mode": "always"}; self.last_stats = {}
        def chat_json(self, system, user, image=None):
            self.calls.append(image is not None)
            if image is not None: raise LLMBadOutput("empty answer (generation aborted)")
            return {"fields": {"total_amount": fl(10)}}, "{}", 0.1
    def test_aborted_vision_call_retries_once_text_only(self):
        from PIL import Image
        f = self.Flaky(); js, raw, secs = EN.run_llm(f, segs_of("Total 10.00"), (500, 600), Image.new("RGB", (10, 10))); self.assertEqual(f.calls, [True, False]); self.assertTrue(f.last_fallback); self.assertIn("fields", js)
    def test_text_only_failure_is_not_retried(self):
        class Bad(self.Flaky):
            def chat_json(self, *a, **k): self.calls.append(False); raise LLMBadOutput("x")
        f = Bad()
        with self.assertRaises(LLMBadOutput): EN.run_llm(f, segs_of("Total 10.00"), (500, 600), None)
        self.assertEqual(len(f.calls), 1)

class ZeroVsMissing(unittest.TestCase):
    def test_zero_only_when_labelled(self):
        o = V.OCR(segs_of("CGST 0.00", "SGST 0.00", "Total 0.00", "Discount 0.00"))
        for f, ok in (("cgst", True), ("sgst", True), ("discount", True), ("igst", False), ("total_amount", False), ("subtotal", False)): self.assertEqual(V.ground(f, 0, o)["status"] == "grounded", ok, f)
    def test_missing_is_none_not_zero(self): self.assertEqual(llm(segs_of("Total 10.00"), {"fields": {"cgst": {"value": None}}})["fields"], {})

class AmountStages(unittest.TestCase):
    R = E.load_rules("2026-27")
    def claim(self, amt, doc, st="approved"): return dict(id="1", employee_id="e", fy="2026-27", type="invest_80c", subtype="ppf", amount=amt, details={"document_amount": doc}, docs=["d"], doc_hashes=[], hr_status=st, created_at="1")
    def test_four_amounts_kept_apart(self):
        r = E.compute_regime(dict(basic=1000000), [self.claim(50000, 30000)], "old", self.R)["claim_results"]["1"]
        self.assertEqual((r["claimed"], r["document_amount"], r["eligible"], r["approved_amount"]), (50000, 30000, 30000, 30000)); self.assertTrue(any("exceeds" in x for x in r["reasons"]))
    def test_not_approved_has_no_approved_amount(self): self.assertIsNone(E.compute_regime(dict(basic=1000000), [self.claim(10000, 10000, "submitted")], "old", self.R, "projected")["claim_results"]["1"]["approved_amount"])
if __name__ == "__main__": unittest.main()
