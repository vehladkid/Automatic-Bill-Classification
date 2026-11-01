"""Taxonomy/profile classification, per-field report, autofill provenance and the tax rules' FY / authorisation behaviour."""
import os, tempfile, unittest, _path
from doc_profile import classify
import field_report as FR
from tax import autofill as AF, store as ST, engine as E
TA = lambda l="miscellaneous", c=0.8, r="auto": {"label": l, "confidence": c, "route": r}
F = lambda **kw: {k: {"value": v} for k, v in kw.items()}
RENT = "RENT RECEIPT\nReceived from Mr. Ravi Kumar rent for the month of April 2026 Rs 25000\nLandlord Name: Mohan Lal\nPAN ABCDE1234F"

class Profile(unittest.TestCase):
    def test_existing_expense_bills_unchanged(self):
        for txt, lab in (("HP Petrol Pump fuel 40 litres", "fuel"), ("Taj hotel room charges", "hotel"), ("Dominos pizza", "food"), ("Airtel postpaid bill", "internet")):
            p = classify(txt, {}, TA(lab, 0.9)); self.assertEqual((p["document_type"], p["benefit_category"], p["expense_category"], p["tax_section_candidates"]), ("expense_bill", "none", lab, []))
    def test_rent_receipt(self):
        p = classify(RENT, F(total_amount=25000.0, invoice_date="2026-04-30"), TA()); self.assertEqual((p["document_type"], p["benefit_category"], p["primary_claim_type"]), ("rent_receipt", "hra_rent", "hra_rent"))
        self.assertEqual(p["validity"]["financial_year"], "2026-27"); self.assertEqual(p["tax_section_candidates"], ["HRA"]); self.assertFalse(p["classification_confidence"]["calibrated"])
    def test_classification_is_not_eligibility(self):
        p = classify(RENT, F(total_amount=25000.0), TA()); self.assertNotIn("eligible", str(p["validity"]).lower().replace("eligibility", "")); self.assertIn("not tax eligibility", p["validity"]["note"])
    def test_each_new_category(self):
        cases = {"sodexo": ("Pluxee meal card statement Sodexo meal voucher", "meal_coupon_statement", "meal_benefit"), "school": ("School Fee Receipt Tuition Fee 40000 student name: Aarav academic year", "school_fee_receipt", "education_fee"),
                 "80c": ("PPF account statement tax saver ELSS national savings certificate", "investment_proof_80c", "investment_80c"), "80d": ("Star Health premium receipt policy number HDF123456 health insurance", "insurance_premium_receipt", "health_insurance"),
                 "loan": ("Home loan interest certificate housing loan principal repaid", "loan_interest_certificate", "loan"), "ev": ("EV lease agreement electric vehicle lease rental", "ev_purchase_or_lease_doc", "ev_benefit")}
        for k, (t, dt, bc) in cases.items():
            p = classify(t, {}, TA()); self.assertEqual((p["document_type"], p["benefit_category"]), (dt, bc), k)
    def test_ambiguous_document_is_needs_review_not_forced(self):
        p = classify("tuition fee paid and also pluxee statement", {}, TA()); self.assertEqual((p["status"], p["document_type"]), ("needs_review", "needs_review")); self.assertIsNone(p["primary_claim_type"]); self.assertGreaterEqual(len(p["candidates"]), 2)
    def test_imbalanced_evidence_picks_primary_keeps_secondary(self):
        p = classify("tuition fee receipt and also pluxee meal card sodexo", {}, TA()); self.assertEqual(p["primary_claim_type"], "meal_voucher"); self.assertIn("school_fee", p["secondary_claim_types"])
    def test_weak_evidence_is_needs_review(self):
        p = classify("monthly statement tenant landlord", {}, TA()); self.assertIn(p["status"], ("needs_review",)); self.assertEqual(p["tax_section_candidates"], [])
    def test_secondary_labels_for_multi_purpose_document(self):
        p = classify("Home loan interest certificate housing loan; education loan interest certificate student loan", {}, TA()); self.assertTrue(p["primary_claim_type"]); self.assertTrue(p["secondary_claim_types"] or p["status"] == "needs_review")
    def test_unknown_when_no_task_a(self): self.assertEqual(classify("lorem ipsum", {}, None)["document_type"], "unknown")
    def test_new_labels_marked_not_trained(self):
        p = classify(RENT, F(total_amount=1.0), TA()); self.assertIn("NOT a trained", p["model_support"]["document_type_and_benefit"]); self.assertIn("rent", p["model_support"]["data_needed"].lower())
    def test_incomplete_document_lists_missing_fields(self):
        p = classify("school fee receipt tuition fee", F(total_amount=1000.0), TA()); self.assertEqual(p["validity"]["completeness"], "incomplete"); self.assertIn("student_name", p["validity"]["missing_fields"])
    def test_missing_values_not_hallucinated_by_lite_extractor(self):
        self.assertEqual(classify("rent receipt landlord tenant", {}, TA())["ext_fields"], {})
    def test_two_pans_not_guessed_as_landlord(self):
        p = classify("rent receipt landlord tenant PAN ABCDE1234F and tenant PAN XYZAB9876C", {}, TA()); self.assertNotIn("landlord_pan", p["ext_fields"])

class FieldReport(unittest.TestCase):
    SEG = [dict(text="Total 100", bbox=[0, 0, 100, 20], conf=0.95), dict(text="smudge", bbox=[0, 40, 100, 60], conf=0.3)]
    def test_four_things_kept_apart(self):
        r = FR.build({"total_amount": dict(value=100.0, confidence=0.9, source="layoutlm", bbox=[0, 0, 100, 20])}, self.SEG)["total_amount"]
        self.assertEqual(r["ocr_confidence"], 0.95); self.assertEqual(r["extraction"]["calibrated_confidence"], 0.9); self.assertEqual(r["validation"]["status"], "passed"); self.assertEqual(r["presence"], "extracted")
    def test_llm_score_is_not_presented_as_calibrated(self):
        r = FR.build({"total_amount": dict(value=100.0, confidence=None, source="llm(validated)", origin="llm", level="medium", evidence_score=0.5, needs_review=True, bbox=None)}, self.SEG)["total_amount"]
        self.assertIsNone(r["extraction"]["calibrated_confidence"]); self.assertFalse(r["extraction"]["evidence_score_is_probability"]); self.assertEqual(r["presence"], "extracted_unverified")
    def test_illegible_absent_conflict_distinguished(self):
        r = FR.build({"invoice_date": dict(value="2026-01-01", confidence=0.9, source="x", bbox=[0, 40, 100, 60]), "total_amount": dict(value=5.0, confidence=0.9, source="x", bbox=None)}, self.SEG, expected=["gstin"], conflicts=[{"field": "total_amount"}])
        self.assertEqual((r["invoice_date"]["presence"], r["gstin"]["presence"], r["total_amount"]["presence"]), ("illegible", "absent", "conflict")); self.assertIsNone(r["gstin"]["value"])
    def test_arithmetic_and_format_checks(self):
        r = FR.build({"subtotal": dict(value=100.0, source="x"), "tax_total": dict(value=18.0, source="x"), "total_amount": dict(value=500.0, source="x"), "invoice_date": dict(value="2999-01-01", source="x")}, [])
        self.assertEqual(r["total_amount"]["validation"]["status"], "failed"); self.assertIn("arithmetic_consistent", r["total_amount"]["validation"]["failed"]); self.assertEqual(r["invoice_date"]["validation"]["status"], "failed")

class Autofill(unittest.TestCase):
    def setUp(self): os.environ["HRBC_OUT"] = tempfile.mkdtemp(prefix="hrbc_af_"); self.rec = lambda **k: dict(doc_id="D1", owner="E1", profile=classify(RENT, F(total_amount=25000.0, invoice_date="2026-04-30"), TA()), **k)
    FIELDS = F(total_amount=25000.0, invoice_date="2026-04-30")
    def profile(self, **kw):
        ST.save_profile("E1", "2026-27", "Ravi Kumar", dict(basic=800000, hra_received=300000, city="Pune", chosen_regime="old", **kw), "E1"); return ST.get_profile("E1", "2026-27")
    def test_every_value_has_provenance_and_kind(self):
        af = AF.autofill(self.rec(), self.FIELDS, "E1", profile=self.profile())
        self.assertEqual(af["claim_type"], "hra_rent"); self.assertEqual(af["fy"], "2026-27"); self.assertEqual(af["details"]["month"], "2026-04")
        kinds = {v["field"]: v["kind"] for v in af["values"]}; self.assertEqual((kinds["amount"], kinds["financial_year"], kinds["month"], kinds["profile.city"]), ("extracted", "calculated", "calculated", "employee_declared"))
        for v in af["values"]: self.assertIn("source", v); self.assertTrue(v["source"])
        self.assertEqual(af["status"], "ready_for_review")
    def test_verified_profile_marked_payroll_verified_and_edit_drops_verification(self):
        self.profile(); ST.verify_profile("E1", "2026-27", "HR1", "hr")
        af = AF.autofill(self.rec(), self.FIELDS, "E1"); self.assertEqual({v["kind"] for v in af["values"] if v["field"].startswith("profile.")}, {"payroll_verified"})
        self.profile(); self.assertNotIn("_verified_by", ST.get_profile("E1", "2026-27"))
    def test_only_hr_other_than_owner_can_verify(self):
        self.profile()
        with self.assertRaises(ST.PermissionDenied): ST.verify_profile("E1", "2026-27", "E1", "employee")
        with self.assertRaises(ST.PermissionDenied): ST.verify_profile("E1", "2026-27", "E1", "hr")
    def test_missing_amount_never_invented(self):
        rec = dict(doc_id="D1", owner="E1", profile=classify(RENT, {}, TA())); af = AF.autofill(rec, {}, "E1", profile=self.profile())
        self.assertIsNone(af["amount"]); self.assertEqual(af["status"], "needs_review"); self.assertTrue(any("amount" in m for m in af["missing"]))
        with self.assertRaises(ST.StoreError): AF.create_claim(af, "E1", "E1")
    def test_missing_date_means_no_financial_year_no_claim(self):
        rec = dict(doc_id="D1", owner="E1", profile=classify(RENT, F(total_amount=1.0), TA())); af = AF.autofill(rec, F(total_amount=25000.0), "E1")
        self.assertIsNone(af["fy"]); self.assertEqual(af["status"], "needs_review")
        with self.assertRaises(ST.StoreError): AF.create_claim(af, "E1", "E1")
    def test_fy_without_configured_rules_flagged_not_guessed(self):
        af = AF.autofill(self.rec(), F(total_amount=25000.0, invoice_date="2031-05-01"), "E1"); self.assertIsNone(af["fy"]); self.assertTrue(any("no configured tax rules" in n for n in af["notes"]))
    def test_no_regime_chosen_is_flagged(self):
        ST.save_profile("E1", "2026-27", "Ravi Kumar", dict(basic=1, hra_received=1, chosen_regime="undecided"), "E1"); af = AF.autofill(self.rec(), self.FIELDS, "E1")
        self.assertIsNone(af["regime"]); self.assertTrue(any("regime" in m for m in af["missing"]))
    def test_no_profile_flagged(self): self.assertTrue(any("salary profile" in m for m in AF.autofill(self.rec(), self.FIELDS, "E1")["missing"]))
    def test_tenant_name_mismatch_is_conflict(self):
        ST.save_profile("E1", "2026-27", "Someone Else Entirely", dict(basic=1, hra_received=1, chosen_regime="old"), "E1")
        af = AF.autofill(self.rec(), self.FIELDS, "E1"); self.assertTrue(any(c["field"] == "tenant_name" for c in af["conflicts"])); self.assertEqual(af["status"], "needs_review")
    def test_profile_without_hra_component_conflicts(self):
        ST.save_profile("E1", "2026-27", "Ravi Kumar", dict(basic=800000, hra_received=0, chosen_regime="old"), "E1"); self.assertTrue(any(c["field"] == "hra_received" for c in AF.autofill(self.rec(), self.FIELDS, "E1")["conflicts"]))
    def test_duplicate_month_and_amount_conflict_detected(self):
        self.profile(); ST.add_claim("E1", "2026-27", "hra_rent", 20000, {"month": "2026-04", "landlord_name": "X"}, docs=["Dz"], doc_hashes=["hh"])
        af = AF.autofill(self.rec(), self.FIELDS, "E1", existing_claims=ST.list_claims("E1", "2026-27")); self.assertTrue(af["duplicates"]); self.assertTrue(any(c["field"] == "amount" for c in af["conflicts"]))
    def test_same_document_hash_is_duplicate(self):
        self.profile(); ST.add_claim("E1", "2026-27", "hra_rent", 25000, {"month": "2026-03", "landlord_name": "X"}, docs=["D0"], doc_hashes=["abc"])
        self.assertTrue(AF.autofill(self.rec(), self.FIELDS, "E1", existing_claims=ST.list_claims("E1", "2026-27"), doc_hashes=["abc"])["duplicates"])
    def test_other_employees_document_denied(self):
        rec = dict(self.rec(), owner="E9"); af = AF.autofill(rec, self.FIELDS, "E1", profile=self.profile()); self.assertEqual(af["status"], "denied"); self.assertEqual(af["values"], [])
    def test_multi_month_receipt_not_auto_split(self):
        t = RENT.replace("April 2026", "April 2026 to June 2026"); rec = dict(doc_id="D1", owner="E1", profile=classify(t, self.FIELDS, TA())); af = AF.autofill(rec, self.FIELDS, "E1", profile=self.profile())
        self.assertNotIn("month", af["details"]); self.assertTrue(any("one claim per month" in n for n in af["notes"]))
    def test_school_fee_does_not_prefill_tuition_or_deductibility(self):
        t = "School Fee Receipt Tuition Fee 40000 Transport Fee 5000 Student Name: Aarav Sharma School Name: Sunrise Public School"; rec = dict(doc_id="D2", owner="E1", profile=classify(t, F(total_amount=45000.0, invoice_date="2026-06-01", invoice_number="R1"), TA()))
        af = AF.autofill(rec, F(total_amount=45000.0, invoice_date="2026-06-01"), "E1", profile=self.profile()); self.assertNotIn("tuition_component", af["details"]); self.assertTrue(any("tuition" in m for m in af["missing"]))
        cl = AF.create_claim(af, "E1", "E1"); self.assertEqual(E.compute_regime(ST.get_profile("E1", "2026-27"), [dict(cl, hr_status="approved")], "old", E.load_rules("2026-27"))["claim_results"][cl["id"]]["eligible"], 0)   # unevaluable without breakup
    def test_claim_creation_keeps_provenance_and_correction_history(self):
        self.profile(); af = AF.autofill(self.rec(), self.FIELDS, "E1"); cl = AF.create_claim(af, "E1", "E1", edits={"amount": 24000.0}, doc_hashes=["h1"])
        self.assertEqual(cl["amount"], 24000.0); self.assertEqual(cl["hr_status"], "draft"); self.assertTrue(cl["source"]["autofill"]["values"]); self.assertEqual(cl["docs"], ["D1"])
        ST.update_claim(cl["id"], "E1", amount=25000.0); h = ST.get_claim(cl["id"])["source"]["corrections"]; self.assertEqual((h[0]["old"]["amount"], h[0]["new"]["amount"], h[0]["actor"]), (24000.0, 25000.0, "E1"))
    def test_other_employee_cannot_create_claim_for_me(self):
        af = AF.autofill(self.rec(), self.FIELDS, "E1", profile=self.profile())
        with self.assertRaises(ST.PermissionDenied): AF.create_claim(af, "E1", "E2")
if __name__ == "__main__": unittest.main()
