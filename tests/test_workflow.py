"""Declaration -> HR review workflow, permissions, audit trail, suggestion step. Uses a temporary DB (HRBC_OUT is a temp dir)."""
import os, tempfile, unittest, _path
from tax import store as ST, engine as E
from tax.suggest import suggest
R = E.load_rules("2026-27")

class Flow(unittest.TestCase):
    def setUp(self): os.environ["HRBC_OUT"] = tempfile.mkdtemp(prefix="hrbc_wf_")
    def claim(self, **kw):
        a = dict(employee_id="e1", fy="2026-27", ctype="invest_80c", amount=50000, subtype="ppf", docs=["doc1"], doc_hashes=["h1"]); a.update(kw); return ST.add_claim(**a)
    def test_full_path_upload_to_tax_summary(self):
        ST.save_profile("e1", "2026-27", "Asha", dict(basic=800000, hra_received=400000, special_allowance=1000000, city="Pune", age=33), "e1")
        c = self.claim(); self.assertEqual(c["hr_status"], "draft")
        self.assertEqual(ST.submit("e1", "2026-27", "e1"), [c["id"]])
        prof = ST.get_profile("e1", "2026-27")
        proj = E.compute_regime(prof, ST.list_claims("e1", "2026-27"), "old", R, "projected"); appr = E.compute_regime(prof, ST.list_claims("e1", "2026-27"), "old", R, "approved")
        self.assertEqual(proj["chapter_vi_a"]["80C"], 50000); self.assertEqual(appr["chapter_vi_a"]["80C"], 0)          # not counted until HR approves
        ST.hr_decide(c["id"], "approved", "receipt matches", "hr1", "hr")
        appr = E.compute_regime(prof, ST.list_claims("e1", "2026-27"), "old", R, "approved"); self.assertEqual(appr["chapter_vi_a"]["80C"], 50000)
    def test_employee_cannot_approve(self):
        c = self.claim(); ST.submit("e1", "2026-27", "e1")
        with self.assertRaises(ST.PermissionDenied): ST.hr_decide(c["id"], "approved", "x", "e1", "employee")
    def test_employee_cannot_touch_other_employees_claims(self):
        c = self.claim()
        with self.assertRaises(ST.PermissionDenied): ST.update_claim(c["id"], "e2", amount=1)
        with self.assertRaises(ST.PermissionDenied): ST.add_claim("e1", "2026-27", "invest_80c", 5, actor="e2")
        with self.assertRaises(ST.PermissionDenied): ST.submit("e1", "2026-27", "e2")
    def test_hr_cannot_decide_own_claim(self):
        c = self.claim(employee_id="hr1"); ST.submit("hr1", "2026-27", "hr1")
        with self.assertRaises(ST.PermissionDenied): ST.hr_decide(c["id"], "approved", "ok", "hr1", "hr")
    def test_reject_needs_reason_and_draft_cannot_be_decided(self):
        c = self.claim()
        with self.assertRaises(ST.StoreError): ST.hr_decide(c["id"], "approved", "ok", "hr1", "hr")                    # still draft
        ST.submit("e1", "2026-27", "e1")
        with self.assertRaises(ST.StoreError): ST.hr_decide(c["id"], "rejected", "", "hr1", "hr")
        self.assertEqual(ST.hr_decide(c["id"], "rejected", "illegible", "hr1", "hr")["hr_status"], "rejected")
    def test_cannot_approve_without_document_unless_reason(self):
        c = self.claim(docs=[], doc_hashes=[]); ST.submit("e1", "2026-27", "e1")
        with self.assertRaises(ST.StoreError): ST.hr_decide(c["id"], "approved", "", "hr1", "hr")
        with self.assertRaises(ST.StoreError): ST.hr_decide(c["id"], "approved", "", "hr1", "hr", allow_without_docs=True)
        self.assertEqual(ST.hr_decide(c["id"], "approved", "physical copy seen at desk", "hr1", "hr", allow_without_docs=True)["hr_status"], "approved")
    def test_editing_approved_claim_resets_to_draft(self):
        c = self.claim(); ST.submit("e1", "2026-27", "e1"); ST.hr_decide(c["id"], "approved", "ok", "hr1", "hr")
        self.assertEqual(ST.update_claim(c["id"], "e1", amount=90000)["hr_status"], "draft")
    def test_needs_documents_loop(self):
        c = self.claim(); ST.submit("e1", "2026-27", "e1"); ST.hr_decide(c["id"], "needs_documents", "upload premium receipt", "hr1", "hr")
        self.assertEqual(ST.submit("e1", "2026-27", "e1"), [c["id"]]); self.assertEqual(ST.get_claim(c["id"])["hr_status"], "submitted")
    def test_only_drafts_deletable(self):
        c = self.claim(); ST.submit("e1", "2026-27", "e1")
        with self.assertRaises(ST.StoreError): ST.delete_draft(c["id"], "e1")
    def test_audit_trail_records_every_step(self):
        c = self.claim(); ST.submit("e1", "2026-27", "e1"); ST.hr_decide(c["id"], "approved", "ok", "hr1", "hr")
        acts = [a["action"] for a in ST.audit_log(c["id"])]; self.assertEqual(acts[0], "hr_approved"); self.assertIn("add_claim", acts)
    def test_duplicate_claims_stored_but_counted_once(self):
        a, b = self.claim(), self.claim(); ST.submit("e1", "2026-27", "e1"); [ST.hr_decide(x["id"], "approved", "ok", "hr1", "hr") for x in (a, b)]
        o = E.compute_regime(dict(basic=800000, special_allowance=900000, age=30), ST.list_claims("e1", "2026-27"), "old", R); self.assertEqual(o["chapter_vi_a"]["80C"], 50000)
    def test_input_validation(self):
        for amt in (0, -5, "abc"):
            with self.assertRaises(ST.StoreError): self.claim(amount=amt)
        with self.assertRaises(ST.StoreError): self.claim(ctype="crypto_gain")

class Suggest(unittest.TestCase):
    def types(self, text, cat=None): return [s["claim_type"] for s in suggest(text, None, cat)]
    def test_rent_receipt(self): self.assertIn("hra_rent", self.types("RENT RECEIPT Received rent for the month of April from tenant. Landlord PAN ABCDE1234F"))
    def test_sodexo(self): self.assertIn("meal_voucher", self.types("Pluxee meal card statement"))
    def test_school_fee_flags_non_tuition_items(self):
        s = [x for x in suggest("Tuition Fee 30000 Transport Fee 6000 Development Fee 4000 School Fee Receipt", None, "miscellaneous") if x["claim_type"] == "school_fee"][0]
        self.assertTrue(s["non_qualifying_items_seen"]); self.assertIn("note", s); self.assertNotIn("eligible", s["note"].lower().replace("eligibility", ""))
    def test_ordinary_bills_get_no_tax_suggestion(self):
        for t, c in (("HP Petrol Pump fuel 40 litres Rs 4000", "fuel"), ("Hotel Taj room charges GST", "hotel"), ("Dominos Pizza dine in", "food")): self.assertEqual(self.types(t, c), [])
    def test_hospital_bill_is_not_80d_premium(self): self.assertNotIn("health_insurance", self.types("Apollo Hospital bill insurance claim ward charges", "medical"))
    def test_premium_receipt_is_80d_candidate(self): self.assertIn("health_insurance", self.types("Star Health premium receipt policy number 12345 sum insured"))
    def test_every_suggestion_carries_caveats_and_required_docs(self):
        for s in suggest("tuition fee school fee receipt academic year student name"): self.assertTrue(s["caveats"] and s["required_documents"])
if __name__ == "__main__": unittest.main()
