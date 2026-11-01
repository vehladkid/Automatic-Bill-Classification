"""Quota per bill label, default employee profiles (persist across restarts, never overwrite edits), per-bill rebate."""
import os, sys, json, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
_TMP = tempfile.mkdtemp(); os.environ["HRBC_OUT"] = _TMP
from tax import quota as Q, store as ST, engine as E
TASK_A = ["bus", "electricity", "flight", "food", "fuel", "gas", "hotel", "internet", "medical", "miscellaneous", "non_expense", "parking", "rent", "train", "transport", "water"]
NEW = ["rent_receipt", "rental_agreement", "insurance_premium_receipt", "school_fee_receipt", "loan_interest_certificate", "loan_statement", "investment_certificate", "meal_voucher_statement", "ev_document", "form_16", "salary_slip", "bank_statement"]

class T(unittest.TestCase):
    def test_every_label_has_a_quota(self):
        for l in TASK_A + NEW: self.assertIn("quota_text", Q.label_quota(l), l)
        self.assertEqual(json.loads(Q.CFG.read_text())["labels"].keys() & set(TASK_A + NEW), set(TASK_A + NEW))
    def test_caps_come_from_rules(self):
        self.assertEqual(Q.label_quota("school_fee_receipt")["cap"], 150000); self.assertEqual(Q.label_quota("meal_voucher_statement")["cap"], 200)
        self.assertIsNone(Q.label_quota("food")["claim_type"]); self.assertIsNone(Q.label_quota("rent_receipt")["cap"])
    def test_two_default_profiles_persist_and_are_not_overwritten(self):
        os.environ["HRBC_OUT"] = tempfile.mkdtemp()                                        # fresh store = first ever start
        made = Q.seed_default_profiles(); self.assertEqual({m[0] for m in made}, {"E1", "E2"})
        self.assertEqual(Q.seed_default_profiles(), [])                                   # second start: nothing to do
        ST.save_profile("E1", "2026-27", "Edited", dict(ST.get_profile("E1", "2026-27"), city="Mumbai"), actor="E1"); Q.seed_default_profiles()
        self.assertEqual(ST.get_profile("E1", "2026-27")["city"], "Mumbai")                # employee edit survives the next start
        self.assertEqual(ST.get_profile("E2", "2026-27")["city"], "Jaipur")
    def test_expense_label_gives_no_rebate(self):
        r = Q.bill_rebate({"doc_id": "x", "category": "food", "owner": "E1"}, {}, "E1")
        if Q.demo_quota("food") is None: self.assertEqual(r["effect"]["kind"], "no_income_tax_deduction"); self.assertEqual(r["effect"]["income_tax_saving"], 0)     # statutory behaviour
        else: self.assertIn(r["effect"]["kind"], ("demo_estimate", "cannot_estimate")); self.assertTrue(r["effect"].get("demo_allowance"))                         # temporary demo switch is on
    def test_demo_rebate_uses_real_slabs_and_caps_at_quota(self):
        if Q.demo_quota("food") is None: self.skipTest("demo rebate switched off")
        Q.seed_default_profiles()
        rec = {"doc_id": "x", "category": "food", "owner": "E1"}
        f = {"total_amount": {"value": 50000.0}, "invoice_date": {"value": "2026-06-10"}}
        e = Q.bill_rebate(rec, f, "E1")["effect"]; self.assertEqual(e["kind"], "demo_estimate"); self.assertEqual(e["eligible_amount"], Q.demo_quota("food")); self.assertGreater(e["income_tax_saving"], 0)
        e2 = Q.bill_rebate(rec, {"total_amount": {"value": 50000.0}, "invoice_date": {"value": "2010-06-10"}}, "E1")["effect"]; self.assertEqual(e2["fy"], E.available_years()[-1])   # out-of-range FY -> latest FY, noted
    def test_second_stage_label_maps_to_section_without_rule_profile(self):
        Q.seed_default_profiles()
        rec = {"doc_id": "x", "category": "insurance_premium_receipt", "owner": "E1", "profile": {}}
        f = {"total_amount": {"value": 20000.0}, "invoice_date": {"value": "2026-06-10"}}
        e = Q.bill_rebate(rec, f, "E1")["effect"]; self.assertEqual(e["kind"], "estimate"); self.assertEqual(e["section"], "80D"); self.assertGreater(e["income_tax_saving"], 0)
    def test_literal_fallback_reads_only_printed_values(self):
        f, notes, extra = Q.literal_fallback("Received Rs.69,250.00 by Cash\nDate 10-01-2026\nTuition Fee 69,250.00", {})
        self.assertEqual(f["total_amount"]["value"], 69250.0); self.assertEqual(f["invoice_date"]["value"], "2026-01-10"); self.assertEqual(extra["tuition_component"], 69250.0)
        f2, _, ex2 = Q.literal_fallback("nothing useful here", {}); self.assertNotIn("total_amount", f2); self.assertNotIn("invoice_date", f2)
if __name__ == "__main__": unittest.main()
