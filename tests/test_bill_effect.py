"""Per-bill income-tax effect estimate: honest scope (no deduction for ordinary bills), synthetic-profile labelling, regime rules, no guessing."""
import unittest, tempfile, os, _path
os.environ["HRBC_OUT"] = tempfile.mkdtemp(prefix="hrbc_be_")
from tax import bill_effect as BE, store as ST

def rec(ct=None, owner="E1", cat="food"):
    return dict(doc_id="20261001T000000_aaaaaaaa", owner=owner, category=cat, profile=dict(primary_claim_type=ct, ext_fields={}))
def f(**kw): return {k: {"value": v} for k, v in kw.items()}

class T(unittest.TestCase):
    def test_ordinary_bill_gives_no_deduction_and_gst_is_information_only(self):
        r = BE.estimate(rec(None), f(total_amount=1180.0, cgst=90.0, sgst=90.0)); self.assertEqual(r["kind"], "no_income_tax_deduction"); self.assertEqual(r["income_tax_saving"], 0)
        self.assertEqual(r["gst_on_bill"], {"cgst": 90.0, "sgst": 90.0}); self.assertIn("not an income-tax rebate", r["gst_note"]); self.assertIn("GST", r["reason"])
    def test_health_insurance_old_regime_synthetic_profile_labelled(self):
        r = BE.estimate(rec("health_insurance", "E7"), f(premium_amount=20000.0, total_amount=20000.0, invoice_date="2026-06-10"))
        self.assertEqual(r["kind"], "estimate"); self.assertTrue(r["synthetic_profile"]); self.assertIn("SYNTHETIC", r["assumption"]); self.assertEqual(r["section"], "80D")
        self.assertGreater(r["eligible_amount"], 0); self.assertLessEqual(r["eligible_amount"], 20000.0); self.assertGreater(r["income_tax_saving"], 0); self.assertEqual(r["tax_without"] - r["tax_with"], r["income_tax_saving"])
        self.assertTrue(r["what_if_assumptions"]); self.assertEqual(r["rules_status"], "SECONDARY_SOURCES_ONLY"); self.assertTrue(r["estimate_only"])
    def test_saved_profile_under_new_regime_gives_no_80d_saving(self):
        ST.save_profile("E8", "2026-27", "Test", dict(basic=720000, da=0, hra_received=288000, special_allowance=192000, city="metro", chosen_regime="new", employee_pf=0, professional_tax=0), "E8")
        r = BE.estimate(rec("health_insurance", "E8"), f(premium_amount=20000.0, invoice_date="2026-06-10")); self.assertEqual(r["kind"], "estimate"); self.assertFalse(r["synthetic_profile"])
        self.assertEqual(r["income_tax_saving"], 0); self.assertEqual(r["eligible_amount"], 0); self.assertTrue(any("not available under the new" in x for x in r["reasons"]))
    def test_missing_amount_or_date_is_not_guessed(self):
        r = BE.estimate(rec("health_insurance", "E9"), f(invoice_date="2026-06-10")); self.assertEqual(r["kind"], "cannot_estimate"); self.assertIsNone(r["income_tax_saving"])
        r = BE.estimate(rec("health_insurance", "E9"), f(premium_amount=5000.0)); self.assertEqual(r["kind"], "cannot_estimate"); self.assertIsNone(r["income_tax_saving"])
    def test_unowned_document_not_estimated(self):
        r = BE.estimate(rec("health_insurance", None), f(premium_amount=5000.0, invoice_date="2026-06-10")); self.assertEqual(r["kind"], "cannot_estimate")
    def test_financial_year_without_configured_rules_is_not_guessed(self):
        r = BE.estimate(rec("health_insurance", "E9"), f(premium_amount=5000.0, invoice_date="2031-06-10")); self.assertEqual(r["kind"], "cannot_estimate")
if __name__ == "__main__": unittest.main()
