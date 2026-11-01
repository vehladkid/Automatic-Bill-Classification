"""Tax calculation, eligibility, duplicate, missing-document and regime tests. Run: .venv/bin/python -m unittest discover -s tests -v"""
import unittest, _path
from tax import engine as E

R = E.load_rules("2026-27"); R25 = E.load_rules("2025-26")
def claim(i, t, amt, status="approved", docs=("d",), hashes=(), details=None, subtype=None, created=None, fy="2026-27"):
    return dict(id=str(i), employee_id="e1", fy=fy, type=t, subtype=subtype, amount=amt, details=details or {}, docs=list(docs), doc_hashes=list(hashes), hr_status=status, created_at=created or f"{int(i):04d}")
P = dict(employee_id="e1", basic=800000, da=0, hra_received=400000, special_allowance=1000000, city="Bengaluru", age=35, employee_pf=96000, professional_tax=2400)

class Slabs(unittest.TestCase):
    def t(self, inc, reg, rules=R, age=30): return E.tax_with_rebate_surcharge_cess(inc, reg, rules, age)
    def test_new_15L(self): self.assertEqual(self.t(1500000, "new")["total_tax"], 109200)           # 105000 + 4% cess
    def test_old_10L(self): self.assertEqual(self.t(1000000, "old")["total_tax"], 117000)           # 112500 + 4%
    def test_new_rebate_zero_tax_up_to_12L(self): self.assertEqual(self.t(1200000, "new")["total_tax"], 0)
    def test_new_marginal_relief_just_above_12L(self):
        r = self.t(1210000, "new"); self.assertEqual(r["total_tax"], 10400)                         # tax limited to the 10,000 excess + cess
    def test_old_rebate_5L(self): self.assertEqual(self.t(500000, "old")["total_tax"], 0)
    def test_old_just_above_5L_no_rebate(self): self.assertGreater(self.t(510000, "old")["total_tax"], 0)
    def test_surcharge_marginal_relief(self):
        r = self.t(5010000, "new"); self.assertEqual(r["surcharge"], 7000)
    def test_senior_old_slab(self): self.assertEqual(self.t(300000, "old", age=65)["total_tax"], 0); self.assertGreater(self.t(300000, "old", age=30)["total_tax"] + 1, 0)
    def test_missing_fy_rules_fail_loudly(self):
        with self.assertRaises(E.RulesError): E.load_rules("2031-32")

class HRA(unittest.TestCase):
    def rent(self, rules, n=12): return [claim(i + 1, "hra_rent", 40000, details=dict(month=f"{2026 if i < 9 else 2027}-{(i + 3) % 12 + 1:02d}", landlord_name="L", landlord_pan="ABCDE1234F")) for i in range(n)]
    def test_metro_vs_nonmetro_rules_by_year(self):
        c = self.rent(R); o = E.compute_regime(P, c, "old", R); self.assertEqual(o["exemptions"]["hra"], 400000)          # Bengaluru metro in FY26-27 rules
        c25 = [dict(x, fy="2025-26", details=dict(x["details"], month=x["details"]["month"].replace("2026", "2025").replace("2027", "2026"))) for x in c]
        o25 = E.compute_regime(P, c25, "old", R25); self.assertEqual(o25["exemptions"]["hra"], 320000)                  # non-metro 40% in FY25-26 rules
    def test_not_allowed_in_new_regime(self):
        n = E.compute_regime(P, self.rent(R), "new", R); self.assertEqual(n["exemptions"]["hra"], 0)
        self.assertIn("not available under the new", n["claim_results"]["1"]["reasons"][0])
    def test_landlord_pan_required_over_1L(self):
        c = self.rent(R); [x["details"].pop("landlord_pan") for x in c]; o = E.compute_regime(P, c, "old", R)
        self.assertEqual(o["exemptions"]["hra"], 0); self.assertTrue(any("PAN" in m for _, m in o["missing_documents"]))
    def test_small_rent_needs_no_pan(self):
        c = [claim(1, "hra_rent", 8000, details=dict(month="2026-04", landlord_name="L"))]; o = E.compute_regime(P, c, "old", R); self.assertGreater(o["exemptions"]["hra"], 0)
    def test_no_hra_component_no_exemption(self):
        o = E.compute_regime(dict(P, hra_received=0), self.rent(R), "old", R); self.assertEqual(o["exemptions"]["hra"], 0)
    def test_month_outside_fy_rejected(self):
        o = E.compute_regime(P, [claim(1, "hra_rent", 40000, details=dict(month="2025-01", landlord_name="L"))], "old", R); self.assertEqual(o["claim_results"]["1"]["status"], "rejected")

class Deductions(unittest.TestCase):
    def old(self, claims, p=P, mode="approved"): return E.compute_regime(p, claims, "old", R, mode)
    def test_80c_cap_and_payroll_epf_first(self):
        o = self.old([claim(1, "invest_80c", 100000, subtype="ppf")]); self.assertEqual(o["chapter_vi_a"]["80C"], 150000)    # 96,000 EPF + 54,000 of 100,000
        self.assertEqual(o["claim_results"]["1"]["status"], "limited"); self.assertIn("limit", o["claim_results"]["1"]["reasons"][0])
    def test_epf_cannot_be_claimed_twice(self):
        o = self.old([claim(1, "invest_80c", 96000, subtype="epf_employee")]); self.assertEqual(o["claim_results"]["1"]["eligible"], 0)
    def test_school_fee_only_tuition_component(self):
        o = self.old([claim(1, "school_fee", 60000, details=dict(child_name="A", tuition_component=40000))], dict(P, employee_pf=0)); r = o["claim_results"]["1"]
        self.assertEqual((r["eligible"], r["rejected"], r["status"]), (40000, 20000, "limited"))
    def test_school_fee_without_breakup_not_allowed(self):
        r = self.old([claim(1, "school_fee", 60000, details=dict(child_name="A"))])["claim_results"]["1"]; self.assertEqual(r["eligible"], 0); self.assertTrue(r["missing"])
    def test_school_fee_third_child_rejected(self):
        cl = [claim(i, "school_fee", 10000, details=dict(child_name=n, tuition_component=10000)) for i, n in ((1, "A"), (2, "B"), (3, "C"))]
        o = self.old(cl, dict(P, employee_pf=0)); self.assertEqual(o["claim_results"]["3"]["eligible"], 0); self.assertEqual(o["claim_results"]["2"]["eligible"], 10000)
    def test_school_fee_not_deductible_in_new_regime(self):
        n = E.compute_regime(P, [claim(1, "school_fee", 40000, details=dict(child_name="A", tuition_component=40000))], "new", R); self.assertEqual(n["claim_results"]["1"]["eligible"], 0)
    def test_80d_cash_premium_rejected(self):
        r = self.old([claim(1, "health_insurance", 20000, details=dict(group="self_family", payment_mode="cash"))])["claim_results"]["1"]; self.assertEqual(r["eligible"], 0)
    def test_80d_caps_self_and_senior_parents(self):
        cl = [claim(1, "health_insurance", 40000, details=dict(group="self_family", payment_mode="upi")), claim(2, "health_insurance", 60000, details=dict(group="parents", beneficiary_senior=True, payment_mode="netbanking"))]
        o = self.old(cl); self.assertEqual((o["claim_results"]["1"]["eligible"], o["claim_results"]["2"]["eligible"]), (25000, 50000)); self.assertEqual(o["chapter_vi_a"]["80D"], 75000)
    def test_80d_preventive_checkup_limit(self):
        r = self.old([claim(1, "health_insurance", 9000, details=dict(group="self_family", kind="preventive_checkup", payment_mode="cash"))])["claim_results"]["1"]; self.assertEqual(r["eligible"], 5000)
    def test_80e_interest_only_and_window(self):
        ok = self.old([claim(1, "education_loan_interest", 30000, details=dict(repayment_start_fy="2024-25", lender_type="bank"))])["claim_results"]["1"]; self.assertEqual(ok["eligible"], 30000)
        old = self.old([claim(1, "education_loan_interest", 30000, details=dict(repayment_start_fy="2016-17", lender_type="bank"))])["claim_results"]["1"]; self.assertEqual(old["eligible"], 0)
        pr = self.old([claim(1, "education_loan_interest", 50000, details=dict(principal_included=True, principal_amount=20000, repayment_start_fy="2024-25", lender_type="bank"))])["claim_results"]["1"]; self.assertEqual(pr["eligible"], 30000)
    def test_home_loan_interest_cap_and_principal_in_80c(self):
        o = self.old([claim(1, "home_loan_interest", 260000, details=dict(occupancy="self")), claim(2, "home_loan_principal", 100000)], dict(P, employee_pf=0))
        self.assertEqual(o["house_property_loss"], 200000); self.assertEqual(o["chapter_vi_a"]["80C"], 100000)
    def test_let_out_property_not_auto_computed(self):
        self.assertEqual(self.old([claim(1, "home_loan_interest", 90000, details=dict(occupancy="let_out"))])["claim_results"]["1"]["eligible"], 0)
    def test_80eeb_only_for_loans_in_window(self):
        a = self.old([claim(1, "ev_loan_interest", 40000, details=dict(sanction_date="2022-06-01"))])["claim_results"]["1"]["eligible"]
        b = self.old([claim(1, "ev_loan_interest", 40000, details=dict(sanction_date="2025-06-01"))])["claim_results"]["1"]
        self.assertEqual(a, 40000); self.assertEqual(b["eligible"], 0); self.assertIn("sanctioned between", b["reasons"][0])
    def test_ev_lease_never_auto_deductible(self):
        for t in ("ev_lease", "ev_reimbursement"):
            r = self.old([claim(1, t, 500000)])["claim_results"]["1"]; self.assertEqual((r["eligible"], r["status"]), (0, "pending_review"))
    def test_other_claim_zero_until_hr_override_with_reason(self):
        self.assertEqual(self.old([claim(1, "other", 10000)])["claim_results"]["1"]["eligible"], 0)
        self.assertEqual(self.old([claim(1, "other", 10000, details=dict(hr_override_amount=7000, hr_override_reason="s.80GGB donation, receipt verified"))])["claim_results"]["1"]["eligible"], 7000)
    def test_meal_voucher_per_meal_cap(self):
        p = dict(P, meal_voucher_annual=26400)
        r = self.old([claim(1, "meal_voucher", 26400, details=dict(meals=100))], p)["claim_results"]["1"]; self.assertEqual(r["eligible"], 20000)        # 100 meals x Rs200
        n = E.compute_regime(p, [claim(1, "meal_voucher", 26400, details=dict(meals=132))], "new", R); self.assertEqual(n["exemptions"]["meal_vouchers"], 26400)  # allowed in both regimes
        self.assertEqual(self.old([claim(1, "meal_voucher", 5000, details=dict(meals=0))], p)["claim_results"]["1"]["eligible"], 0)
    def test_meal_cap_uses_fy_specific_rule(self):
        p = dict(P, meal_voucher_annual=10000, fy="2025-26"); r = E.compute_regime(p, [claim(1, "meal_voucher", 10000, details=dict(meals=100), fy="2025-26")], "old", R25)
        self.assertEqual(r["claim_results"]["1"]["eligible"], 5000)                                                                                       # Rs50/meal in FY25-26
    def test_80ccd2_employer_nps_both_regimes(self):
        p = dict(P, employer_nps=100000); self.assertEqual(self.old([], p)["chapter_vi_a"]["80CCD2"], 80000)       # 10% of basic in old regime
        self.assertEqual(E.compute_regime(p, [], "new", R)["chapter_vi_a"]["80CCD2"], 100000)                      # 14% = 112,000 cap, 100,000 declared

class Duplicates(unittest.TestCase):
    def test_same_document_hash_counts_once(self):
        cl = [claim(1, "invest_80c", 50000, subtype="ppf", hashes=["h1"]), claim(2, "invest_80c", 50000, subtype="ppf", hashes=["h1"])]
        o = E.compute_regime(dict(P, employee_pf=0), cl, "old", R); self.assertEqual(o["chapter_vi_a"]["80C"], 50000); self.assertIn("Duplicate", o["claim_results"]["2"]["reasons"][0])
    def test_same_bill_identity_counts_once(self):
        d = dict(vendor="Sun School", doc_number="R-77", doc_date="2026-06-01")
        cl = [claim(1, "school_fee", 30000, details=dict(d, child_name="A", tuition_component=30000)), claim(2, "school_fee", 30000, details=dict(d, child_name="A", tuition_component=30000))]
        self.assertEqual(E.compute_regime(dict(P, employee_pf=0), cl, "old", R)["chapter_vi_a"]["80C"], 30000)
    def test_same_rent_month_twice(self):
        d = dict(month="2026-05", landlord_name="L", landlord_pan="ABCDE1234F")
        o = E.compute_regime(P, [claim(1, "hra_rent", 30000, details=d), claim(2, "hra_rent", 30000, details=d)], "old", R); self.assertEqual(o["claim_results"]["2"]["status"], "rejected")
    def test_different_documents_not_flagged(self):
        cl = [claim(1, "invest_80c", 50000, subtype="ppf", hashes=["h1"]), claim(2, "invest_80c", 50000, subtype="elss", hashes=["h2"])]
        self.assertEqual(E.find_duplicates(cl), {})

class MissingAndStatus(unittest.TestCase):
    def test_missing_document_is_reported(self):
        o = E.compute_regime(P, [claim(1, "invest_80c", 20000, subtype="ppf", docs=())], "old", R); self.assertTrue(o["claim_results"]["1"]["missing"]); self.assertTrue(o["missing_documents"])
    def test_pending_claims_excluded_from_approved_but_in_projected(self):
        cl = [claim(1, "invest_80c", 50000, status="submitted", subtype="ppf")]; p = dict(P, employee_pf=0)
        self.assertEqual(E.compute_regime(p, cl, "old", R, "approved")["chapter_vi_a"]["80C"], 0)
        self.assertEqual(E.compute_regime(p, cl, "old", R, "projected")["chapter_vi_a"]["80C"], 50000)
    def test_rejected_and_draft_never_counted(self):
        p = dict(P, employee_pf=0)
        for st in ("rejected", "draft", "needs_documents"): self.assertEqual(E.compute_regime(p, [claim(1, "invest_80c", 50000, status=st, subtype="ppf")], "old", R, "projected")["chapter_vi_a"]["80C"], 0)
    def test_new_regime_rejects_old_only_sections(self):
        n = E.compute_regime(P, [claim(1, "invest_80c", 50000, subtype="ppf"), claim(2, "health_insurance", 10000, details=dict(payment_mode="upi"))], "new", R)
        self.assertEqual(n["total_deductions"], 0); self.assertIn("not available under the new", n["claim_results"]["1"]["reasons"][0])
    def test_unknown_80c_instrument_rejected(self):
        self.assertEqual(E.compute_regime(P, [claim(1, "invest_80c", 5000, subtype="bitcoin")], "old", R)["claim_results"]["1"]["eligible"], 0)
    def test_compare_picks_lower_regime_and_is_labelled_estimate(self):
        c = E.compare(P, [claim(1, "invest_80c", 54000, subtype="ppf")], R); self.assertIn(c["lower_tax_regime"], ("old", "new")); self.assertIn("ESTIMATE", c["old"]["disclaimer"]); self.assertEqual(c["saving"], abs(c["old"]["total_tax"] - c["new"]["total_tax"]))
    def test_no_double_count_across_totals(self):
        p = dict(P, employee_pf=150000); o = E.compute_regime(p, [claim(1, "invest_80c", 50000, subtype="ppf"), claim(2, "school_fee", 40000, details=dict(child_name="A", tuition_component=40000))], "old", R)
        self.assertEqual(o["chapter_vi_a"]["80C"], 150000)
if __name__ == "__main__": unittest.main()
