"""Collection-manifest schema/validator tests and approval-gating tests for approved-sources-v2 against the licence register and the v2.1 manifest."""
import sys, json, copy, unittest
from pathlib import Path
import _path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts" / "dataset_discovery"))
import collection_manifest as C
import doc_type_lib as L
def good(**kw):
    r = {"doc_id": "COL-3f9a1c0d-7be2", "doc_type": "rent_receipt", "expense_category": "rent", "tax_evidence": ["hra_evidence"], "real": True, "source_id": "COLL-ORG-01", "permission_basis": "individual_written_consent", "permission_ref": "CR-0042", "issuer_group_id": "IG-9c41ab07e3",
         "split": "unassigned", "annotation_status": "pending", "sensitive_review_status": "pending", "redacted_copy": None, "ambiguity_flags": [], "reviewer_notes": "", "eligibility_status": "pending_verification", "country": "IN", "language": "en", "created": "2026-10-19"}
    r.update(kw); return r
def ready(**kw):
    base = dict(split="train", sensitive_review_status="redacted_verified", redacted_copy="rent_receipt/COL-3f9a1c0d-7be2.png", annotation_status="single_annotated"); base.update(kw); return good(**base)
class Schema(unittest.TestCase):
    def test_valid_unassigned_and_ready_records_pass(self): self.assertEqual(C.validate([good(), ready(doc_id="COL-00000000-0001", redacted_copy="a.png")]), {})
    def test_missing_and_unknown_fields_fail(self):
        r = good(); r.pop("permission_ref"); self.assertIn("missing_fields", C.validate([r])); self.assertIn("unknown_fields", C.validate([good(holder_name="x")]))
    def test_enums_and_id_formats_fail(self):
        for kw, key in (({"doc_type": "passport"}, "bad_doc_type"), ({"expense_category": "salary"}, "bad_expense_category"), ({"tax_evidence": ["tax_free"]}, "bad_tax_evidence"), ({"doc_id": "doc1"}, "bad_doc_id"), ({"issuer_group_id": "HDFC"}, "bad_issuer_group_id"),
                        ({"permission_basis": "assumed"}, "bad_permission_basis"), ({"eligibility_status": "eligible"}, "eligibility_status_invalid"), ({"real": False}, "synthetic_or_non_real_in_collection"), ({"country": "India"}, "bad_country"), ({"permission_ref": " "}, "missing_permission_ref")):
            self.assertIn(key, C.validate([good(**kw)]), key)
    def test_three_axes_are_independent_fields(self):
        r = good(doc_type="salary_slip", expense_category=None, tax_evidence=["income_evidence"]); self.assertEqual(C.validate([r]), {})
        self.assertFalse(set(C.SCHEMA["document_type_enum"]) & set(C.SCHEMA["expense_category_enum"]))
    def test_duplicate_ids_and_redacted_paths_fail(self):
        a, b = ready(redacted_copy="x.png"), ready(redacted_copy="x.png"); e = C.validate([a, b]); self.assertIn("duplicate_doc_id", e); self.assertIn("duplicate_redacted_copy", e)
    def test_redacted_copy_must_be_relative_and_verified(self):
        self.assertIn("redacted_copy_must_be_relative", C.validate([ready(redacted_copy="/home/u/x.png")])); self.assertIn("redacted_copy_without_verified_status", C.validate([good(redacted_copy="x.png")]))
class Gating(unittest.TestCase):
    def test_split_requires_verified_redaction_and_copy(self):
        e = C.validate([good(split="train")]); self.assertIn("gate_split_without_verified_redaction", e); self.assertIn("gate_split_without_redacted_copy", e)
    def test_eval_split_requires_double_annotation(self):
        r = ready(); r["split"] = "test"; self.assertIn("gate_eval_split_without_double_annotation", C.validate([r])); r["annotation_status"] = "double_annotated"; self.assertEqual(C.validate([r]), {})
    def test_specimen_cannot_be_in_eval_split(self):
        r = ready(); r.update(split="val", annotation_status="adjudicated", ambiguity_flags=["specimen_or_template"]); self.assertIn("gate_specimen_in_eval_split", C.validate([r]))
        r["split"] = "train"; self.assertEqual(C.validate([r]), {})
    def test_issuer_group_cannot_span_splits(self):
        a = ready(); b = ready(doc_id="COL-00000000-0002", redacted_copy="b.png"); b.update(split="val", annotation_status="double_annotated"); self.assertIn("issuer_group_spans_splits", C.validate([a, b]))
    def test_personal_data_patterns_are_rejected_everywhere(self):
        for note, key in (("PAN ABCDE1234F", "PAN"), ("aadhaar 2345 6789 0123", "AADHAAR"), ("call 9876543210", "PHONE"), ("mail a.b@example.com", "EMAIL"), ("acct 123456789012", "LONG_NUMBER"), ("branch HDFC0001234", "IFSC"), ("gst 27ABCDE1234F1Z5", "GSTIN")):
            e = C.validate([good(reviewer_notes=note)]); self.assertIn("personal_data_pattern_" + key, e, note)
        self.assertIn("personal_data_pattern_PAN", C.validate([good(source_id="ABCDE1234F")]))
    def test_clean_note_passes(self): self.assertEqual(C.validate([good(reviewer_notes="handwritten, revenue stamp present", ambiguity_flags=["low_quality_scan"])]), {})
    def test_unassigned_and_rejected_records_never_count_toward_coverage(self):
        rows = [good(), good(doc_id="COL-00000000-0003", sensitive_review_status="rejected_unredactable"), ready(redacted_copy="z.png")]; cov = {c["doc_type"]: c for c in C.coverage(rows)}; self.assertEqual(cov["rent_receipt"]["train"], 1); self.assertFalse(cov["rent_receipt"]["ready_for_per_class_metric"])
    def test_empty_repo_manifest_passes_and_has_no_documents(self):
        p = ROOT / "reports/doc_type/licensing_and_collection/collection_manifest.jsonl"; self.assertTrue(p.exists()); self.assertEqual([l for l in open(p) if l.strip()], [])
class ApprovalConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT / "config/approved_sources_v2.json").read_text()); cls.reg = {s["id"]: s for s in json.loads((L.META / "license_register_v2.json").read_text())["sources"]}
        cls.rows = L.read_jsonl(L.META / "doc_type_manifest_v2_1.jsonl")
    def test_every_approved_id_is_usable_or_restricted_in_the_register(self):
        for i in self.cfg["tier1_usable"]: self.assertEqual(self.reg[i]["status"], "usable", i)
        for i in self.cfg["tier2_restricted_known_terms"]: self.assertEqual(self.reg[i]["status"], "usable_with_restrictions", i)
    def test_excluded_sources_are_never_approved_and_not_usable_in_register(self):
        ex = set(self.cfg["excluded_until_cleared"]); self.assertTrue({"in_financial", "sroie", "hf_medical", "user_real", "receiptbench"} <= ex)
        for i in ex: self.assertNotIn(self.reg[i]["status"], ("usable", "usable_with_restrictions"), i)
        for r in self.rows:
            if r["source"] in ex: self.assertFalse(r["approved_for_training"], r["doc_id"])
        self.assertFalse(set(self.cfg["tier1_usable"]) & ex); self.assertFalse(set(self.cfg["tier2_restricted_known_terms"]) & ex)
    def test_approved_rows_match_config_exactly(self):
        ok = set(self.cfg["tier1_usable"]) | set(self.cfg["tier2_restricted_known_terms"])
        for r in self.rows: self.assertEqual(r["approved_for_training"], r["license_register_id"] in ok, r["doc_id"])
    def test_in_financial_stays_blocked_with_primary_evidence(self):
        e = self.reg["in_financial"]; self.assertEqual(e["status"], "unverified"); self.assertEqual(e["evidence"]["level"], "primary_verified"); self.assertIn("kaggle.com/datasets/mehaksingal", e["url"]); self.assertEqual(sum(r["approved_for_training"] for r in self.rows if r["source"] == "in_financial"), 0)
    def test_no_tier_permits_commercial_use_or_redistribution(self): self.assertIn("No tier permits redistribution", self.cfg["rule"])
    def test_decision_table_covers_every_excluded_source(self):
        d = {x["source"]: x for x in json.loads((ROOT / "reports/doc_type/licensing_and_collection/licensing_decisions.json").read_text())["decisions"]}
        for i in self.cfg["excluded_until_cleared"]: self.assertTrue(d[i]["decision"].startswith(("BLOCKED", "REQUIRES")), i); self.assertTrue(d[i]["exact_permission_needed"])
if __name__ == "__main__": unittest.main()
