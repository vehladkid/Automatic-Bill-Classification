"""Dataset tooling tests: generator determinism and label consistency, anti-brand deny-list, grouped split rules, and validator mutation tests (the validator must FAIL on deliberately corrupted manifests)."""
import os, sys, json, tempfile, hashlib, random, subprocess, unittest, collections
from pathlib import Path
import _path
ROOT = Path(__file__).resolve().parents[1]; DD = ROOT / "scripts" / "dataset_discovery"; sys.path.insert(0, str(DD))
import generate_synthetic_docs as G
import doc_type_lib as L
import validate_doc_type_manifest as V
import schema as S

def gen(out, typ, i, seed=7): return G.generate_one(seed, typ, i, Path(out))
class Generator(unittest.TestCase):
    def test_deterministic_same_seed_same_record_and_pixels(self):
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        for typ in ("rent_receipt", "form_16", "school_fee_receipt"):
            r1, r2 = gen(a, typ, 3), gen(b, typ, 3); self.assertEqual(r1["fields"], r2["fields"]); self.assertEqual(r1["lines"], r2["lines"])
            h = lambda d, r: [hashlib.sha256((Path(d) / p).read_bytes()).hexdigest() for p in r["pages"]]; self.assertEqual(h(a, r1), h(b, r2))
    def test_different_seed_or_index_differs(self):
        a = tempfile.mkdtemp(); self.assertNotEqual(gen(a, "rent_receipt", 1)["fields"], gen(a, "rent_receipt", 2)["fields"]); self.assertNotEqual(gen(a, "rent_receipt", 1, 7)["fields"], gen(a, "rent_receipt", 1, 8)["fields"])
    def test_every_type_generates_labelled_synthetic_docs_with_footer(self):
        d = tempfile.mkdtemp()
        for typ in G.BUILDERS:
            r = gen(d, typ, 0); self.assertEqual(r["doc_type"], typ); self.assertTrue(r["synthetic"]); self.assertEqual(r["country"], "IN"); self.assertEqual(r["generator_version"], G.VERSION)
            self.assertTrue(all((Path(d) / p).exists() for p in r["pages"])); self.assertTrue(any("SYNTHETIC SAMPLE" in l["text"] for l in r["lines"]), typ); self.assertTrue(r["fields"]); self.assertTrue(r["key_phrases"])
            txt = " ".join(l["text"] for l in r["lines"]).lower(); self.assertTrue(any(k.lower() in txt for k in r["key_phrases"]), f"{typ}: class-defining phrase not rendered")
    def test_records_internally_consistent_for_many_seeds(self):
        d = tempfile.mkdtemp(); bad = []
        for typ in G.BUILDERS:
            for i in range(12): r = gen(d, typ, i, seed=99); b = V.consistency(r); bad += [(typ, i, b)] if b else []
        self.assertEqual(bad, [])
    def test_consistency_checker_detects_a_wrong_total(self):
        d = tempfile.mkdtemp(); r = gen(d, "salary_slip", 0); self.assertEqual(V.consistency(r), []); r["fields"]["net_pay"] += 500; self.assertIn("net=gross-deductions", V.consistency(r))
        r = gen(d, "school_fee_receipt", 0); r["fields"]["total"] += 1; self.assertIn("total=sum(heads)", V.consistency(r))
    def test_gstin_checksum_valid(self):
        r = random.Random(1)
        for _ in range(50): self.assertTrue(S.gstin_checksum_ok(G.gstin(r, "Karnataka")))
    def test_fictional_names_never_hit_the_real_brand_deny_list(self):
        r = random.Random(5)
        for kind in ("school", "insurer", "bank", "meal", "ev_dealer", "ev_brand", "charger", "employer", "amc"):
            for _ in range(300): G.org(r, kind)               # clean() asserts internally
        with self.assertRaises(AssertionError): G.clean("Sodexo Meal Services")
        with self.assertRaises(AssertionError): G.clean("HDFC Greenfield Bank")
    def test_amount_formatting_indian_grouping(self):
        self.assertEqual(G.inr(125000), "1,25,000.00"); self.assertEqual(G.inr(1234567.5), "12,34,567.50"); self.assertEqual(G.inr(999), "999.00"); self.assertEqual(G.words_inr(125000), "Rupees One Lakh Twenty-Five Thousand Only")
    def test_multipage_documents_paginate_onto_new_pages(self):
        d = tempfile.mkdtemp(); pages = collections.Counter(gen(d, "loan_statement", i, seed=3)["n_pages"] for i in range(40)); pages_f = collections.Counter(gen(d, "form_16", i, seed=3)["n_pages"] for i in range(20))
        self.assertTrue(max(pages_f) >= 2 or max(pages) >= 2)
    def test_degradation_modes_all_occur_and_images_decode(self):
        from PIL import Image
        d = tempfile.mkdtemp(); modes = set()
        for i in range(25):
            r = gen(d, "salary_slip", i, seed=11); modes.add(r["degrade"]["mode"])
            for p in r["pages"]: Image.open(Path(d) / p).verify()
        self.assertEqual(modes, {"clean", "scan", "photo"})

class Splits(unittest.TestCase):
    def test_layout_D_is_test_only(self):
        for g in ("a", "b", "c"): self.assertEqual(L.synthetic_split(g, "D"), "test")
    def test_same_group_same_split_and_ratios_roughly_70_15_15(self):
        sp = collections.Counter(L.synthetic_split(f"g{i}", "A") for i in range(4000)); self.assertAlmostEqual(sp["train"] / 4000, 0.70, delta=0.03); self.assertAlmostEqual(sp["val"] / 4000, 0.15, delta=0.03)
        self.assertEqual(L.synthetic_split("same", "B"), L.synthetic_split("same", "B"))
    def test_fatura_templates_never_split_across_sets(self):
        T = list(range(1, 51)); a = {t: L.template_split(t, T) for t in T}; c = collections.Counter(a.values()); self.assertEqual((c["train"], c["val"], c["test"]), (35, 7, 8)); self.assertEqual(a, {t: L.template_split(t, T) for t in T})

REAL = ROOT / "datasets/metadata/doc_type_manifest_v2.jsonl"
@unittest.skipUnless(REAL.exists(), "unified manifest not built")
class ValidatorMutations(unittest.TestCase):
    """the validator must pass the real manifest and FAIL on each deliberate corruption"""
    @classmethod
    def setUpClass(cls): cls.rows = L.read_jsonl(REAL); cls.tmp = Path(tempfile.mkdtemp())
    def run_validator(self, rows, name):
        m = self.tmp / f"{name}.jsonl"; L.write_jsonl(m, rows); rep = self.tmp / f"{name}.report.json"
        p = subprocess.run([sys.executable, str(DD / "validate_doc_type_manifest.py"), "--manifest", str(m), "--skip-files", "--report", str(rep)], capture_output=True, text=True, cwd=ROOT)
        return p.returncode, json.loads(rep.read_text()) if rep.exists() else {}
    def mutate(self, fn):
        rows = [dict(r) for r in self.rows]; fn(rows); return rows
    def test_unmodified_manifest_passes(self): rc, rep = self.run_validator(self.rows, "ok"); self.assertEqual(rc, 0); self.assertTrue(rep["passed"], rep.get("errors"))
    def test_unknown_doc_type_fails(self):
        rc, rep = self.run_validator(self.mutate(lambda rows: rows[0].update(doc_type="made_up_type")), "t1"); self.assertEqual(rc, 1); self.assertIn("unknown_doc_type", rep["errors"])
    def test_synthetic_marked_eval_eligible_fails(self):
        def f(rows): next(r for r in rows if r["synthetic"]).update(eval_eligible=True)
        rc, rep = self.run_validator(self.mutate(f), "t2"); self.assertEqual(rc, 1); self.assertIn("eval_eligible_vs_synthetic", rep["errors"])
    def test_eligibility_claim_fails(self):
        rc, rep = self.run_validator(self.mutate(lambda rows: rows[0].update(eligibility_status="eligible")), "t3"); self.assertEqual(rc, 1); self.assertIn("eligibility_status_invalid", rep["errors"])
    def test_missing_licence_entry_and_blocked_source_fail(self):
        rc, rep = self.run_validator(self.mutate(lambda rows: rows[1].update(license_register_id="no_such_source")), "t4"); self.assertEqual(rc, 1); self.assertIn("license_register_missing", rep["errors"])
        rc, rep = self.run_validator(self.mutate(lambda rows: rows[2].update(license_register_id="docile")), "t5"); self.assertEqual(rc, 1); self.assertIn("blocked_source_in_manifest", rep["errors"])
    def test_group_spanning_splits_fails(self):
        def f(rows):
            a = next(r for r in rows if r["split"] == "train"); b = next(r for r in rows if r["split"] == "test" and r["doc_id"] != a["doc_id"]); b["group_id"] = a["group_id"]
        rc, rep = self.run_validator(self.mutate(f), "t6"); self.assertEqual(rc, 1); self.assertIn("group_spans_splits", rep["errors"])
    def test_split_files_disagreement_fails(self):
        def f(rows): r = next(r for r in rows if not r["synthetic"] and r["split"] == "train"); r["split"] = "val"
        rc, rep = self.run_validator(self.mutate(f), "t7"); self.assertEqual(rc, 1); self.assertIn("split_file_disagrees", rep["errors"])
    def test_missing_required_field_fails(self):
        def f(rows): rows[0].pop("license")  if False else rows[0].pop("split_basis")
        rc, rep = self.run_validator(self.mutate(f), "t8"); self.assertEqual(rc, 1); self.assertIn("missing_fields", rep["errors"])
    def test_real_and_synthetic_never_mixed_in_split_manifests(self):
        for sp in ("train", "val", "test"):
            self.assertTrue(all(not r["synthetic"] for r in L.read_jsonl(L.META / "doc_type_v2_splits" / f"real_{sp}.jsonl"))); self.assertTrue(all(r["synthetic"] for r in L.read_jsonl(L.META / "doc_type_v2_splits" / f"synthetic_{sp}.jsonl")))
    def test_protected_test_documents_stay_in_test(self):
        prot = [r for r in self.rows if r["in_protected_task_test"]]; self.assertTrue(prot); self.assertTrue(all(r["split"] == "test" and not r["synthetic"] for r in prot))
        orig = {json.loads(l)["document_id"]: json.loads(l)["split"] for l in open(L.META / "manifest.jsonl")}
        for r in self.rows:
            if not r["synthetic"]: self.assertEqual(r["split"], orig[r["doc_id"]], "real split must equal the existing manifest split")

V21 = ROOT / "datasets/metadata/doc_type_manifest_v2_1.jsonl"
@unittest.skipUnless(V21.exists(), "v2.1 manifest not built")
class ManifestV21(unittest.TestCase):
    """v2.1: template-held-out generated splits, approval gate, corrected derived tax labels, protected assets untouched"""
    @classmethod
    def setUpClass(cls):
        cls.rows = L.read_jsonl(V21); cls.old = {r["doc_id"]: r for r in L.read_jsonl(REAL)}; cls.tmp = Path(tempfile.mkdtemp())
        cls.reg = {s["id"]: s for s in json.loads((L.META / "license_register_v2.json").read_text())["sources"]}
    def run_validator(self, rows, name):
        m = self.tmp / f"{name}.jsonl"; L.write_jsonl(m, rows); rep = self.tmp / f"{name}.report.json"
        p = subprocess.run([sys.executable, str(DD / "validate_doc_type_manifest.py"), "--manifest", str(m), "--schema", "v2_1", "--tag", "v2_1", "--skip-files", "--report", str(rep)], capture_output=True, text=True, cwd=ROOT)
        return p.returncode, json.loads(rep.read_text()) if rep.exists() else {}
    def test_v21_manifest_passes_validator(self): rc, rep = self.run_validator(self.rows, "ok21"); self.assertEqual(rc, 0, rep.get("errors")); self.assertTrue(rep["passed"])
    def test_no_template_spans_splits_and_layout_D_is_test(self):
        t = collections.defaultdict(set)
        for r in self.rows:
            if r["template_id"]: t[r["template_id"]].add(r["split"])
        self.assertTrue(t); self.assertTrue(all(len(v) == 1 for k, v in t.items() if not k.startswith("synthetic_in__")))
        self.assertEqual(len(t["synthetic_in__single_generator_template_family"]), 3, "KNOWN, documented exception: older protected synthetic split shares one template family across train/val/test")
        self.assertTrue(all(r["in_protected_task_test"] for r in self.rows if r["source"] == "synthetic_in" and r["split"] == "test"))
        self.assertTrue(all(r["split"] == "test" for r in self.rows if r["source"] == "synthgen_v1" and r["layout"] == "D"))
    def test_each_generated_type_has_train_val_and_test_with_distinct_layouts(self):
        by = collections.defaultdict(lambda: collections.defaultdict(set))
        for r in self.rows:
            if r["source"] == "synthgen_v1": by[r["doc_type"]][r["split"]].add(r["layout"])
        for typ, d in by.items():
            self.assertEqual(set(d), {"train", "val", "test"}, typ); self.assertEqual(len(d["val"]), 1); self.assertEqual(d["test"], {"D"}); self.assertEqual(len(d["train"]), 2); self.assertFalse(d["train"] & d["val"])
    def test_agami_split_by_template(self):
        m = {r["original"]["template"]: set() for r in self.rows if r["source"] == "agami"}
        for r in self.rows:
            if r["source"] == "agami": m[r["original"]["template"]].add(r["split"])
        self.assertTrue(all(len(v) == 1 for v in m.values())); self.assertEqual(len({next(iter(v)) for v in m.values()}), 3)
    def test_real_splits_and_protected_test_docs_unchanged(self):
        for r in self.rows:
            if r["source"] not in ("synthgen_v1", "agami"): self.assertEqual(r["split"], self.old[r["doc_id"]]["split"])
        self.assertTrue(all(r["split"] == "test" for r in self.rows if r["in_protected_task_test"]))
    def test_v2_files_are_not_rewritten_by_v21(self):
        for r in self.rows: self.assertEqual(r["split_v2_0"], self.old[r["doc_id"]]["split"])
    def test_approval_gate_excludes_unresolved_licences(self):
        for r in self.rows:
            st = self.reg[r["license_register_id"]]["status"]
            if r["approved_for_training"]: self.assertIn(st, ("usable", "usable_with_restrictions"), r["doc_id"])
            if st in ("conflict", "unverified", "blocked", "not_acquired"): self.assertFalse(r["approved_for_training"], r["doc_id"])
        cfg = json.loads((ROOT / "config/approved_sources_v2.json").read_text())
        for src in cfg["excluded_until_cleared"]: self.assertFalse(any(r["approved_for_training"] for r in self.rows if r["source"] == src), src)
        appr = L.read_jsonl(L.META / "doc_type_v2_1_approved/manifest.jsonl"); self.assertEqual({r["doc_id"] for r in appr}, {r["doc_id"] for r in self.rows if r["approved_for_training"]})
    def test_validator_rejects_approved_row_with_unresolved_licence(self):
        rows = [dict(r) for r in self.rows]; r = next(x for x in rows if x["source"] == "sroie"); r["approved_for_training"] = True
        rc, rep = self.run_validator(rows, "bad_appr"); self.assertEqual(rc, 1); self.assertIn("unresolved_licence_in_approved_training", rep["errors"])
    def test_derived_tax_labels_follow_subtype(self):
        for r in self.rows:
            if r["source"] != "synthgen_v1": continue
            if (r["doc_type"], r["subtype"]) == ("insurance_premium_receipt", "health"): self.assertEqual(r["tax_evidence_candidates"], ["section_80d_evidence"])
            if (r["doc_type"], r["subtype"]) == ("insurance_premium_receipt", "life"): self.assertEqual(r["tax_evidence_candidates"], ["section_80c_evidence"])
            if r["doc_type"] == "ev_document": self.assertNotIn("loan_interest_evidence", r["tax_evidence_candidates"])
            self.assertIn(r["eligibility_status"], ("pending_verification", "not_applicable"))
    def test_original_labels_preserved(self):
        base = {x["document_id"]: x for x in L.read_jsonl(L.META / "manifest.jsonl")}
        for r in random.Random(1).sample([r for r in self.rows if r["doc_id"] in base], 400):
            b = base[r["doc_id"]]; self.assertEqual((r["original"]["document_type"], r["original"]["expense_category"]), (b["document_type"], b["expense_category"]))
    def test_three_axes_are_separate(self):
        types = {r["doc_type"] for r in self.rows}; cats = {r["expense_category"] for r in self.rows if r["expense_category"]}; self.assertFalse(types & cats)

import tfidf_baseline_v2_1 as B
class BaselineHelpers(unittest.TestCase):
    def test_footer_is_stripped_and_digits_masked(self):
        t = "RENT RECEIPT Rs. 12,500\nSYNTHETIC SAMPLE - generated for ML dataset development - not a real document"
        c = B.clean(t); self.assertNotIn("synthetic", c); self.assertNotIn("not a real", c); self.assertIn("rent receipt", c)
        self.assertNotIn("1", B.clean(t, mask_digits=True)); self.assertIn("synthetic", B.clean(t, strip_footer=False))
    def test_title_masking_removes_phrase(self): self.assertNotIn("rent receipt", B.clean("RENT RECEIPT for May", mask_titles=["rent receipt"]))
    def test_metrics_per_class_support(self):
        m = B.metrics(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"]); self.assertEqual(m["per_class"]["a"]["support"], 2); self.assertNotIn("c", m["per_class"]); self.assertEqual(m["n_classes_with_support"], 2)
if __name__ == "__main__": unittest.main()
