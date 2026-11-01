"""Label-review outputs: coverage of the 286-record queue, status/blocked rules, evidence completeness, axis independence, label preservation and reversibility, no application of low-confidence or unapproved-source changes."""
import sys, json, hashlib, unittest
from pathlib import Path
import _path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts" / "dataset_discovery"))
import label_review as LR, apply_label_overlay as AO, doc_type_lib as L
OUT = LR.OUT; META = L.META
def jl(p): return L.read_jsonl(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Outputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = [META / "doc_type_manifest_v2_1.jsonl", META / "mapping_review_queue_v2_1.jsonl", META / "manifest.jsonl", ROOT / "datasets/processed/taskB/test.jsonl", ROOT / "config/approved_sources_v2.json"]; cls.before = {str(p): sha(p) for p in cls.guard}
        LR.main()  # regenerate: must be deterministic and must not touch inputs
        cls.res = jl(OUT / "review_results.jsonl"); cls.chg = jl(OUT / "proposed_changes.jsonl"); cls.ov = jl(OUT / "label_overlay_v1.jsonl"); cls.hq = jl(OUT / "human_review_queue.jsonl"); cls.q = jl(META / "mapping_review_queue_v2_1.jsonl")
        cls.man = {r["doc_id"]: r for r in jl(META / "doc_type_manifest_v2_1.jsonl")}; cls.excl = set(json.loads((ROOT / "config/approved_sources_v2.json").read_text())["excluded_until_cleared"])
    def test_inputs_and_protected_files_untouched(self):
        for p in self.guard: self.assertEqual(sha(p), self.before[str(p)], p)
    def test_every_queue_record_reviewed_exactly_once(self):
        self.assertEqual(len(self.q), 286); self.assertEqual([x["doc_id"] for x in self.res], [x["doc_id"] for x in self.q]); self.assertEqual(len({x["doc_id"] for x in self.res}), 286)
    def test_statuses_and_blocked_rule(self):
        for x in self.res:
            self.assertIn(x["status"], {"supported", "likely_incorrect", "ambiguous", "blocked_provenance"}); self.assertEqual(x["status"] == "blocked_provenance", x["source"] in self.excl, x["doc_id"]); self.assertEqual(x["approved_source"], x["source"] not in self.excl)
            self.assertTrue(x["reason"]); self.assertIn("evidence", x)
        s = json.loads((OUT / "summary.json").read_text()); self.assertEqual(sum(s["by_status"].values()), 286)
    def test_original_labels_preserved_in_results(self):
        for x in self.res:
            r = self.man[x["doc_id"]]; o = x["original_labels"]; self.assertEqual((o["doc_type"], o["expense_category"], o["tax_evidence_candidates"]), (r["doc_type"], r["expense_category"], r["tax_evidence_candidates"])); self.assertEqual(o["source_label"], r["original"])
    def test_every_change_records_old_new_evidence_confidence_reason(self):
        self.assertTrue(self.chg)
        for c in self.chg:
            for k in ("old", "proposed", "confidence", "evidence", "reason", "field", "change_id"): self.assertTrue(c[k], k)
            self.assertEqual(c["field"], "doc_type"); self.assertEqual(c["old"], self.man[c["doc_id"]]["doc_type"]); self.assertNotEqual(c["old"], c["proposed"]); self.assertIn(c["confidence"], ("high", "medium", "low"))
    def test_only_high_confidence_approved_changes_enter_overlay(self):
        ids = {o["doc_id"] for o in self.ov}
        for c in self.chg:
            self.assertEqual(c["applied"], c["confidence"] == "high" and c["approved_source"] and c["status"] == "likely_incorrect"); self.assertEqual(c["doc_id"] in ids, c["applied"])
        for o in self.ov: self.assertNotIn(self.man[o["doc_id"]]["source"], self.excl); self.assertEqual(o["confidence"], "high"); self.assertTrue(o["reversible"])
    def test_axes_independent_no_expense_or_tax_changes(self):
        self.assertTrue(all(c["field"] == "doc_type" for c in self.chg)); self.assertTrue(all(o["field"] == "doc_type" for o in self.ov))
        for x in self.res: self.assertTrue(x["tax_evidence"].startswith("not assessed")); self.assertEqual(x["original_labels"]["tax_evidence_candidates"], self.man[x["doc_id"]]["tax_evidence_candidates"])
    def test_human_queue_ranked_excludes_applied_and_blocked(self):
        applied = {o["doc_id"] for o in self.ov}
        for h in self.hq: self.assertIn(h["status"], ("likely_incorrect", "ambiguous")); self.assertTrue(h["approved_source"]); self.assertNotIn(h["doc_id"], applied); self.assertTrue(h["why"])
        self.assertEqual([h["rank"] for h in self.hq], list(range(1, len(self.hq) + 1))); st = [0 if h["status"] == "likely_incorrect" else 1 for h in self.hq]; self.assertEqual(st, sorted(st))
        self.assertEqual(len(self.hq), sum(1 for x in self.res if x["status"] in ("likely_incorrect", "ambiguous")) - len(applied))
    def test_deterministic(self):
        LR.main(); self.assertEqual(jl(OUT / "review_results.jsonl"), self.res)
class Overlay(unittest.TestCase):
    rows = [dict(doc_id="a", doc_type="receipt", expense_category="food", tax_evidence_candidates=[]), dict(doc_id="b", doc_type="invoice", expense_category=None, tax_evidence_candidates=[])]
    ov = [dict(change_id="X1", doc_id="a", field="doc_type", old="receipt", new="hotel_bill")]
    def test_apply_preserves_original_and_revert_restores_exactly(self):
        eff = AO.apply_overlay(self.rows, self.ov); self.assertEqual(eff[0]["doc_type"], "hotel_bill"); self.assertEqual(eff[0]["doc_type_original"], "receipt"); self.assertEqual(eff[0]["label_overlay_id"], "X1"); self.assertEqual(self.rows[0]["doc_type"], "receipt")
        self.assertEqual(eff[0]["expense_category"], "food"); self.assertEqual(AO.revert_overlay(eff), self.rows)
    def test_stale_unknown_and_forbidden_overlays_refused(self):
        with self.assertRaises(ValueError): AO.apply_overlay(self.rows, [dict(change_id="X", doc_id="a", field="doc_type", old="invoice", new="x")])
        with self.assertRaises(KeyError): AO.apply_overlay(self.rows, [dict(change_id="X", doc_id="zz", field="doc_type", old="a", new="b")])
        for f in ("expense_category", "tax_evidence_candidates"):
            with self.assertRaises(ValueError): AO.apply_overlay(self.rows, [dict(change_id="X", doc_id="a", field=f, old=self.rows[0][f], new="x")])
    def test_real_overlay_applies_cleanly_to_manifest_and_reverts(self):
        rows = jl(META / "doc_type_manifest_v2_1.jsonl"); ov = jl(OUT / "label_overlay_v1.jsonl"); eff = AO.apply_overlay(rows, ov); self.assertEqual(sum("label_overlay_id" in r for r in eff), len(ov)); self.assertEqual(AO.revert_overlay(eff), rows)
class Judge(unittest.TestCase):
    def rec(self, **kw):
        r = dict(doc_type="travel_document", expense_category="hotel", source="my_receipts", original=dict(label_quality="manual_review"), ann_path=None); r.update(kw); return r
    def test_expense_and_ocr_contradicting_type_gives_high_confidence_change(self):
        j = LR.judge(self.rec(), set(), "invoice arrival departure room reservation guest total " * 3); self.assertEqual((j["status"], j["proposed_doc_type"], j["confidence"]), ("likely_incorrect", "hotel_bill", "high"))
    def test_strong_source_label_with_conflicting_ocr_stays_ambiguous_low(self):
        r = self.rec(doc_type="utility_bill", expense_category="gas", original=dict(label_quality="manual_folder")); j = LR.judge(r, set(), "tax invoice gstin hsn cgst sgst lpg cylinder " * 3); self.assertEqual(j["status"], "ambiguous"); self.assertEqual(j["confidence"], "low")
    def test_excluded_source_is_blocked_but_verdict_kept(self):
        j = LR.judge(self.rec(source="sroie"), {"sroie"}, "hotel room guest arrival departure reservation " * 3); self.assertEqual(j["status"], "blocked_provenance"); self.assertEqual(j["evidence_verdict"], "likely_incorrect")
    def test_too_little_text_is_ambiguous(self): self.assertEqual(LR.judge(self.rec(), set(), "hotel room")["status"], "ambiguous")
    def test_supported_when_cues_agree(self):
        r = self.rec(doc_type="fuel_receipt", expense_category="fuel", original=dict(label_quality="source_label")); self.assertEqual(LR.judge(r, set(), "petrol pump litre diesel vehicle no total amount " * 2)["status"], "supported")
    def test_invoice_without_invoice_cues_but_receipt_cues_is_not_supported(self):
        r = self.rec(doc_type="invoice", expense_category="miscellaneous", original=dict(label_quality="source_label")); j = LR.judge(r, set(), "receipt subtotal tendered thank you cashier total " * 3); self.assertEqual(j["status"], "ambiguous"); self.assertEqual(j["proposed_doc_type"], "receipt")
if __name__ == "__main__": unittest.main()
