"""LLM layer: hallucination prevention, OCR-correction validation, privacy guard, fusion and review flagging. Uses a fake client - no model needed."""
import unittest, _path
from llm import validate as V, enhance as EN, config as C, fuse as F
from llm.client import LLMUnavailable, parse_json
S = V.S

def make_gstin(body14="27AAPFU0939F1Z"):
    for last in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if S.gstin_checksum_ok(body14 + last): return body14 + last
GST = make_gstin()
def seg(t, y=0, c=0.95): return dict(text=t, bbox=[10, y, 400, y + 20], conf=c)
SEGS = [seg("ACME STORES PVT LTD", 10), seg(f"GSTIN {GST}", 40), seg("Invoice No: INV-2041", 70), seg("Date 12/03/2026", 100), seg("Sub Total 1,000.00", 130), seg("CGST 90.00", 160), seg("SGST 90.00", 190), seg("Total 1,180.00", 220)]
V1 = {"merchant_name": dict(value="ACME STORES PVT LTD", raw_text="ACME STORES PVT LTD", confidence=0.9, source="layoutlm", bbox=None),
      "total_amount": dict(value=1180.0, raw_text="1,180.00", confidence=0.95, source="layoutlm", bbox=None),
      "invoice_date": dict(value="2026-03-12", raw_text="12/03/2026", confidence=0.9, source="layoutlm", bbox=None)}
class Fake:
    def __init__(self, js=None, err=None): self.js, self.err = js, err; self.s = C.settings({}) | {"mode": "always"}
    def chat_json(self, system, user, image=None):
        if self.err: raise self.err
        return self.js, str(self.js), 0.1
def run(js, v1=V1, segs=SEGS, mode="always", err=None, force=False):
    return EN.enhance(segs, (500, 400), v1, None, client=Fake(js, err), cfg=C.settings({}) | {"mode": mode, "cache": False, "llm_only": "add"}, force=force)
def fld(v, ev=None): return {"value": v, "evidence": ev or str(v)}

class Hallucination(unittest.TestCase):
    def test_invented_amount_rejected_not_used(self):
        r = run({"fields": {"total_amount": fld(1999.0), "subtotal": fld(1000.0)}}); self.assertEqual(r["status"], "ok")
        self.assertEqual(r["fields"]["total_amount"]["value"], 1180.0)                              # V1 value stays
        self.assertTrue(any(x["field"] == "total_amount" and "does not appear" in x["why"] for x in r["rejected"]))
    def test_invented_merchant_and_invoice_rejected(self):
        r = run({"fields": {"merchant_name": fld("Reliance Retail"), "invoice_number": fld("ZX-9999")}}, v1={}); self.assertNotIn("merchant_name", r["fields"]); self.assertNotIn("invoice_number", r["fields"])
        self.assertEqual({x["field"] for x in r["rejected"]}, {"merchant_name", "invoice_number"})
    def test_invented_gstin_rejected(self):
        r = run({"fields": {"gstin": fld("29ABCDE1234F1Z5")}}, v1={}); self.assertNotIn("gstin", r["fields"])
    def test_invented_date_rejected(self):
        r = run({"fields": {"invoice_date": fld("2025-01-01")}}, v1={}); self.assertNotIn("invoice_date", r["fields"])
    def test_hsn_must_be_in_ocr(self):
        r = run({"fields": {"hsn_sac": fld(["9983", "1234"])}}, v1={}, segs=SEGS + [seg("HSN 9983", 250)]); self.assertEqual(r["fields"]["hsn_sac"]["value"], ["9983"])
    def test_null_fields_not_invented(self):
        r = run({"fields": {"invoice_number": {"value": None}, "gstin": None}}, v1={}); self.assertEqual(r["fields"], {})
    def test_grounded_llm_only_value_added_but_flagged_for_review(self):
        r = run({"fields": {"invoice_number": fld("INV-2041")}}, v1={}); p = r["fields"]["invoice_number"]
        self.assertEqual(p["value"], "INV2041"); self.assertTrue(p["needs_review"]); self.assertEqual(p["origin"], "llm")
    def test_garbage_llm_output_is_bad_output_and_v1_untouched(self):
        r = run("not a dict"); self.assertEqual(r["status"], "bad_output"); self.assertIsNone(r["fields"])          # caller keeps the V1 result
    def test_unparseable_json(self):
        with self.assertRaises(Exception): parse_json("no json here")

class OcrCorrection(unittest.TestCase):
    def setUp(self): self.ocr = V.OCR(SEGS)
    def test_gstin_confusable_fix_with_valid_checksum_accepted(self):
        garbled = GST.replace("0", "O", 1); o = V.OCR([seg(f"GSTIN {garbled}", 40)])
        ok, why = V.validate_correction(dict(ocr=garbled, corrected=GST), o); self.assertTrue(ok, why)
        self.assertEqual(V.ground("gstin", GST, o)["status"], "corrected")
    def test_gstin_correction_rejected_when_checksum_fails(self):
        wrong = GST[:-1] + ("A" if GST[-1] != "A" else "B"); o = V.OCR([seg(f"GSTIN {wrong}", 40)])
        ok, why = V.validate_correction(dict(ocr=wrong, corrected=wrong), o); self.assertFalse(ok)
    def test_correction_of_text_not_in_ocr_rejected(self):
        ok, why = V.validate_correction(dict(ocr="TOTAL 9999", corrected="TOTAL 9990"), self.ocr); self.assertFalse(ok); self.assertIn("not in the OCR", why)
    def test_non_confusable_rewrite_rejected(self):
        ok, why = V.validate_correction(dict(ocr="Total 1,180.00", corrected="Total 7,777.00"), self.ocr); self.assertFalse(ok)
    def test_amount_O_for_zero_correction_accepted(self):
        o = V.OCR([seg("Total 1,18O.00", 5)]); ok, why = V.validate_correction(dict(ocr="1,18O.00", corrected="1,180.00"), o); self.assertTrue(ok, why)
    def test_date_letter_for_digit_accepted_only_if_valid_date(self):
        o = V.OCR([seg("Date 12/O3/2026", 5)]); ok, _ = V.validate_correction(dict(ocr="12/O3/2026", corrected="12/03/2026"), o); self.assertTrue(ok)
        ok2, _ = V.validate_correction(dict(ocr="12/O3/2026", corrected="32/03/2026"), o); self.assertFalse(ok2)
    def test_invoice_id_one_char_confusable_is_corrected_status(self):
        o = V.OCR([seg("Invoice INV-2O41", 5)]); g = V.ground("invoice_number", "INV2041", o); self.assertEqual(g["status"], "corrected")
    def test_corrections_reported_in_enhance_result(self):
        r = run({"fields": {}, "ocr_corrections": [dict(ocr="TOTAL 9999", corrected="TOTAL 9990", reason="x")]}); self.assertFalse(r["corrections"][0]["accepted"])

class FusionAndReview(unittest.TestCase):
    def test_agreement_scores_higher_than_single_source(self):
        agree = run({"fields": {"total_amount": fld(1180.0)}})["fields"]["total_amount"]; single = run({"fields": {}})["fields"]["total_amount"]
        self.assertGreater(agree["evidence_score"], single["evidence_score"]); self.assertEqual(agree["origin"], "v1+llm")
    def test_conflict_flagged_for_human_review(self):
        bad = dict(V1, total_amount=dict(V1["total_amount"], value=1000.0, confidence=0.5))
        r = run({"fields": {"total_amount": fld(1180.0)}}, v1=bad); p = r["fields"]["total_amount"]
        self.assertTrue(p["needs_review"]); self.assertEqual(r["conflicts"][0]["field"], "total_amount")
    def test_arithmetic_mismatch_flags_review(self):
        v1 = {"subtotal": dict(value=1000.0, confidence=0.9, source="x"), "tax_total": dict(value=180.0, confidence=0.9, source="x"), "total_amount": dict(value=1500.0, confidence=0.9, source="x")}
        r = run({"fields": {}}, v1=v1); self.assertTrue(r["fields"]["total_amount"]["needs_review"]); self.assertTrue(any("MISMATCH" in e for e in r["fields"]["total_amount"]["evidence"]))
    def test_arithmetic_ok_adds_evidence(self):
        v1 = {"subtotal": dict(value=1000.0, confidence=0.9, source="x"), "tax_total": dict(value=180.0, confidence=0.9, source="x"), "total_amount": dict(value=1180.0, confidence=0.9, source="x")}
        self.assertTrue(any("arithmetic ok" in e for e in run({"fields": {}}, v1=v1)["fields"]["total_amount"]["evidence"]))
    def test_zero_total_never_high_confidence(self):
        r = run({"fields": {}}, v1={"total_amount": dict(value=0.0, confidence=0.99, source="x")}); self.assertTrue(r["fields"]["total_amount"]["needs_review"])
    def test_llm_cannot_override_confident_grounded_v1_value(self):
        r = run({"fields": {"merchant_name": fld("ACME STORES")}}); self.assertEqual(r["fields"]["merchant_name"]["value"], "ACME STORES PVT LTD")
    def test_v1_ungrounded_value_replaced_by_grounded_llm_value(self):
        bad = {"total_amount": dict(value=1999.0, raw_text="1,999.00", confidence=0.5, source="ranker")}
        r = run({"fields": {"total_amount": fld(1180.0)}}, v1=bad); self.assertEqual(r["fields"]["total_amount"]["value"], 1180.0); self.assertTrue(r["fields"]["total_amount"]["needs_review"])

class Gating(unittest.TestCase):
    CLEAN = dict(V1, gstin=dict(value=GST, confidence=0.9, source="x"))
    def test_skipped_when_v1_is_clean_in_auto_mode(self): self.assertEqual(run({"fields": {}}, v1=self.CLEAN, mode="auto")["status"], "skipped")
    def test_called_when_field_missing(self): self.assertEqual(run({"fields": {}}, v1={}, mode="auto")["status"], "ok")
    def test_called_when_gstin_checksum_bad(self):
        bad = GST[:-1] + ("A" if GST[-1] != "A" else "B"); v1 = dict(self.CLEAN, gstin=dict(value=bad, confidence=0.9, source="x")); self.assertIn("GSTIN checksum invalid", EN.needs_llm(v1, V.OCR(SEGS)))
    def test_off_mode_never_calls(self): self.assertEqual(run({"fields": {}}, v1={}, mode="off")["status"], "off")
    def test_unavailable_server_falls_back_without_raising(self):
        r = run(None, err=LLMUnavailable("connection refused"), v1=V1); self.assertEqual(r["status"], "unavailable"); self.assertIsNone(r["fields"])

class Privacy(unittest.TestCase):
    def s(self, **kw): return C.settings({}) | kw
    def test_loopback_allowed(self):
        for u in ("http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"): self.assertEqual(C.check_privacy(self.s(url=u)), "local")
    def test_remote_blocked_by_default(self):
        with self.assertRaises(C.PrivacyError): C.check_privacy(self.s(url="https://api.example.com"))
    def test_remote_needs_both_flag_and_exact_ack(self):
        with self.assertRaises(C.PrivacyError): C.check_privacy(self.s(url="http://10.0.0.5:11434", allow_remote=True))
        with self.assertRaises(C.PrivacyError): C.check_privacy(self.s(url="http://10.0.0.5:11434", allow_remote=True, remote_ack="yes"))
        self.assertEqual(C.check_privacy(self.s(url="http://10.0.0.5:11434", allow_remote=True, remote_ack=C.ACK)), "remote-acknowledged")
    def test_privacy_block_surfaces_as_status_and_sends_nothing(self):
        called = []
        class Spy(Fake):
            def chat_json(self, *a, **k): called.append(1); return super().chat_json(*a, **k)
        r = EN.enhance(SEGS, (500, 400), {}, None, client=Spy({"fields": {}}), cfg=C.settings({}) | {"mode": "always", "url": "https://evil.example.com"}); self.assertEqual(r["status"], "privacy_blocked"); self.assertEqual(called, [])
    def test_default_mode_is_off(self): self.assertEqual(C.settings({})["mode"], "off")
if __name__ == "__main__": unittest.main()

class Salvage(unittest.TestCase):
    def test_truncated_repetition_loop_keeps_complete_fields_only(self):
        from llm.client import parse_json
        v = parse_json('{"fields": {"merchant_name": "ASIA MART", "total_amount": "32.70", "hsn_sac": ["8888", "8888", "88'); self.assertEqual(v["fields"]["total_amount"], "32.70"); self.assertNotIn("88", str(v["fields"].get("hsn_sac", [])).replace("8888", ""))
    def test_garbage_still_rejected(self):
        from llm.client import parse_json, LLMBadOutput
        with self.assertRaises(LLMBadOutput): parse_json("sorry, I cannot read this image")
    def test_salvaged_values_still_must_be_grounded(self):
        from llm.client import parse_json
        v = parse_json('{"fields": {"total_amount": {"value": 999999.0}, "merchant_name": {"value": "ACME STORES PVT LTD"}, "hsn_sac": {"value": ["1', ); r = run(v)
        self.assertNotIn("total_amount", [k for k, p in r["fields"].items() if p.get("origin") == "llm"]); self.assertTrue(any(x["field"] == "total_amount" for x in r["rejected"]))


class LlmOnlyPolicy(unittest.TestCase):
    def go(self, policy, v1=None):
        return EN.enhance(SEGS, (500, 400), v1 or {}, None, client=Fake({"fields": {"invoice_number": fld("INV-2041")}}), cfg=C.settings({}) | {"mode": "always", "cache": False, "llm_only": policy})
    def test_default_policy_is_suggest(self): self.assertEqual(C.settings({})["llm_only"], "suggest")
    def test_llm_only_value_is_a_suggestion_not_a_field(self):
        r = self.go("suggest"); self.assertNotIn("invoice_number", r["fields"]); self.assertEqual(r["suggestions"]["invoice_number"]["value"], "INV2041"); self.assertTrue(r["suggestions"]["invoice_number"]["needs_review"])
    def test_add_policy_restores_old_behaviour(self): r = self.go("add"); self.assertIn("invoice_number", r["fields"]); self.assertEqual(r["suggestions"], {})
    def test_agreement_with_v1_still_a_field(self):
        r = self.go("suggest", v1={"invoice_number": dict(value="INV2041", confidence=0.7, source="layoutlm")}); self.assertEqual(r["fields"]["invoice_number"]["origin"], "v1+llm")
