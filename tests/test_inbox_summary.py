"""Inbox row summary: merchant/amount/date shown only as verified when calibrated confidence is adequate; missing values stay missing; review reasons are listed."""
import unittest, _path
from ingest.api import summary
class T(unittest.TestCase):
    def rec(self, **k): return dict({"task_b": {"fields": {"merchant_name": {"value": "ACME", "confidence": 0.9}, "total_amount": {"value": 10.0, "confidence": 0.3}}}, "task_a": {"route": "auto"}, "warnings": []}, **k)
    def test_verified_unverified_missing(self):
        s, why = summary(self.rec())
        self.assertTrue(s["merchant"]["verified"]); self.assertFalse(s["amount"]["verified"]); self.assertIsNone(s["date"]["value"])
        self.assertIn("amount unverified", why); self.assertIn("date not found", why)
    def test_low_task_a_confidence_and_warnings_flagged(self):
        s, why = summary(self.rec(task_a={"route": "hr_review"}, warnings=["OCR failed on page 2."]))
        self.assertIn("expense category below confidence threshold", why); self.assertIn("OCR failed on page 2.", why)
    def test_uncalibrated_value_is_not_verified(self):
        s, _ = summary({"task_b": {"fields": {"merchant_name": {"value": "X", "confidence": None}}}}); self.assertFalse(s["merchant"]["verified"])
    def test_no_fields_at_all(self):
        s, why = summary({}); self.assertTrue(all(v["value"] is None for v in s.values())); self.assertEqual(len(why), 3)
if __name__ == "__main__": unittest.main()
