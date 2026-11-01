import unittest, _path
from field_sanity import drop_implausible
from llm import config as C
class T(unittest.TestCase):
    def test_zero_total_dropped_and_recorded(self):
        k, d = drop_implausible({"total_amount": {"value": 0.0}, "merchant_name": {"value": "X"}}); self.assertNotIn("total_amount", k); self.assertIn("merchant_name", k); self.assertEqual(d[0]["field"], "total_amount")
    def test_nonzero_and_other_zero_fields_kept(self):
        k, d = drop_implausible({"total_amount": {"value": 10.0}, "cgst": {"value": 0.0}}); self.assertEqual(set(k), {"total_amount", "cgst"}); self.assertEqual(d, [])
    def test_empty(self): self.assertEqual(drop_implausible(None), ({}, []))
    def test_llm_text_only_by_default(self): self.assertFalse(C.settings({})["use_image"]); self.assertEqual(C.settings({})["mode"], "off")
if __name__ == "__main__": unittest.main()
