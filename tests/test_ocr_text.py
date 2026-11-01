"""Glued date+time repair (app/ocr_text.py). Pure text - no model. Examples are real RapidOCR outputs seen on the validation split (reports/extraction_audit)."""
import unittest, random, _path
import ocr_text as T
from words import split_words

class Repair(unittest.TestCase):
    def test_real_examples(self):
        for raw, want in [("25/12/20188:13:39PM", "25/12/2018 8:13:39PM"), ("Date:05-03-201819:02:50", "Date:05-03-2018 19:02:50"), ("17/1/201812:12:21", "17/1/2018 12:12:21"),
                          ("Date:25Mar 20189:27PM", "Date:25Mar 2018 9:27PM"), ("15/01/20181:48:36PM", "15/01/2018 1:48:36PM"), ("2018-12-2508:13", "2018-12-25 08:13")]:
            self.assertEqual(T.repair_text(raw), want, raw)
    def test_untouched_when_nothing_is_glued(self):
        for t in ["25/12/2018 8:13:39PM", "Date: 25/12/2018", "Total 1,234.50", "INV/2024/0012", "GSTIN 27AAPFU0939F1Z8", "Tel:03-62619337", "12:30", "Rs.1,500.00 12/05/2024"]:
            self.assertEqual(T.repair_text(t), t, t)
    def test_idempotent_and_only_inserts_spaces(self):
        rnd = random.Random(1); alphabet = "0123456789/-.: APMabc"
        for _ in range(3000):
            t = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 30))); r = T.repair_text(t)
            self.assertEqual(r.replace(" ", ""), t.replace(" ", "")); self.assertEqual(T.repair_text(r), r)      # no digit/char is ever changed, nothing is repaired twice
    def test_segments_keep_boxes_and_do_not_mutate_input(self):
        segs = [dict(text="25/12/20188:13:39PM", bbox=[10, 20, 300, 44], conf=0.9), dict(text="TOTAL 9.00", bbox=[10, 60, 200, 80], conf=0.95)]; before = [dict(s) for s in segs]
        out = T.repair_segments(segs); self.assertEqual(segs, before)
        self.assertEqual(out[0]["text"], "25/12/2018 8:13:39PM"); self.assertEqual(out[0]["bbox"], [10, 20, 300, 44]); self.assertEqual(out[0]["text_original"], "25/12/20188:13:39PM")
        self.assertIs(out[1], segs[1]); self.assertEqual(T.repair_segments(None), [])
    def test_downstream_word_split_now_exposes_the_date(self):
        ws = [w["text"] for w in split_words(T.repair_segments([dict(text="25/12/20188:13:39PM", bbox=[0, 0, 190, 20])]))]
        self.assertEqual(ws, ["25/12/2018", "8:13:39PM"]); self.assertNotIn("25/12/2018", [w["text"] for w in split_words([dict(text="25/12/20188:13:39PM", bbox=[0, 0, 190, 20])])])

if __name__ == "__main__": unittest.main()
