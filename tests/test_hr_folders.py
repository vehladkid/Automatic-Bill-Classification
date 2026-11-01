"""HR review folders on disk follow claim state (submitted -> approved / rejected)."""
import os, sys, json, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
os.environ["HRBC_OUT"] = tempfile.mkdtemp()
from tax import store as ST, hr_folders as HF, quota as Q
class T(unittest.TestCase):
    def test_flow(self):
        out = Path(os.environ["HRBC_OUT"]); Q.seed_default_profiles()
        (out / "classified/rent_receipt").mkdir(parents=True); f = out / "classified/rent_receipt/D1__r.jpg"; f.write_bytes(b"x"); (out / "results/D1").mkdir(parents=True)
        (out / "results/D1/record.json").write_text(json.dumps(dict(doc_id="D1", saved_path="classified/rent_receipt/D1__r.jpg", category="rent_receipt", owner="E1")))
        c = ST.add_claim("E1", "2026-27", "hra_rent", 10000, {}, docs=["D1"], actor="E1"); r = out / "hr_review"
        self.assertTrue((r / "processed/D1__r.jpg").is_symlink()); self.assertTrue((r / "draft/E1" / f"{c['id']}.json").exists())
        ST.submit("E1", "2026-27", "E1"); self.assertTrue((r / "pending_approval/E1" / f"{c['id']}.json").exists()); self.assertFalse((r / "draft/E1").exists())
        ST.hr_decide(c["id"], "approved", "ok", "HR1", "hr"); self.assertTrue((r / "approved/E1" / f"{c['id']}.json").exists()); self.assertTrue((r / "approved/E1" / f"{c['id']}__D1__r.jpg").exists())
        c2 = ST.add_claim("E1", "2026-27", "hra_rent", 5000, {}, docs=["D1"], actor="E1"); ST.submit("E1", "2026-27", "E1"); ST.hr_decide(c2["id"], "rejected", "no PAN", "HR1", "hr")
        self.assertTrue((r / "rejected/E1" / f"{c2['id']}.json").exists())
if __name__ == "__main__": unittest.main()
