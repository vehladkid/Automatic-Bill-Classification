"""Re-generates every figure and CSV table from outputs/ in a TEMP directory (aggregate.py logic, outputs redirected so nothing is overwritten) and compares them with the saved ones:
figures pixel-by-pixel, CSVs byte-by-byte. Shows that the graphs and tables in the report are exactly what the saved results produce."""
import sys, re, json, tempfile, hashlib, runpy
from pathlib import Path
import numpy as np
from PIL import Image
EXP = Path(__file__).resolve().parents[1]; TMP = Path(tempfile.mkdtemp(prefix="agg_verify_"))
src = (EXP / "aggregate.py").read_text(); src = src.replace("from exp_common import *", "from exp_common import *\nEXP = Path(r'%s')\ndef save_json(name, obj): (EXP / name).write_text(json.dumps(obj, indent=1, default=float))" % TMP, 1)
(TMP / "agg_copy.py").write_text(src); sys.path.insert(0, str(EXP)); runpy.run_path(str(TMP / "agg_copy.py"), run_name="__main__")
res = dict(figures={}, csv={})
for p in sorted((EXP / "figures").glob("*.png")):
    q = TMP / "figures" / p.name; a, b = np.asarray(Image.open(p).convert("RGB")), np.asarray(Image.open(q).convert("RGB")) if q.exists() else None
    res["figures"][p.name] = bool(b is not None and a.shape == b.shape and (a == b).all())
for p in sorted(EXP.glob("table_*.csv")): q = TMP / p.name; res["csv"][p.name] = bool(q.exists() and q.read_bytes() == p.read_bytes())
a = json.loads((EXP / "outputs/results.json").read_text()); b = json.loads((TMP / "results.json").read_text()); res["results_json_identical"] = (a == b)
print(json.dumps(res, indent=1)); (EXP / "verification/figures_check.json").write_text(json.dumps(res, indent=1))
