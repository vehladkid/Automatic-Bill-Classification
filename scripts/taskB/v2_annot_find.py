import sys, json, re
from pathlib import Path
POOL = Path(__file__).resolve().parents[2] / "datasets/interim/v2_pool"
o = json.load(open(POOL / "ocr" / f"{sys.argv[1]}.json"))
for w in o["words"]:
    if re.search(sys.argv[2], w["text"], re.I): print(w["bbox"], repr(w["text"]), w["conf"])
