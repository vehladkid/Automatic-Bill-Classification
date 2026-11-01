"""Independent leakage audit (does NOT reuse the grouping rule): for every real val/test image, nearest TRAIN image by pHash Hamming distance.
Writes datasets/metadata/leakage_audit.json. Reports counts at <=4 (near-identical) and <=8 (similar template)."""
import json, numpy as np, imagehash
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from common import *
def h(r):
    try:
        with Image.open(DS / r["file_path"]) as im: return np.array(imagehash.phash(im.convert("L"), 8).hash).ravel()
    except Exception: return np.zeros(64, bool)
if __name__ == "__main__":
    rows = [json.loads(l) for l in open(META / "manifest.jsonl")]; real = [r for r in rows if not r["synthetic"]]
    with ProcessPoolExecutor(8) as ex: H = np.stack(list(ex.map(h, real, chunksize=64)))
    sp = np.array([r["split"] for r in real]); tr = np.where(sp == "train")[0]; out = {}
    for s in ("val", "test"):
        idx = np.where(sp == s)[0]; n8 = n4 = n0 = 0; ex_ = []
        for i in idx:
            d = (H[tr] != H[i]).sum(1); m = int(d.min()); n8 += m <= 8; n4 += m <= 4; n0 += m == 0
            if m <= 4: ex_.append([real[i]["document_id"], real[tr[d.argmin()]]["document_id"], m])
        out[s] = {"real_docs": len(idx), "train_neighbour_le8": int(n8), "train_neighbour_le4": int(n4), "identical_phash": int(n0), "examples_le4": ex_[:10]}
    dump(META / "leakage_audit.json", out); print(json.dumps({k: {a: b for a, b in v.items() if a != "examples_le4"} for k, v in out.items()}), out["test"]["examples_le4"][:4], out["val"]["examples_le4"][:4])
