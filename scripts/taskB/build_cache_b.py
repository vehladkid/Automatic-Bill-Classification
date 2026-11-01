"""Small image cache for Task B (max side 448, JPEG q85) -> datasets/interim/taskB_cache/<document_id>.jpg ; resumable."""
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from schema import *
from data import load
OUTD = DS / "interim/taskB_cache"
def one(d):
    out = OUTD / f"{d['document_id']}.jpg"
    if out.exists(): return 0
    try:
        with Image.open(DS / d["image"]) as im:
            im = im.convert("RGB"); s = 448 / max(im.size)
            if s < 1: im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            OUTD.mkdir(parents=True, exist_ok=True); im.save(out, "JPEG", quality=85)
        return 0
    except Exception as e: print("FAIL", d["document_id"], str(e)[:60]); return 1
if __name__ == "__main__":
    docs = [d for s in ("train", "val", "test") for d in load(s)]
    with ProcessPoolExecutor(10) as ex: bad = sum(ex.map(one, docs, chunksize=32))
    print("cached", len(docs) - bad, "failed", bad)
