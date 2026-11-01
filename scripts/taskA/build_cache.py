"""Resize every classification image once (max side 768, JPEG q90) into datasets/interim/taskA_cache/. Originals untouched; resumable."""
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from common_a import *
def one(r):
    out = Path(r["cache"])
    if out.exists(): return 0
    try:
        with Image.open(r["image"]) as im:
            im = im.convert("RGB"); s = 768 / max(im.size)
            if s < 1: im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            out.parent.mkdir(parents=True, exist_ok=True); im.save(out, "JPEG", quality=90)
        return 0
    except Exception as e: print("FAIL", r["id"], str(e)[:80]); return 1
if __name__ == "__main__":
    rows = [r for s in SPLITS for r in load(s)]
    with ProcessPoolExecutor(10) as ex: bad = sum(ex.map(one, rows, chunksize=32))
    print("cached", len(rows) - bad, "failed", bad)
