"""V2 step 1: orientation-correct + OCR the in_grocery_gst pool. Originals are never modified.
Orientation = the 0/90/180/270 rotation whose OCR text has the highest sum(conf*alnum_chars) (after EXIF transpose).
Outputs (all under datasets/interim/v2_pool/): images/<id>.jpg (corrected working copy, long side<=3000), ocr/<id>.json, orientation.json."""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DS, META
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_ocr import _gpu_libs
import numpy as np
from PIL import Image, ImageOps
POOL = DS / "interim" / "v2_pool"
MAXW = 3000

def ocr(o, arr):
    res, _ = o(arr)
    return [{"text": t, "bbox": [int(min(p[0] for p in b)), int(min(p[1] for p in b)), int(max(p[0] for p in b)), int(max(p[1] for p in b))], "conf": round(float(c), 3)} for b, t, c in (res or [])]

def score(ws): return sum(w["conf"] * len(re.findall(r"[A-Za-z0-9]", w["text"])) for w in ws)

if __name__ == "__main__":
    _gpu_libs()
    from rapidocr_onnxruntime import RapidOCR
    o_probe = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, max_side_len=1600)
    o_full = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, max_side_len=MAXW)
    rows = [json.loads(l) for l in open(META / "manifest.jsonl")]
    rows = [r for r in rows if r["source_dataset"] == "in_grocery_gst" and not r.get("drop_reason")]
    out = {}
    of = POOL / "orientation.json"
    if of.exists(): out = json.loads(of.read_text())
    for i, r in enumerate(rows):
        d = r["document_id"]
        if d in out and (POOL / "ocr" / f"{d}.json").exists(): continue
        im = ImageOps.exif_transpose(Image.open(DS / r["file_path"]).convert("RGB"))
        sc = {}
        small = im.copy(); small.thumbnail((1800, 1800))
        for rot in (0, 90, 180, 270):  # PIL rotate is counter-clockwise
            sc[rot] = score(ocr(o_probe, np.array(small.rotate(rot, expand=True))))
        best = max(sc, key=sc.get)
        wk = im.rotate(best, expand=True)
        wk.thumbnail((MAXW, MAXW))
        wk.save(POOL / "images" / f"{d}.jpg", quality=92)
        ws = ocr(o_full, np.array(wk))
        (POOL / "ocr" / f"{d}.json").write_text(json.dumps({"document_id": d, "image": f"interim/v2_pool/images/{d}.jpg", "width": wk.width, "height": wk.height, "ocr_engine": "rapidocr-onnx-gpu", "words": ws}, ensure_ascii=False))
        sorted_sc = sorted(sc.values(), reverse=True)
        out[d] = {"original": r["file_path"], "exif_applied": True, "rotation_ccw": best, "scores": sc, "orientation_margin": round(sorted_sc[0] / max(sorted_sc[1], 1), 2), "n_words": len(ws), "mean_conf": round(float(np.mean([w["conf"] for w in ws])) if ws else 0, 3), "size": [wk.width, wk.height]}
        if i % 10 == 0: print(i, d, best, out[d]["n_words"], out[d]["mean_conf"], flush=True); of.write_text(json.dumps(out))
    of.write_text(json.dumps(out, indent=0))
    print("done", len(out))
