"""V2 step 1c - final orientation decision (supersedes the two probe passes, which were unreliable on dense, tiny-text invoices at 1800px).
1) OCR the EXIF-corrected page at 3000px with NO angle classifier; text-line box geometry says whether text runs horizontally (candidates 0/180) or vertically (90/270).
2) Run the other candidate too; keep the rotation with the higher sum(conf*alnum chars) (no cls => upside-down text recognises as low-confidence garbage).
3) Rewrite working image + OCR (with cls) only when the rotation differs from what is on disk. Originals untouched. Everything recorded in orientation.json under 'final'."""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_prepare_pool import ocr, POOL, MAXW
from common import DS
from run_ocr import _gpu_libs
import numpy as np
from PIL import Image, ImageOps

def raw(o, arr, cls):
    res, _ = o(arr, use_cls=cls)
    return [{"text": t, "bbox": [int(min(p[0] for p in b)), int(min(p[1] for p in b)), int(max(p[0] for p in b)), int(max(p[1] for p in b))], "conf": float(c)} for b, t, c in (res or [])]
def sc(ws): return sum(w["conf"] * len(re.findall(r"[A-Za-z0-9]", w["text"])) for w in ws)
def vert_frac(ws):
    v = h = 0
    for w in ws:
        bw, bh = w["bbox"][2] - w["bbox"][0], w["bbox"][3] - w["bbox"][1]
        a = bw * bh
        if len(w["text"]) < 3: continue
        if bh > 1.3 * bw: v += a
        elif bw > 1.3 * bh: h += a
    return v / max(v + h, 1)

if __name__ == "__main__":
    _gpu_libs()
    from rapidocr_onnxruntime import RapidOCR
    o = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, max_side_len=MAXW)
    ori = json.loads((POOL / "orientation.json").read_text()); n_ch = 0
    for i, (d, v) in enumerate(sorted(ori.items())):
        if "final" in v: continue
        im = ImageOps.exif_transpose(Image.open(DS / v["original"]).convert("RGB")); im.thumbnail((MAXW, MAXW))
        w0 = raw(o, np.array(im), False); vf = vert_frac(w0)
        cand = (90, 270) if vf > 0.5 else (0, 180)
        s = {}
        for r in cand:
            s[r] = sc(w0) if r == 0 else sc(raw(o, np.array(im.rotate(r, expand=True)), False))
        if vf <= 0.5: pass
        best = max(s, key=s.get); other = min(s.values()); ratio = round(s[best] / max(other, 1), 2)
        v["final"] = {"vert_frac_at_0": round(vf, 3), "candidates": list(cand), "scores": {str(k): round(x, 1) for k, x in s.items()}, "ratio": ratio, "rotation_ccw": best, "confident": ratio >= 1.25}
        if best != v["rotation_ccw"] or not (POOL / "ocr" / f"{d}.json").exists():
            wk = im.rotate(best, expand=True); wk.save(POOL / "images" / f"{d}.jpg", quality=92)
            ws = [dict(w, conf=round(w["conf"], 3)) for w in raw(o, np.array(wk), True)]
            (POOL / "ocr" / f"{d}.json").write_text(json.dumps({"document_id": d, "image": f"interim/v2_pool/images/{d}.jpg", "width": wk.width, "height": wk.height, "ocr_engine": "rapidocr-onnx-gpu", "words": ws}, ensure_ascii=False))
            v.update(rotation_ccw=best, n_words=len(ws), mean_conf=round(float(np.mean([w["conf"] for w in ws])), 3) if ws else 0, size=[wk.width, wk.height]); n_ch += 1
        if i % 10 == 0: print(i, n_ch, flush=True); (POOL / "orientation.json").write_text(json.dumps(ori))
    (POOL / "orientation.json").write_text(json.dumps(ori, indent=0)); print("changed", n_ch)
