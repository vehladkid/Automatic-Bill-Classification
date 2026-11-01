"""V2 step 1b (pass 2 = no angle classifier; pass 1 vocab+cls could not tell upright from upside-down because the cls model flips lines before recognition): the conf*chars orientation score is fooled by rotated/upside-down text (cls model flips lines). Re-decide orientation for every pool doc with a
vocabulary score (count of invoice words recognised in the OCR of each of the 4 rotations); keeps orientation.json history, rewrites working image + OCR only if the rotation changes."""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_prepare_pool import ocr, POOL, MAXW
from common import DS
from run_ocr import _gpu_libs
import numpy as np
from PIL import Image, ImageOps
V = set("GST TOTAL INVOICE AMOUNT RATE QTY PRODUCT DATE TAX HSN MRP BILL RUPEES ONLY DESCRIPTION PARTICULARS SIGNATORY BANK IFSC TERMS CONDITIONS CGST SGST IGST DISC TAXABLE VALUE NET PAYABLE ROUND SUPPLIER BUYER ADDRESS KOLKATA BENGAL PHONE MOBILE PAN STATE CODE NO GOODS ITEM PCS UNIT CASH CREDIT SALE TCS".split())
def ocr2(o, arr):
    res, _ = o(arr, use_cls=False)
    return [{"text": t} for b, t, c in (res or [])]
def vscore(ws):
    return sum(1 for w in ws for t in re.findall(r"[A-Za-z]{2,}", w["text"].upper()) if t in V)
if __name__ == "__main__":
    _gpu_libs()
    from rapidocr_onnxruntime import RapidOCR
    o_probe = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, max_side_len=1600)
    o_full = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, max_side_len=MAXW)
    ori = json.loads((POOL / "orientation.json").read_text()); changed = []
    for i, (d, v) in enumerate(sorted(ori.items())):
        cur = json.loads((POOL / "ocr" / f"{d}.json").read_text())["words"]
        if "nocls_scores" in v: continue
        im = ImageOps.exif_transpose(Image.open(DS / v["original"]).convert("RGB")); small = im.copy(); small.thumbnail((1800, 1800))
        vs = {rot: vscore(ocr2(o_probe, np.array(small.rotate(rot, expand=True)))) for rot in (0, 90, 180, 270)}
        best = max(vs, key=vs.get); srt = sorted(vs.values(), reverse=True)
        v["nocls_scores"] = vs; v["nocls_margin"] = round(srt[0] / max(srt[1], 1), 2); v["rotation_ccw_before_pass2"] = v["rotation_ccw"]
        if vs[best] > vs[v["rotation_ccw"]] * 1.15 + 2 and best != v["rotation_ccw"]:
            wk = im.rotate(best, expand=True); wk.thumbnail((MAXW, MAXW)); wk.save(POOL / "images" / f"{d}.jpg", quality=92)
            ws = ocr(o_full, np.array(wk))
            (POOL / "ocr" / f"{d}.json").write_text(json.dumps({"document_id": d, "image": f"interim/v2_pool/images/{d}.jpg", "width": wk.width, "height": wk.height, "ocr_engine": "rapidocr-onnx-gpu", "words": ws}, ensure_ascii=False))
            v.update(rotation_ccw=best, n_words=len(ws), mean_conf=round(float(np.mean([w["conf"] for w in ws])), 3) if ws else 0, size=[wk.width, wk.height]); changed.append((d, v["rotation_ccw_before_pass2"], best))
        if i % 20 == 0: print(i, len(changed), flush=True); (POOL / "orientation.json").write_text(json.dumps(ori))
    (POOL / "orientation.json").write_text(json.dumps(ori, indent=0)); print("changed", len(changed), changed)
