"""Phase 2 experiments on the REAL VALIDATION split: do cheap, existing-dependency changes to the OCR input improve what the (frozen) models receive?
Variants (all use the SAME RapidOCR weights; no model is changed):
  base            current app behaviour (page <= 3000 px, engine max_side_len=1600)
  side2400/3200   engine max_side_len raised (less down-scaling of large photos)
  up              up-scale pages whose long side < 1400 px by 1.5x (small screenshots)
  +repair         deterministic text repair of glued tokens (app/ocr_text.py) applied to the segments
Per variant: OCR text quality on SROIE/CORD (CER/WER/token recall), field recoverability, and Task B V1 micro-F1 by slice (same scorer). Selection is on validation only.
Usage: .venv/bin/python reports/extraction_audit/ocr_variants.py base side2400 ... [--repair]"""
import sys, json, re, argparse, time, tempfile, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/"app", ROOT/"scripts", ROOT/"scripts"/"taskB", ROOT/"scripts"/"taskA", ROOT/"reports"/"extraction_audit"): sys.path.insert(0, str(p))
import numpy as np
from PIL import Image
from schema import *
from data import load
from evaluate import evaluate, micro
from words import split_words, reading_order
import service as SV
import ocr_live_val as OL
HERE = ROOT/"reports/extraction_audit"; F = OL.F

def engine_for(side):
    from run_ocr import _gpu_libs; _gpu_libs()
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, intra_op_num_threads=4, inter_op_num_threads=1, max_side_len=side)

def ocr_variant(name, docs, svc):
    f = HERE/f"live_ocr_val__{name}.json"
    if name == "base": return json.loads((HERE/"live_ocr_val.json").read_text())
    if f.exists(): return json.loads(f.read_text())
    side = {"side2400": 2400, "side3200": 3200}.get(name, 1600); eng = engine_for(side) if side != 1600 else svc.ocr_engine; out = {}; tmp = Path(tempfile.mkdtemp(prefix="ocrvar_"))
    for d in docs:
        im = SV._load_image(ROOT/"datasets"/d["image"]); sc = 1.0
        if name == "up" and max(im.size) < 1400: sc = 1.5; im = im.resize((round(im.width*sc), round(im.height*sc)), Image.LANCZOS)
        p = tmp/"p.png"; im.save(p); res, _ = eng(str(p))
        out[d["document_id"]] = dict(size=list(im.size), segs=[{"text": t, "bbox": [int(min(q[0] for q in b)/sc), int(min(q[1] for q in b)/sc), int(max(q[0] for q in b)/sc), int(max(q[1] for q in b)/sc)], "conf": round(float(c), 3)} for b, t, c in (res or [])])
        out[d["document_id"]]["size"] = [round(x/sc) for x in im.size]
    f.write_text(json.dumps(out)); return out

def run(name, repair, docs, svc, repair_fn):
    live = ocr_variant(name, docs, svc)
    if repair: live = {k: dict(v, segs=repair_fn(v["segs"])) for k, v in live.items()}
    per = {"sroie": [], "cord": []}; rec = collections.Counter(); pl = {}
    for d in docs:
        did = d["document_id"]; segs = live[did]["segs"]
        if d["source_dataset"] in per: per[d["source_dataset"]].append(OL.text_scores([dict(text=w, bbox=b) for w, b in zip(d["words"], d["boxes"])], segs))
        text = "\n".join(s["text"] for s in segs)
        for f in F:
            if f in d["supported_fields"]:
                r = OL.recoverable(f, d, segs, text)
                if r is not None: rec["n"] += 1; rec["ok"] += bool(r); rec[f + "_n"] += 1; rec[f + "_ok"] += bool(r)
        im = SV._load_image(ROOT/"datasets"/d["image"]).convert("RGB"); ws = reading_order(split_words(segs))
        dl = dict(document_id=did, words=[w["text"] for w in ws], boxes=[w["bbox"] for w in ws], width=im.width, height=im.height, country="?", supported_fields=list(svc.mB.FIELDS), fields={}, labels=[])
        pl[did] = OL.to_pred(svc.B.run(dl, im) if ws else {})
    res = dict(variant=name + ("+repair" if repair else ""), text={s: {k: round(float(np.mean([x[k] for x in v])), 4) for k in ("cer", "wer", "tok_recall", "tok_precision")} for s, v in per.items()}, recoverable=dict(rec))
    sl = {"all": docs, "gt_ocr_source": [d for d in docs if d["source_dataset"] in ("sroie", "cord")], "rapidocr_source": [d for d in docs if d["source_dataset"] in ("receiptbench", "invoices_pt")], "sroie": [d for d in docs if d["source_dataset"] == "sroie"], "cord": [d for d in docs if d["source_dataset"] == "cord"]}
    res["v1"] = {k: micro(evaluate(v, pl, F)["real"], F) for k, v in sl.items()}
    res["v1_per_field"] = {k: {f: r["f1"] for f, r in evaluate(v, pl, F)["real"].items()} for k, v in sl.items() if k in ("sroie", "cord", "rapidocr_source")}
    (HERE/f"v1_preds_val__{res['variant']}.json").write_text(json.dumps(pl, default=float)); return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("variants", nargs="+"); ap.add_argument("--repair", action="store_true"); ap.add_argument("--both", action="store_true"); a = ap.parse_args()
    docs = [d for d in load("val") if not d["synthetic"]]; svc = SV.BillService().load()
    repair_fn = None
    if a.repair or a.both:
        import ocr_text; repair_fn = ocr_text.repair_segments
    allr = []
    for v in a.variants:
        for rp in ((False, True) if a.both else (a.repair,)):
            t0 = time.time(); r = run(v, rp, docs, svc, repair_fn); allr.append(r)
            print(f"{r['variant']:22s} CER sroie {r['text']['sroie']['cer']} cord {r['text']['cord']['cer']} | recov {r['recoverable'].get('ok')}/{r['recoverable'].get('n')} | V1 F1 all {r['v1']['all']['f1']} gt-ocr {r['v1']['gt_ocr_source']['f1']} (sroie {r['v1']['sroie']['f1']} cord {r['v1']['cord']['f1']}) rapid {r['v1']['rapidocr_source']['f1']}  [{time.time()-t0:.0f}s]", flush=True)
    out = HERE/"ocr_variants_results.json"; prev = json.loads(out.read_text()) if out.exists() else {}
    for r in allr: prev[r["variant"]] = r
    out.write_text(json.dumps(prev, indent=1, default=float))
