"""Phase 4: Task A (16-class expense category) on REAL validation documents, with the OCR text the app would really produce.
Inputs compared (same documents, same frozen Task A models, same scorer):
  stored      the OCR words stored with the dataset (what docs/TASK_A_REPORT.md was computed on; SROIE/CORD = ground-truth text)
  live1600    RapidOCR as the app runs it today (max_side_len=1600)
  live2400    RapidOCR max_side_len=2400 (candidate from Phase 2)
  live2400r   live2400 + glued date/time repair (app/ocr_text.py)
Reports accuracy, macro-F1 (classes present), per-class P/R/F1, confusion pairs, routing at tau, by source. Validation split only; test is NOT used.
Usage: .venv/bin/python reports/extraction_audit/taskA_live_val.py [--split val]"""
import sys, json, argparse, time, tempfile, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/"app", ROOT/"scripts", ROOT/"scripts"/"taskA", ROOT/"scripts"/"taskB"): sys.path.insert(0, str(p))
import numpy as np
import common_a as CA
import service as SV
import ocr_text
import importlib.util
HERE = ROOT/"reports/extraction_audit"

def metrics(y, p, conf, tau, srcs):
    yi, pi = [CA.L2I[a] for a in y], [CA.L2I[a] for a in p]; acc = float(np.mean([a == b for a, b in zip(y, p)]))
    pc = CA.per_class(yi, pi); auto = [c >= tau for c in conf]
    conf_pairs = collections.Counter((a, b) for a, b in zip(y, p) if a != b)
    by_src = {s: dict(n=sum(1 for x in srcs if x == s), acc=round(float(np.mean([a == b for a, b, x in zip(y, p, srcs) if x == s])), 4)) for s in sorted(set(srcs))}
    return dict(n=len(y), accuracy=round(acc, 4), macro_f1=round(CA.macro_f1(yi, pi, sorted(set(yi))), 4), per_class=pc, top_confusions=[(a, b, n) for (a, b), n in conf_pairs.most_common(10)],
                routing=dict(tau=tau, auto_coverage=round(float(np.mean(auto)), 4), auto_precision=round(float(np.mean([a == b for a, b, k in zip(y, p, auto) if k])), 4) if any(auto) else None), by_source=by_src)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); a = ap.parse_args()
    rows = [r for r in CA.load(a.split) if not r["synthetic"]]; svc = SV.BillService().load(); P, tau = svc.A, svc.tauA; print(len(rows), "real docs; tau", tau, svc.errors, flush=True)
    from run_ocr import _gpu_libs; _gpu_libs()
    from rapidocr_onnxruntime import RapidOCR
    eng24 = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, intra_op_num_threads=4, inter_op_num_threads=1, max_side_len=2400)
    tmp = Path(tempfile.mkdtemp(prefix="taA_")); cache = HERE/f"taskA_live_ocr_{a.split}.json"; ocr = json.loads(cache.read_text()) if cache.exists() else {}
    t0 = time.time()
    for i, r in enumerate(rows):
        if r["id"] in ocr: continue
        try:
            im = SV._load_image(r["image"]); pth = tmp/"p.png"; im.save(pth); o = {}
            for nm, e in (("1600", svc.ocr_engine), ("2400", eng24)):
                res, _ = e(str(pth)); o[nm] = [{"text": t, "bbox": [int(min(q[0] for q in b)), int(min(q[1] for q in b)), int(max(q[0] for q in b)), int(max(q[1] for q in b))], "conf": round(float(c), 3)} for b, t, c in (res or [])]
            ocr[r["id"]] = o
        except Exception as ex: ocr[r["id"]] = {"error": f"{type(ex).__name__}: {ex}"}
        if (i + 1) % 50 == 0: print(f"  OCR {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    cache.write_text(json.dumps(ocr)); del eng24; import gc, torch; gc.collect(); torch.cuda.empty_cache()      # a second OCR engine + Task A/B models do not fit on the 6 GB GPU together
    variants = {"stored": None, "live1600": ("1600", False), "live2400": ("2400", False), "live2400r": ("2400", True), "live1600r": ("1600", True)}; out = {}; preds = {}
    for name, cfgv in variants.items():
        y, p, conf, srcs, pr = [], [], [], [], []
        for r in rows:
            try:
                if cfgv is None: ws = CA.words(r)
                else:
                    o = ocr[r["id"]]
                    if "error" in o: raise RuntimeError(o["error"])
                    segs = ocr_text.repair_segments(o[cfgv[0]]) if cfgv[1] else o[cfgv[0]]; ws = [(s["text"], s["bbox"]) for s in segs]
                pth = r["image"] if cfgv is None else None
                if cfgv is not None: im = SV._load_image(r["image"]); pth = tmp/"q.png"; im.save(pth)
                if not ws: res_ = dict(label="non_expense", confidence=0.0, route="hr_review", empty=True)
                else: res_ = P.predict(str(pth), words=ws)
            except Exception as ex: res_ = dict(label="non_expense", confidence=0.0, route="hr_review", error=str(ex)[:80])
            y.append(r["label"]); p.append(res_["label"]); conf.append(res_["confidence"]); srcs.append(r["source"]); pr.append(dict(id=r["id"], label=res_["label"], conf=res_["confidence"], gt=r["label"], err=res_.get("error") or res_.get("empty")))
        first_err = next((x["err"] for x in pr if x["err"]), None)
        if first_err: print("   first error:", first_err, flush=True)
        out[name] = metrics(y, p, conf, tau, srcs); preds[name] = pr
        print(f"{name:10s} n={out[name]['n']} acc {out[name]['accuracy']} macroF1 {out[name]['macro_f1']} auto cov {out[name]['routing']['auto_coverage']} prec {out[name]['routing']['auto_precision']} errors/empty {sum(1 for x in pr if x['err'])}", flush=True)
    (HERE/f"taskA_live_{a.split}_results.json").write_text(json.dumps(out, indent=1, default=float)); (HERE/f"taskA_live_{a.split}_preds.json").write_text(json.dumps(preds))
if __name__ == "__main__": main()
