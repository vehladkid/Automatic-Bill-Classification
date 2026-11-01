"""Task B inference utility (research component, NOT a production OCR service).
 python scripts/taskB/predict.py <image | pdf | manifest document_id> [--out result.json] [--no-layoutlm]
 image -> GPU RapidOCR (words+boxes); pdf -> page-1 raster + PDF text layer; document_id -> stored OCR of that dataset document.
 -> baseline + ranker (+ LayoutLMv3 if available); per-field system chosen on validation (models/taskB/hybrid.json).
 Output: structured JSON; absent fields are omitted. 'confidence' is the ISOTONIC-CALIBRATED probability that the value is correct (null when the chosen
 system has no score, e.g. the regex baseline); 'model_score' is the raw model score and is NOT a probability."""
import sys, json, pickle, argparse, os
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schema import *
from words import split_words, reading_order
from data import bbox_of
import baseline, ranker as R, decode_ie
from gating import gate

def typed(f, p):
    t = FIELDS[f]; n = p["norm"]
    return n if t in ("amount", "date", "gstin", "id", "int", "code", "enum") else p["value"]

class Extractor:
    def __init__(self, use_layoutlm=True):
        self.hy = json.loads((ROOT / "models/taskB/hybrid.json").read_text()); rk = pickle.load(open(ROOT / "models/taskB/ranker.pkl", "rb")); self.models, self.rtaus, self.riso = rk["models"], rk["taus"], rk["iso"]
        self.ll = None
        if use_layoutlm and (ROOT / "models/taskB/layoutlmv3_ie.pt").exists() and (ROOT / "models/taskB/layoutlm_calib.pkl").exists():
            import torch, ie_layoutlm as IE
            self.torch, self.IE = torch, IE; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            m = IE.LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(IE.TAGS)); m.load_state_dict(torch.load(ROOT / "models/taskB/layoutlmv3_ie.pt", map_location="cpu")); self.ll = m.to(self.dev).eval()
            c = pickle.load(open(ROOT / "models/taskB/layoutlm_calib.pkl", "rb")); self.ltaus, self.liso = c["taus"], c["iso"]

    def run(self, doc, image=None):
        S = {"baseline": baseline.extract(doc)}
        CD = {doc["document_id"]: R.build_cands([doc])[doc["document_id"]]}; rp = R.predict_doc(self.models, doc, CD)
        S["ranker"] = {f: dict(p, confidence=float(self.riso[f].predict([p["score"]])[0]) if f in self.riso else None) for f, p in rp.items() if p["score"] >= self.rtaus.get(f, 0.0)}
        if self.ll is not None and image is not None:
            d2 = dict(doc, _image=image); P = self.IE.predict_probs(self.ll, [d2], self.dev)[0]; lp = decode_ie.decode(doc, P); S["layoutlm"] = {}
            for f, p in lp.items():
                if p["score"] >= self.ltaus.get(f, 0.0): S["layoutlm"][f] = dict(p, confidence=float(self.liso[f].predict([p["score"]])[0]) if f in self.liso else None)
        out = {}
        for f in FIELDS:
            s = self.hy["choice"].get(f, "baseline")
            if s not in S: s = "ranker" if "ranker" in S and f in S["ranker"] else "baseline"
            p = S[s].get(f)
            if p is None: continue
            out[f] = {"value": typed(f, p), "raw_text": p["value"], "confidence": p.get("confidence"), "model_score": p.get("score"), "source": p.get("source", s), "bbox": p.get("bbox")}
        if "igst" in out and "tax_total" not in out: out["tax_total"] = dict(out["igst"], source="derived(igst)")      # IGST-only bill: the single tax line is the total tax
        keep = gate(doc["words"], out)          # applicability gating: category-specific fields need evidence words in the document
        return keep

def load_input(arg):
    from run_ocr_gpu_b import ocr_segments
    p = Path(arg)
    if p.exists() and p.suffix.lower() == ".pdf":
        sys.path.insert(0, str(ROOT / "scripts")); from pdf_utils import rasterize, words as pw
        img = Image.open(rasterize(p, ROOT / "datasets/interim/taskB_predict" / (p.stem + ".jpg"))).convert("RGB"); seg = pw(p) or ocr_segments(img)
        return p.stem, img, seg
    if p.exists():
        img = Image.open(p).convert("RGB"); return p.stem, img, ocr_segments(img)
    man = {json.loads(l)["document_id"]: json.loads(l) for l in open(META / "manifest.jsonl")}
    if arg in man:
        r = man[arg]; ann = json.loads((DS / r["ann_path"]).read_text()); return arg, Image.open(DS / r["file_path"]).convert("RGB"), ann["words"]
    raise SystemExit(f"not a file or manifest document_id: {arg}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inputs", nargs="+"); ap.add_argument("--out"); ap.add_argument("--no-layoutlm", action="store_true"); a = ap.parse_args()
    ex = Extractor(use_layoutlm=not a.no_layoutlm); res = []
    for x in a.inputs:
        did, img, seg = load_input(x); words = reading_order(split_words(seg))
        doc = dict(document_id=did, words=[w["text"] for w in words], boxes=[w["bbox"] for w in words], width=img.width, height=img.height, country="?", supported_fields=list(FIELDS), fields={}, labels=[])
        res.append({"document_id": did, "n_words": len(words), "fields": ex.run(doc, img),
                    "notes": "confidence = isotonic-calibrated P(correct) on validation; null where the selected system has no calibrated score. Missing fields are omitted, never guessed."})
    s = json.dumps(res[0] if len(res) == 1 else res, indent=1, ensure_ascii=False, default=float)
    if a.out: Path(a.out).write_text(s)
    print(s)
