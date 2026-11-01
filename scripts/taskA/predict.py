"""Task A inference utility (NOT a serving system): image(s) -> OCR (RapidOCR on GPU) -> text-LR + ConvNeXt + LayoutLMv3 -> calibrated fusion -> label, confidence, route.
route = 'auto' if confidence >= tau (chosen on validation for 95% precision on real documents) else 'hr_review'.
Usage: python predict.py bill1.jpg bill2.png   (prints JSON). Models that are missing from models/taskA/ are skipped and the fusion renormalised."""
import sys, json
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_a import *

def ocr(path):
    from run_ocr_gpu import get_engine
    res, _ = get_engine()(str(path)); ws = [(t, [int(min(p[0] for p in b)), int(min(p[1] for p in b)), int(max(p[0] for p in b)), int(max(p[1] for p in b))]) for b, t, c in (res or [])]
    return ws

class Predictor:
    def __init__(self):
        import torch; self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = json.loads((MODELS / "fusion.json").read_text()); self.models = {}
        names = self.cfg["models"]
        if "text" in names:
            import joblib; self.models["text"] = joblib.load(MODELS / "text_lr.joblib")
        if "image" in names:
            import timm; m = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=len(LABELS)); m.load_state_dict(torch.load(MODELS / "convnext_tiny.pt", map_location="cpu")); self.models["image"] = m.to(self.dev).eval()
        if "layoutlm" in names:
            from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
            self.proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
            m = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(LABELS)); m.load_state_dict(torch.load(MODELS / "layoutlmv3.pt", map_location="cpu")); self.models["layoutlm"] = m.to(self.dev).eval()

    def logits(self, path, ws):
        torch = self.torch; im = Image.open(path).convert("RGB"); W, H = im.size; out = {}
        if "text" in self.models:
            v, m = self.models["text"]; p = np.zeros(len(LABELS)); p[m.classes_] = m.predict_proba(v.transform([" ".join(t for t, _ in ws)]))[0]; out["text"] = np.log(np.clip(p, 1e-6, 1))
        if "image" in self.models:
            from t2_image_convnext import letterbox, MEAN, STD
            s = 768 / max(im.size); small = im.resize((max(1, round(W * s)), max(1, round(H * s)))) if s < 1 else im
            x = torch.from_numpy(((np.asarray(letterbox(small), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None].to(self.dev)
            with torch.no_grad(), torch.autocast(self.dev, dtype=torch.bfloat16): out["image"] = self.models["image"](x).float().cpu().numpy()[0]
        if "layoutlm" in self.models:
            ws2 = sorted(ws, key=lambda t: (t[1][1] // max(H // 120, 8), t[1][0]))[:512] or [("empty", [0, 0, 1, 1])]
            bx = [[int(np.clip(b[0] * 1000 / W, 0, 1000)), int(np.clip(b[1] * 1000 / H, 0, 1000)), int(np.clip(b[2] * 1000 / W, 0, 1000)), int(np.clip(b[3] * 1000 / H, 0, 1000))] for _, b in ws2]
            bx = [[min(a, c), min(b_, d), max(a, c), max(b_, d)] for a, b_, c, d in bx]
            enc = self.proc(im, [t for t, _ in ws2], boxes=bx, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
            with torch.no_grad(), torch.autocast(self.dev, dtype=torch.bfloat16): out["layoutlm"] = self.models["layoutlm"](**{k: v.to(self.dev) for k, v in enc.items()}).logits.float().cpu().numpy()[0]
        return out

    def predict(self, path, words=None):
        ws = words if words is not None else ocr(path); lg = self.logits(path, ws); c = self.cfg; z = 0
        for k, v in lg.items(): z = z + c["weights"][k] * np.log(softmax(v / c["temperatures"][k]))
        p = softmax(z / c["fused_temperature"]); i = int(p.argmax()); top = np.argsort(-p)[:3]
        return {"file": str(path), "label": LABELS[i], "parent_group": CATEGORIES[LABELS[i]], "confidence": round(float(p[i]), 4), "route": "auto" if p[i] >= (c["tau"] or 2) else "hr_review",
                "top3": [(LABELS[j], round(float(p[j]), 4)) for j in top], "ocr_words": len(ws)}

if __name__ == "__main__":
    P = Predictor(); print(json.dumps([P.predict(f) for f in sys.argv[1:]], indent=1))
