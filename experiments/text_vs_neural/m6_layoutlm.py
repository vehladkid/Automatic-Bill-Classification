"""Model 6 - LayoutLMv3-base (existing, PRODUCTION checkpoint; read-only). Uses OCR words + 2-D boxes + the page image (NOT text-only).
Predictions = the stored validation/test logits models/taskA/logits_layoutlm.npz (same split/row order as common_a.load), so no re-training and no production file is touched.
Latency is measured by re-running the checkpoint on 40 cached test pages (GPU), same forward code as scripts/taskA/predict.py."""
import sys, json, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "taskA"))
from exp_common import *
from common_a import words as row_words
import torch
from PIL import Image

tr, va, te = splits(); z = np.load(PROD_MODELS / "logits_layoutlm.npz"); lv, lt = z["val"], z["test"]; assert len(lv) == len(va) and len(lt) == len(te)
res = evaluate_model("LayoutLMv3-base", lv, lt, va, te)
# --- latency / memory (GPU), end-to-end per page = processor (tokenise + resize) + forward
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
dev = "cuda"; proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
m = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=NC); m.load_state_dict(torch.load(PROD_MODELS / "layoutlmv3.pt", map_location="cpu")); m.to(dev).eval()
torch.cuda.reset_peak_memory_stats(); docs = []
for r in te[:40]:
    im = Image.open(r["cache"]).convert("RGB"); W, H = im.size; ws = row_words(r)[:512] or [("empty", [0, 0, 1, 1])]
    sx, sy = W / max(r["w"], 1), H / max(r["h"], 1)          # boxes are in original-image pixels; the cached page is resized
    bx = [[int(np.clip(b[0] * sx * 1000 / W, 0, 1000)), int(np.clip(b[1] * sy * 1000 / H, 0, 1000)), int(np.clip(b[2] * sx * 1000 / W, 0, 1000)), int(np.clip(b[3] * sy * 1000 / H, 0, 1000))] for _, b in ws]
    bx = [[min(a, c), min(b_, d), max(a, c), max(b_, d)] for a, b_, c, d in bx]; docs.append((im, [t for t, _ in ws], bx))
k = [0]
def fwd_only(enc):
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16): m(**enc).logits.float().cpu()
def one():
    im, t, bx = docs[k[0] % 40]; k[0] += 1; enc = proc(im, t, boxes=bx, truncation=True, max_length=512, padding="max_length", return_tensors="pt"); fwd_only({a: v.to(dev) for a, v in enc.items()})
lat = time_fn(one, 5, 80, torch.cuda.synchronize); peak = torch.cuda.max_memory_allocated() / 2**20
ref = json.loads((ROOT / "reports/taskA/layoutlmv3.json").read_text())
res.update(config=ref["config"], train_seconds=ref["log"][-1]["sec"], device="GPU", latency_gpu=lat, latency_cpu=None, model_bytes=(PROD_MODELS / "layoutlmv3.pt").stat().st_size, peak_mem_mb=peak,
           note="logits from the production checkpoint (trained 6 epochs, max_len 448, best epoch by validation); inference here uses 512 tokens as in scripts/taskA/predict.py; train time = last 'sec' in reports/taskA/layoutlmv3.json")
save_logits("layoutlmv3", lv, lt); save_json("m6_layoutlmv3.json", res)
print("real test raw:", {a: round(res["raw"]["real"][a], 4) for a in ("accuracy", "macro_f1", "ece", "nll")}, "| calibrated:", {a: round(res["calibrated"]["real"][a], 4) for a in ("accuracy", "macro_f1", "ece", "nll")}, "| latency", lat, "| peak MB", round(peak))
