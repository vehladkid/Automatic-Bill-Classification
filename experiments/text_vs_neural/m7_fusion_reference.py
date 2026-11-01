"""Reference row - the PRODUCTION Task A fusion (TF-IDF + ConvNeXt + LayoutLMv3), rebuilt read-only from the stored validation/test logits with the formula in scripts/taskA/predict.py:49
(z = sum_k w_k * log softmax(logit_k / T_k);  p = softmax(z / T_fused)). Included so the paired bootstrap can say whether the fusion is really better than TF-IDF alone
(the Task A report only gave overlapping confidence intervals). Not a text-only model: it uses the page image and layout."""
import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import *
tr, va, te = splits(); cfg = json.loads((PROD_MODELS / "fusion.json").read_text()); P = np.load(PROD_MODELS / "probs_text.npz")
src = {"text": (np.log(np.clip(P["val"], 1e-6, 1)), np.log(np.clip(P["test"], 1e-6, 1))), "image": tuple(np.load(PROD_MODELS / "logits_image.npz")[k] for k in ("val", "test")), "layoutlm": tuple(np.load(PROD_MODELS / "logits_layoutlm.npz")[k] for k in ("val", "test"))}
def fuse(i):
    z = 0
    for k in cfg["models"]: z = z + cfg["weights"][k] * np.log(softmax(src[k][i] / cfg["temperatures"][k]))
    return z
zv, zt = fuse(0), fuse(1); pt = softmax(zt / cfg["fused_temperature"]); yt = np.array([r["y"] for r in te]); real = np.array([not r["synthetic"] for r in te])
print("check vs Task A report: real test accuracy %.4f (report 0.907) macro-F1 %.4f (report 0.839)" % ((pt.argmax(1)[real] == yt[real]).mean(), cls_metrics(pt[real], yt[real], False)["macro_f1"]))
save_logits("fusion", zv, zt); res = evaluate_model("Calibrated fusion (TF-IDF + ConvNeXt + LayoutLMv3)", zv, zt, va, te)
res.update(config=dict(weights=cfg["weights"], temperatures=cfg["temperatures"], fused_temperature=cfg["fused_temperature"]), train_seconds=None, device="GPU", latency_gpu=None, latency_cpu=None, model_bytes=None, peak_mem_mb=None,
           note="reference only: combines three models; per-document latency = sum of its branches (not re-measured)")
save_json("m7_fusion.json", res); print("calibrated real:", {a: round(res["calibrated"]["real"][a], 4) for a in ("accuracy", "macro_f1", "ece", "nll")})
