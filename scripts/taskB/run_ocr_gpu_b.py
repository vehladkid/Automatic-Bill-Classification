"""OCR for arbitrary pages (GPU RapidOCR, same engine as Task A) -> segments [{text, bbox}]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "taskA"))
import numpy as np
from run_ocr_gpu import get_engine
def ocr_segments(img):
    res, _ = get_engine()(np.asarray(img.convert("RGB"))[:, :, ::-1])
    return [{"text": t, "bbox": [int(min(p[0] for p in b)), int(min(p[1] for p in b)), int(max(p[0] for p in b)), int(max(p[1] for p in b))]} for b, t, c in (res or [])]
