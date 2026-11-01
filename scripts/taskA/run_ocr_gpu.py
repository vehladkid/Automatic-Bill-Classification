"""Shared GPU RapidOCR engine for predict.py (same setup as scripts/run_ocr.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_e = None
def get_engine():
    global _e
    if _e is None:
        from run_ocr import _gpu_libs; _gpu_libs()
        from rapidocr_onnxruntime import RapidOCR
        _e = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, intra_op_num_threads=4, inter_op_num_threads=1, max_side_len=1600)
    return _e
