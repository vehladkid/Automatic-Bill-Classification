"""OCR (RapidOCR/ONNX on the NVIDIA dGPU, ~0.6 s/img vs 2-3 s CPU) for real images that have no ground-truth OCR (manual Indian + Kaggle grocery).
Writes words+boxes into the same annotation file layout as the GT sources, tagged ocr_source='rapidocr'.
Images are untouched. Resumable: skips docs whose annotation exists."""
import json
from multiprocessing import Pool
from common import *

def _gpu_libs():  # pip-installed CUDA 12 libs (onnxruntime-gpu==1.22.0 + nvidia-*-cu12) are not on the loader path by default
    import ctypes, os, site
    sp = site.getsitepackages()[0]
    for lib in ["cuda_runtime/lib/libcudart.so.12", "cublas/lib/libcublas.so.12", "cublas/lib/libcublasLt.so.12", "cufft/lib/libcufft.so.11", "curand/lib/libcurand.so.10",
                "cuda_nvrtc/lib/libnvrtc.so.12", "nvjitlink/lib/libnvJitLink.so.12", "cudnn/lib/libcudnn.so.9"]:
        try: ctypes.CDLL(os.path.join(sp, "nvidia", lib), mode=ctypes.RTLD_GLOBAL)
        except OSError: pass

def work(r):
    _gpu_libs()
    from rapidocr_onnxruntime import RapidOCR
    global _o
    try: _o
    except NameError: _o = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, intra_op_num_threads=4, inter_op_num_threads=1, max_side_len=1600)  # CPU fallback automatic if CUDA missing
    try: res, _ = _o(str(DS / r["file_path"]))
    except Exception:  # GPU OOM on a huge page -> retry that one image on CPU
        global _cpu
        try: _cpu
        except NameError: _cpu = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1, max_side_len=1600)
        res, _ = _cpu(str(DS / r["file_path"]))
    words = [{"text": t, "bbox": [int(min(p[0] for p in b)), int(min(p[1] for p in b)), int(max(p[0] for p in b)), int(max(p[1] for p in b))], "conf": round(float(c), 3)}
             for b, t, c in (res or [])]
    a = INTERIM / "annotations" / f"{r['document_id']}.json"
    old = json.loads(a.read_text()) if a.exists() else {}  # keep source field annotations (ReceiptBench / PT json), add OCR words
    dump(a, {"words": words, "fields": old.get("fields", {}), "ocr_source": "rapidocr"})
    return r["document_id"], len(words)

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(META / "manifest_raw.jsonl")]
    todo = [r for r in rows if r["image_ok"] and not r["ocr_available"] and not r["synthetic"]]
    def has_words(r):
        a = INTERIM / "annotations" / f"{r['document_id']}.json"
        return a.exists() and bool(json.loads(a.read_text()).get("words"))
    todo = [r for r in todo if not has_words(r)]
    print("to OCR:", len(todo))
    with Pool(2) as p:  # 2 procs share the 6 GB dGPU; CPU-side pre/post-processing is the bottleneck
        for i, (d, n) in enumerate(p.imap_unordered(work, todo)):
            if i % 25 == 0: print(i, d, n, flush=True)
    for r in rows:
        a = INTERIM / "annotations" / f"{r['document_id']}.json"
        if a.exists() and not r["ocr_available"]:
            j = json.loads(a.read_text()); r.update(ocr_available=bool(j["words"]), ocr_source=j["ocr_source"] if j["words"] else None, ann_path=f"interim/annotations/{r['document_id']}.json")
    with open(META / "manifest_raw.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
