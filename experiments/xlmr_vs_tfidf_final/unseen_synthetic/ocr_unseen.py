"""RapidOCR (same models, max_side_len=1600, word order as stored by the project's OCR step) over the unseen synthetic images -> ocr.jsonl. CPU on purpose (GPU is busy training)."""
import json, sys, time
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
HERE = Path(__file__).resolve().parent; eng = RapidOCR(intra_op_num_threads=3, inter_op_num_threads=1, max_side_len=1600)
rows = [json.loads(l) for l in open(HERE / "manifest.jsonl")]; done = {json.loads(l)["id"] for l in open(HERE / "ocr.jsonl")} if (HERE / "ocr.jsonl").exists() else set(); t0 = time.time()
with open(HERE / "ocr.jsonl", "a") as out:
    for i, r in enumerate(rows):
        if r["id"] in done: continue
        res, _ = eng(str(HERE / r["file"])); words = [t for _, t, _ in (res or [])]
        out.write(json.dumps({"id": r["id"], "text": " ".join(words), "n_words": len(words)}) + "\n"); out.flush()
        if i % 50 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
print("OCR done", time.time() - t0)
