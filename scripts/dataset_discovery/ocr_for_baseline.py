"""OCR the first page of every synthetic/generated document that has no OCR text yet (synthgen_v1, fatura, agami) with the project's RapidOCR engine, so that the TF-IDF baseline
reads OCR text for real AND synthetic documents alike (generator ground-truth text is NOT used as model input). Resumable cache: datasets/interim/doctype_v2_1_ocr/<source>.jsonl (doc_id, text). Read-only on the dataset; no network.
Usage: ocr_for_baseline.py [--sources synthgen_v1 fatura agami]"""
import sys, json, argparse, subprocess, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts", "scripts/taskA", "scripts/taskB", "scripts/dataset_discovery"): sys.path.insert(0, str(ROOT / p))
from doc_type_lib import DS, META, read_jsonl
OUT = DS / "interim/doctype_v2_1_ocr"
def page_image(path, tmp):
    p = DS / path
    if p.suffix.lower() == ".pdf":
        subprocess.run(["pdftoppm", "-r", "110", "-f", "1", "-l", "1", "-png", "-singlefile", str(p), tmp + "/p"], capture_output=True, timeout=120); return tmp + "/p.png"
    return str(p)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sources", nargs="+", default=["synthgen_v1", "fatura", "agami"]); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    from run_ocr_gpu import get_engine
    eng = get_engine(); rows = [r for r in read_jsonl(META / "doc_type_manifest_v2.jsonl") if r["source"] in a.sources]
    for src in a.sources:
        f = OUT / f"{src}.jsonl"; done = {json.loads(l)["doc_id"] for l in open(f)} if f.exists() else set(); todo = [r for r in rows if r["source"] == src and r["doc_id"] not in done]; t0 = time.time()
        with open(f, "a") as w, tempfile.TemporaryDirectory() as tmp:
            for i, r in enumerate(todo):
                try:
                    out, _ = eng(page_image(r["pages"][0] if r["pages"] else r["file_path"], tmp)); txt = "\n".join(x[1] for x in (out or []))
                except Exception as e: txt = ""; print("ERR", r["doc_id"], e, flush=True)
                w.write(json.dumps(dict(doc_id=r["doc_id"], text=txt), ensure_ascii=False) + "\n"); w.flush()
                if i % 100 == 0: print(src, i, len(todo), f"{time.time() - t0:.0f}s", flush=True)
        print(src, "done", flush=True)
