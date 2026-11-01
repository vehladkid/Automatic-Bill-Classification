"""Phase 3: run the existing local LLM (text mode, the app's prompt builder, the app's RAW RapidOCR segments) on every real VALIDATION document and cache its JSON answer.
Needs `ollama serve` with qwen2.5vl:3b and a free GPU (stop the Streamlit app and every other model process first). Resumable. No scoring here: see analyse_extraction.py.
Usage: .venv/bin/python reports/extraction_audit/llm_live_val.py [--split val] [--limit N]"""
import sys, json, argparse, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/"app", ROOT/"scripts", ROOT/"scripts"/"taskB"): sys.path.insert(0, str(p))
from data import load
from llm import enhance as EN
from llm.client import OllamaClient, LLMUnavailable, LLMBadOutput
from llm.config import settings
HERE = ROOT/"reports/extraction_audit"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); ap.add_argument("--limit", type=int); ap.add_argument("--prompt", default="app"); ap.add_argument("--ocr", default="side2400"); ap.add_argument("--repair", action="store_true"); a = ap.parse_args()
    cfg = settings() | {"mode": "always", "use_image": False}; cl = OllamaClient(cfg); ok, names = cl.available()
    if not ok: raise SystemExit(f"model {cfg['model']!r} not available (have {names})")
    docs = [d for d in load(a.split) if not d["synthetic"]]
    if a.limit: docs = docs[:a.limit]
    live = json.loads((HERE/(f"live_ocr_{a.split}.json" if a.ocr == "base" else f"live_ocr_{a.split}__{a.ocr}.json")).read_text())
    if a.repair:
        import ocr_text; live = {k: dict(v, segs=ocr_text.repair_segments(v["segs"])) for k, v in live.items()}
    cache_f = HERE/f"llm_live_{a.split}_{a.prompt}_{a.ocr}{'_repair' if a.repair else ''}.jsonl"; cache = {json.loads(l)["id"] for l in open(cache_f)} if cache_f.exists() else set(); t0 = time.time(); fails = 0
    for i, d in enumerate(docs):
        did = d["document_id"]
        if did in cache: continue
        segs, size = live[did]["segs"], tuple(live[did]["size"])
        if not segs: js, secs, err = None, None, "no_ocr_text"
        else:
            try: js, raw, secs = EN.run_llm(cl, segs, size, None); err = None
            except (LLMUnavailable, LLMBadOutput) as e: js, secs, err = None, None, type(e).__name__ + ": " + str(e)[:120]; fails += 1
        if err and err.startswith("LLMUnavailable") and fails >= 5: raise SystemExit("5 transport failures: stopping, nothing is fabricated")
        with open(cache_f, "a") as f: f.write(json.dumps(dict(id=did, js=js, secs=secs, err=err, stats=dict(cl.last_stats))) + "\n")
        if (i + 1) % 10 == 0: print(f"  {i + 1}/{len(docs)} {time.time() - t0:.0f}s fails={fails}", flush=True)
    print("done", len(docs), "docs", f"{time.time() - t0:.0f}s")
if __name__ == "__main__": main()
