"""Phase 4: can the existing local LLM improve the expense category WITHOUT changing any weights? Zero-shot, text only, category definitions copied from docs/TAXONOMY.md (not tuned on errors).
Evaluated on REAL VALIDATION documents where Task A is not confident (fused confidence < 0.90: the only place an assist could help), plus a control sample of confident documents
(to measure regression risk). Compares: Task A top-1, LLM alone, 'LLM overrides Task A', 'LLM breaks ties inside Task A's top-3', 'LLM only if it agrees with Task A top-3'.
Needs `ollama serve`; GPU must be free of other model processes. Validation only; the test split is not used."""
import sys, json, time, random, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/"app", ROOT/"scripts", ROOT/"scripts"/"taskA"): sys.path.insert(0, str(p))
import common_a as CA
from llm.client import OllamaClient, LLMUnavailable, LLMBadOutput
from llm.config import settings
import ocr_text
HERE = ROOT/"reports/extraction_audit"
DEFS = {"flight": "air tickets / boarding passes / airline invoices", "train": "railway tickets (e.g. IRCTC, long-distance rail)", "bus": "bus tickets (intercity / operator)", "transport": "cab, auto, taxi, ride-hailing AND local transit tickets (metro, subway, tram)",
        "hotel": "hotel / guest-house / lodging invoices", "food": "restaurants, meals, catering", "electricity": "electricity bill", "water": "water bill", "gas": "piped / cylinder gas bill", "internet": "internet, telecom, postpaid mobile bill",
        "fuel": "petrol / diesel / EV charging", "parking": "parking receipt", "medical": "hospital, pharmacy/chemist, diagnostics", "rent": "rent receipt / rental invoice", "miscellaneous": "office supplies, equipment, groceries, subscriptions, other retail",
        "non_expense": "letters, forms, memos, anything that is not a bill or receipt"}
SYSTEM = "You classify OCR text of a document into ONE expense category for an employee expense claim. Return JSON only. Use only what the text shows."
USER = ("Categories:\n{defs}\n\nReturn {{\"category\": <one of {labels}>, \"evidence\": \"exact words from the text that justify it\"}}.\n\nOCR text:\n{text}")

def main():
    cfg = settings() | {"mode": "always", "use_image": False}; cl = OllamaClient(cfg); ok, names = cl.available()
    if not ok: raise SystemExit("LLM model not available")
    rows = {r["id"]: r for r in CA.load("val") if not r["synthetic"]}; preds = {x["id"]: x for x in json.loads((HERE/"taskA_live_val_preds.json").read_text())["live2400r"]}
    ocr = json.loads((HERE/"taskA_live_ocr_val.json").read_text()); top3 = {}
    hard = [i for i, p in preds.items() if p["conf"] < 0.90]; easy = sorted(i for i, p in preds.items() if p["conf"] >= 0.90); random.Random(0).shuffle(easy); ctrl = easy[:40]
    ids = hard + ctrl; cache_f = HERE/"taskA_llm_val.jsonl"; done = {json.loads(l)["id"]: json.loads(l) for l in open(cache_f)} if cache_f.exists() else {}; t0 = time.time()
    for k, i in enumerate(ids):
        if i in done: continue
        segs = ocr_text.repair_segments(ocr[i]["2400"]); text = "\n".join(s["text"] for s in segs)[:6000]
        try:
            js, raw, secs = cl.chat_json(SYSTEM, USER.format(defs="\n".join(f"- {k_}: {v}" for k_, v in DEFS.items()), labels=json.dumps(CA.LABELS), text=text)); cat = js.get("category") if isinstance(js, dict) else None; err = None
        except (LLMUnavailable, LLMBadOutput) as e: js, cat, secs, err = None, None, None, type(e).__name__
        done[i] = dict(id=i, cat=cat, err=err, secs=secs, evidence=(js or {}).get("evidence") if isinstance(js, dict) else None)
        with open(cache_f, "a") as f: f.write(json.dumps(done[i]) + "\n")
        if (k + 1) % 10 == 0: print(f"  {k + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    print("done")
if __name__ == "__main__": main()
