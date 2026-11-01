"""Fill `language` for docs where the source did not state it, from OCR/PDF text (langdetect; seed fixed). Marks language_source='langdetect'.
Short/garbled text -> left null (never guess). Runs on manifest_raw.jsonl in place; idempotent."""
import json
from langdetect import detect_langs, DetectorFactory
from common import *
DetectorFactory.seed = 0
if __name__ == "__main__":
    rows = [json.loads(l) for l in open(META / "manifest_raw.jsonl")]; n = 0
    for r in rows:
        if r.get("language") or not r["ann_path"]: continue
        t = " ".join(w["text"] for w in json.loads((DS / r["ann_path"]).read_text()).get("words", []))
        if len([c for c in t if c.isalpha()]) < 40: continue
        try: d = detect_langs(t)[0]
        except Exception: continue
        if d.prob >= 0.8: r["language"], r["language_source"] = d.lang, "langdetect"; n += 1
    with open(META / "manifest_raw.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("language detected for", n)
