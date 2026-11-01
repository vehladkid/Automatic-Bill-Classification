"""Run the full predict.py pipeline (image/PDF -> GPU OCR -> models -> hybrid -> gating) on UNSEEN real documents and compare with GT where it exists.
A: held-out TEST documents (never used for training or any tuning). B: user-collected Indian bills in datasets/indian/ (never part of Task B at all; no GT -> verified by eye).
Writes reports/taskB/inference_check.json (+ prints a compact table)."""
import json, os, sys, random
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from schema import *
from data import load
import predict as P
if __name__ == "__main__":
    te = [d for d in load("test") if not d["synthetic"]]; random.seed(11); man = {json.loads(l)["document_id"]: json.loads(l) for l in open(META / "manifest.jsonl")}
    picks = []
    for src, cond, n in (("receiptbench", lambda d: d["country"] == "IN" and "gstin" in d["fields"], 3), ("receiptbench", lambda d: d["country"] == "IN" and "gstin" not in d["fields"] and d["expense_category"] in ("transport", "food"), 1), ("receiptbench", lambda d: d["country"] != "IN" and d["expense_category"] == "hotel", 1),
                         ("sroie", lambda d: True, 1), ("cord", lambda d: True, 1), ("invoices_pt", lambda d: True, 1)):
        c = [d for d in te if d["source_dataset"] == src and cond(d)]; random.shuffle(c); picks += c[:n]
    unseen = []
    for cat in ("food", "fuel", "electricity", "water", "gas"):
        fs = sorted((DS / "indian" / cat).glob("*")); random.shuffle(fs); unseen += [("indian/" + cat, f) for f in fs[:1]]
    ex = P.Extractor(use_layoutlm=True); out = []
    for d in picks:
        p = DS / man[d["document_id"]]["file_path"]; did, img, seg = P.load_input(str(p))
        from words import split_words, reading_order
        w = reading_order(split_words(seg)); doc = dict(document_id=d["document_id"], words=[x["text"] for x in w], boxes=[x["bbox"] for x in w], width=img.width, height=img.height, country="?", supported_fields=list(FIELDS), fields={}, labels=[])
        pred = ex.run(doc, img); gt = {f: v["norm"] for f, v in d["fields"].items()}
        rows = []
        for f in sorted(set(pred) | set(gt), key=lambda f: list(FIELDS).index(f)):
            if f not in d["supported_fields"]: continue
            pv = pred.get(f); ok = (pv is not None and f in gt and values_match(f, normalize(f, pv["value"] if FIELDS[f] in ("text", "place") else pv["raw_text"], "default", True) if True else None, gt[f])) if pv else None
            rows.append(dict(field=f, predicted=None if pv is None else pv["value"], confidence=None if pv is None or pv["confidence"] is None else round(pv["confidence"], 2), gt=gt.get(f), correct=ok))
        out.append(dict(kind="held-out test (GT)", document_id=d["document_id"], country=d["country"], category=d["expense_category"], file=str(p.relative_to(DS)), rows=rows))
    for tag, f in unseen:
        did, img, seg = P.load_input(str(f)); from words import split_words, reading_order
        w = reading_order(split_words(seg)); doc = dict(document_id=f.name, words=[x["text"] for x in w], boxes=[x["bbox"] for x in w], width=img.width, height=img.height, country="?", supported_fields=list(FIELDS), fields={}, labels=[])
        pred = ex.run(doc, img); out.append(dict(kind="unseen user Indian bill (no GT)", document_id=f.name, country="IN", category=tag.split("/")[1], file=str(f.relative_to(DS)), rows=[dict(field=k, predicted=v["value"], confidence=None if v["confidence"] is None else round(v["confidence"], 2), source=v["source"]) for k, v in pred.items()]))
    json.dump(out, open(ROOT / "reports/taskB/inference_check.json", "w"), indent=1, ensure_ascii=False, default=float)
    for o in out:
        print("==", o["kind"], o["document_id"][:42], o["category"], o["country"]); [print("   ", r["field"].ljust(16), str(r["predicted"])[:32].ljust(32), "conf", r.get("confidence"), "| gt:", str(r.get("gt"))[:30], "| ok:", r.get("correct")) for r in o["rows"]]
