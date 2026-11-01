"""Post-OCR label normalisation. Currently: assign electricity/water/gas to the unlabeled `in_financial` utility bills from
OCR keywords. Only an UNAMBIGUOUS match (>=3 hits for one type, 0 for the others) becomes rule_high; the rest stay unlabeled.
Then applies datasets/metadata/label_overrides.json (human-reviewed fixes). Reads/writes manifest_raw.jsonl in place. Idempotent."""
import json, re
from common import *

RX = {"electricity": r"electric|kwh|kilowatt|power (?:supply|co)|discom|bses|energy charge", "water": r"water|sewer|jal |wssb|bwssb|\bkl\b|hydrant",
      "gas": r"\bgas\b|lpg|cylinder|therm|indane|bharat gas|\bhp gas"}

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(META / "manifest_raw.jsonl")]; n = 0
    for r in rows:
        if r["source_dataset"] != "in_financial" or r["document_type"] != "utility_bill" or not r["ann_path"]: continue
        t = " ".join(w["text"] for w in json.loads((DS / r["ann_path"]).read_text())["words"]).lower()
        h = {c: len(re.findall(rx, t)) for c, rx in RX.items()}; best = max(h, key=h.get)
        if h[best] >= 3 and sum(v > 0 for v in h.values()) == 1:
            r.update(expense_category=best, parent_group=CATEGORIES[best], label_quality="rule_high", label_source=f"OCR keyword rule ({h[best]} hits)"); n += 1
        else: r.update(expense_category="unlabeled", parent_group="unlabeled", label_quality="unlabeled", label_source=f"ambiguous OCR keywords {h}")
    ov = json.loads((META / "label_overrides.json").read_text()) if (META / "label_overrides.json").exists() else {}
    for r in rows:  # human-reviewed corrections win over every rule
        if r["document_id"] in ov:
            c = ov[r["document_id"]]["expense_category"]
            r.update(expense_category=c, parent_group=CATEGORIES[c], label_quality="manual_review", label_source="label_overrides.json: " + ov[r["document_id"]]["reason"][:80])
    with open(META / "manifest_raw.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("utility bills labelled by rule:", n)
