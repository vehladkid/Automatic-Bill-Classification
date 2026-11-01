"""Specificity check of the RULE-BASED employee-document labels (app/doc_profile.py) on the REAL labelled documents we have.
All of these documents are ordinary expense bills (food, fuel, hotel, medical, utilities, travel...), none is a rent/school/80C/80D/loan/EV/Sodexo document - so any
tax-document label fired here is a FALSE POSITIVE. This measures false-positive rate only. RECALL for the new categories CANNOT be measured: the dataset has no labelled
examples of them (datasets/indian/rent is empty). Read-only; writes reports/profile_rules_specificity.json. Uses the annotated OCR words (not a new OCR run)."""
import sys, json, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "app", ROOT / "scripts", ROOT / "scripts" / "taskB"): sys.path.insert(0, str(p))
from doc_profile import classify
import baseline
TA_LABELS = {"bus", "electricity", "flight", "food", "fuel", "gas", "hotel", "internet", "medical", "miscellaneous", "non_expense", "parking", "rent", "train", "transport", "water"}
rows = [json.loads(l) for l in open(ROOT / "datasets/metadata/manifest.jsonl")]
rows = [r for r in rows if not r["synthetic"] and r["expense_category"] in TA_LABELS and r.get("ann_path") and r.get("split") in ("train", "validation", "test")]
res = collections.defaultdict(lambda: collections.Counter()); trig = collections.Counter(); examples = []
for r in rows:
    try: ann = json.loads((ROOT / "datasets" / r["ann_path"]).read_text())
    except Exception: continue
    ws = ann.get("words") or []
    if not ws: continue
    w, b = [x["text"] for x in ws], [x["bbox"] for x in ws]
    text = "\n".join(l["text"] for l in baseline.lines_of(w, b))
    p = classify(text, {}, {"label": r["expense_category"], "confidence": 1.0, "route": "auto"})
    for key in ("ALL", r["split"]):
        c = res[key]; c["n"] += 1
        if p["primary_claim_type"]: c["false_positive_primary"] += 1
        elif p["status"] == "needs_review": c["needs_review"] += 1
    if p["primary_claim_type"]:
        trig[(r["expense_category"], p["primary_claim_type"])] += 1
        if len(examples) < 12: examples.append(dict(doc=r["document_id"], cat=r["expense_category"], claim=p["primary_claim_type"], matched=p["candidates"][0]["matched"][:4]))
out = {k: dict(v) | {"fp_rate_primary": round(v["false_positive_primary"] / v["n"], 4), "needs_review_rate": round(v["needs_review"] / v["n"], 4)} for k, v in res.items()}
out["triggers_by_(expense_category, claim)"] = {f"{a}|{b}": n for (a, b), n in trig.most_common(20)}; out["examples"] = examples
out["caveat"] = "specificity only; recall for the new categories is unmeasurable (no labelled examples exist). Rules were written from domain knowledge, not tuned on these documents."
(ROOT / "reports/profile_rules_specificity.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
