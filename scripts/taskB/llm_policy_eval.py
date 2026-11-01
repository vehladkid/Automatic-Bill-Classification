"""Offline evaluation of the document-classification DECISION POLICY (app/doc_profile.py::decide) using the cached LLM answers of llm_bench.py. Read-only; writes reports/taskB/llm/policy_eval_<split>.json.
What can and cannot be measured: every real document in the dataset is an ordinary expense bill (or a non_expense paper), none is a rent/school/80C/80D/loan/EV/Sodexo document, so a tax-document
label on them is a FALSE POSITIVE. Recall for the new categories is unmeasurable (no labelled examples). Task A expense category is not changed by the LLM, so its accuracy/macro-F1 is identical by construction.
Usage: .venv/bin/python scripts/taskB/llm_policy_eval.py --split val --mode text"""
import sys, json, argparse, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "app", ROOT / "scripts", ROOT / "scripts" / "taskB"): sys.path.insert(0, str(p))
from data import load
import llm_eval as LE
from llm import validate as V, enhance as EN
from doc_profile import classify, CFG
TAX = set(CFG["profiles"])
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); ap.add_argument("--mode", default="text"); a = ap.parse_args()
    sp = "test" if a.split == "test" else "val"; docs = [d for d in load(sp) if not d["synthetic"]]
    f = ROOT / f"reports/taskB/llm/bench_cache_{sp}_qwen2.5vl-3b_{a.mode}.jsonl"; cache = {json.loads(l)["id"]: json.loads(l) for l in open(f)}
    docs = [d for d in docs if d["document_id"] in cache]; man = {json.loads(l)["document_id"]: json.loads(l) for l in open(ROOT / "datasets/metadata/manifest.jsonl")}
    out = {"split": sp, "mode": a.mode, "n_docs": len(docs)}; rows = collections.defaultdict(lambda: collections.Counter())
    for d in docs:
        segs = LE.segs_of(d); ocr = V.OCR(segs); text = "\n".join(s["text"] for s in segs); js = cache[d["document_id"]]["js"]; cat = man.get(d["document_id"], {}).get("expense_category")
        ta = {"label": cat if cat not in (None, "unlabeled") else None, "confidence": 1.0, "route": "auto"}
        lc = (js or {}).get("classification") if isinstance(js, dict) else None; llm_class = None
        if isinstance(lc, dict) and lc.get("document_kind") in EN.KINDS:
            ev = str(lc.get("evidence") or ""); g = V.ground("merchant_name", ev, ocr) if len(ev.strip()) >= 4 else {"status": "ungrounded"}
            llm_class = dict(kind=lc["document_kind"], evidence=ev, grounded=g["status"] in ("grounded", "corrected") or lc["document_kind"] in ("expense_bill", "unknown", "other", "non_expense_document"))
        rules = classify(text, {}, ta); both = classify(text, {}, ta, llm_class=llm_class)
        for grp in ("all", "ordinary_bills" if cat != "non_expense" else "non_expense_papers"):
            c = rows[grp]; c["n"] += 1; c["llm_answered"] += js is not None; c["llm_kind_given"] += llm_class is not None
            c["rules_tax_label"] += rules["primary_claim_type"] is not None
            c["llm_tax_kind_raw"] += bool(llm_class and llm_class["kind"] in TAX)
            c["llm_tax_kind_grounded"] += bool(llm_class and llm_class["kind"] in TAX and llm_class["grounded"])
            c["policy_tax_label"] += both["primary_claim_type"] is not None
            c["policy_needs_review"] += both["status"] == "needs_review"
            c["policy_contradiction"] += any("contradiction" in r for r in both["decision"]["reasons"])
            c["policy_llm_only_suggestion"] += any("only the LLM suggests" in r for r in both["decision"]["reasons"])
    for g, c in rows.items():
        n = c["n"]; out[g] = {k: v for k, v in c.items()}; out[g].update({f"{k}_rate": round(c[k] / n, 4) for k in ("rules_tax_label", "llm_tax_kind_raw", "llm_tax_kind_grounded", "policy_tax_label", "policy_needs_review")})
    out["caveat"] = "false-positive rates on ordinary bills only; recall for the new categories is unmeasurable. Task A expense category is untouched by the LLM (identical by construction)."
    (ROOT / f"reports/taskB/llm/policy_eval_{sp}_{a.mode}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
if __name__ == "__main__": main()
