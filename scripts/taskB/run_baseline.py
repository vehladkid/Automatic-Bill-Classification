import json
from schema import *
from data import load
from evaluate import *
import baseline
if __name__ == "__main__":
    res = {}
    for sp in ("val", "test"):
        docs = load(sp); preds = {d["document_id"]: baseline.extract(d) for d in docs}
        res[sp] = {"metrics": evaluate(docs, preds), "by_category_real": by_category([d for d in docs if not d["synthetic"]], preds, ["merchant_name", "invoice_date", "total_amount", "gstin", "cgst", "sgst"])}
        json.dump({k: {f: {"value": p["value"], "norm": p["norm"], "word_idx": p["word_idx"]} for f, p in v.items()} for k, v in preds.items()}, open(ROOT / f"reports/taskB/baseline_predictions_{sp}.json", "w"))
    json.dump(res, open(ROOT / "reports/taskB/baseline.json", "w"), indent=1)
    for sl in ("real", "synthetic", "real_indian"):
        m = res["test"]["metrics"][sl]; print(f"--- BASELINE test {sl}  micro {micro(m, list(m))}")
        for f in FIELDS:
            if f in m and m[f]["n_gt"] >= 3: print(f"  {f:20s} n={m[f]['n_gt']:4d} P={m[f]['precision']} R={m[f]['recall']} F1={m[f]['f1']}")
