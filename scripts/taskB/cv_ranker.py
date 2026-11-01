"""Group 5-fold cross-validation of the ranker over ALL real Task B documents (train+val+test pooled; groups = manifest group_id so templates/near-duplicates never straddle folds);
all synthetic documents stay in every training fold. Purpose: stable estimates for Indian-real fields whose test sets are tiny (GSTIN/CGST/SGST).
Thresholds: tau fixed per field from run_ranker (validation). Outputs reports/taskB/cv_ranker.json."""
import json, collections
import numpy as np
from sklearn.model_selection import GroupKFold
from schema import *
from data import load
from evaluate import *
import ranker as R
if __name__ == "__main__":
    allr = [d for s in ("train", "val", "test") for d in load(s)]; man = {}
    for l in open(META / "manifest.jsonl"):
        r = json.loads(l); man[r["document_id"]] = r["group_id"]
    real = [d for d in allr if not d["synthetic"]]; synth = [d for d in allr if d["synthetic"]]
    CD = R.build_cands(allr); taus = json.load(open(ROOT / "reports/taskB/ranker.json"))["taus"]
    groups = [man.get(d["document_id"], d["document_id"]) for d in real]; preds = {}
    for k, (a, b) in enumerate(GroupKFold(5).split(real, groups=groups)):
        trn = [real[i] for i in a] + synth; models = {}
        for f in R.TYPE_OF:
            m = R.train_field(trn, CD, f)
            if m is not None: models[f] = m
        for i in b:
            d = real[i]; preds[d["document_id"]] = {f: p for f, p in R.predict_doc(models, d, CD).items() if p["score"] >= taus.get(f, 0.0)}
        print("fold", k, flush=True)
    res = {sl: evaluate([d for d in real if doc_flags(d)[sl]], preds)[sl] for sl in ("real", "real_indian")}
    json.dump(res, open(ROOT / "reports/taskB/cv_ranker.json", "w"), indent=1)
    for sl in ("real", "real_indian"):
        m = res[sl]; print(sl, micro(m, list(m)))
        for f in ("merchant_name", "invoice_number", "invoice_date", "total_amount", "tax_total", "subtotal", "gstin", "cgst", "sgst", "igst", "check_in", "check_out", "travel_date"):
            if f in m: print(f"  {f:16s} n={m[f]['n_gt']:4d} P={m[f]['precision']} R={m[f]['recall']} F1={m[f]['f1']}")
