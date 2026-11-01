"""Descriptive diagnostics for the unseen sets (frozen predictions): share of documents predicted non_expense / food per condition, for TF-IDF and the mean over XLM-R seeds. -> outputs/unseen_<set>/extras.json"""
import json, sys, numpy as np
from pathlib import Path
FIN = Path(__file__).resolve().parent
for SET in ("synthetic", "real"):
    O = FIN / "outputs" / f"unseen_{SET}"; m = json.loads((O / "meta.json").read_text()); L = m["labels"]; y = np.array(m["y"]); c = np.array(m["condition"]); tf = np.load(O / "tfidf_probs.npy").argmax(1)
    xs = [np.load(p).argmax(1) for p in sorted(O.glob("xlmr_primary256_s*_logits.npy"))]; out = {}
    for cond in sorted(set(c)):
        k = c == cond; out[cond] = {"n": int(k.sum())}
        for lab in ("non_expense", "food"): i = L.index(lab); out[cond][f"{lab}_pred_share_tfidf"] = float((tf[k] == i).mean()); out[cond][f"{lab}_pred_share_xlmr_mean"] = float(np.mean([(p[k] == i).mean() for p in xs]))
        out[cond]["true_non_expense_share"] = float((y[k] == L.index("non_expense")).mean())
    (O / "extras.json").write_text(json.dumps(out, indent=1)); print(SET, json.dumps(out)[:500])
