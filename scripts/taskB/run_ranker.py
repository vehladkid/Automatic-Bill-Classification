"""Train ranker per field on train; tune tau + isotonic calibration on val; evaluate val/test; save models/taskB/ranker.pkl and reports/taskB/ranker.json"""
import json, pickle, time
import numpy as np
from sklearn.isotonic import IsotonicRegression
from schema import *
from data import load
from evaluate import *
import ranker as R

def apply_tau(preds, taus):
    return {k: {f: p for f, p in v.items() if p["score"] >= taus.get(f, 0.0)} for k, v in preds.items()}

if __name__ == "__main__":
    t0 = time.time(); tr, va, te = load("train"), load("val"), load("test")
    CD = {}; [CD.update(R.build_cands(x)) for x in (tr, va, te)]; print("candidates built", round(time.time() - t0), "s", flush=True)
    models = {}
    for f in R.TYPE_OF:
        m = R.train_field(tr, CD, f)
        if m is not None: models[f] = m
        print("trained", f, "yes" if m is not None else "SKIP (too few positives)", round(time.time() - t0), "s", flush=True)
    raw = {sp: {d["document_id"]: R.predict_doc(models, d, CD) for d in docs} for sp, docs in (("val", va), ("test", te))}
    # ---- threshold + calibration policy (see schema.py: MIN_REAL_VAL / DEFAULT_TAU / MIN_REAL_CAL) ----
    # tau: tuned (max F1) on REAL validation documents only when >= MIN_REAL_VAL real val docs carry the field; otherwise DEFAULT_TAU, flagged low-support. Test is never used.
    taus = {}; iso = {}; tau_info = {}; vreal = [d for d in va if not d["synthetic"]]
    for f in models:
        n_real = score_field(vreal, apply_tau(raw["val"], {f: 0.0}), f)["n_gt"]
        if n_real >= MIN_REAL_VAL:
            best = (-1, 0.0)
            for t in [0.0] + list(np.arange(0.05, 0.96, 0.05)):
                r = score_field(vreal, apply_tau(raw["val"], {f: t}), f); f1 = r["f1"] if r["f1"] is not None else -1
                if f1 > best[0] + 1e-9: best = (f1, float(t))
            taus[f] = best[1]; tau_info[f] = dict(tau=taus[f], source="tuned on real validation", real_val_gt=n_real, low_support=False)
        else:
            taus[f] = DEFAULT_TAU; tau_info[f] = dict(tau=DEFAULT_TAU, source="default (insufficient real validation support)", real_val_gt=n_real, low_support=True)
        def collect(docs):
            xs, ys = [], []
            for d in docs:
                p = raw["val"][d["document_id"]].get(f)
                if p is None or f not in d["supported_fields"]: continue
                g = d["fields"].get(f); xs.append(p["score"]); ys.append(int(g is not None and values_match(f, p["norm"], g["norm"])))
            return xs, ys
        xs, ys = collect(vreal); tau_info[f]["calibration"] = "real validation predictions"
        if len(xs) < MIN_REAL_CAL or not (0 < sum(ys) < len(ys)): xs, ys = collect(va); tau_info[f]["calibration"] = "all validation predictions (incl. synthetic): LOW-SUPPORT"
        if len(xs) >= 30 and 0 < sum(ys) < len(ys): iso[f] = IsotonicRegression(y_min=0.01, y_max=0.995, out_of_bounds="clip").fit(xs, ys)
    for sp in raw:
        for k, v in raw[sp].items():
            for f, p in v.items(): p["confidence"] = float(iso[f].predict([p["score"]])[0]) if f in iso else None
    for sp in raw: json.dump({k: {f: {x: p[x] for x in ("value", "norm", "word_idx", "score", "confidence")} for f, p in v.items()} for k, v in raw[sp].items()}, open(ROOT / f"reports/taskB/ranker_raw_{sp}.json", "w"))   # BEFORE thresholding (for abstention analysis)
    res = {}
    for sp, docs in (("val", va), ("test", te)):
        preds = apply_tau(raw[sp], taus); res[sp] = {"metrics": evaluate(docs, preds), "by_category_real": by_category([d for d in docs if not d["synthetic"]], preds, ["merchant_name", "invoice_date", "total_amount", "gstin", "cgst", "sgst"])}
        json.dump({k: {f: {x: p[x] for x in ("value", "norm", "word_idx", "score", "confidence")} for f, p in v.items()} for k, v in preds.items()}, open(ROOT / f"reports/taskB/ranker_predictions_{sp}.json", "w"))
    res["taus"] = taus; res["tau_info"] = tau_info; json.dump(res, open(ROOT / "reports/taskB/ranker.json", "w"), indent=1)
    pickle.dump({"models": models, "taus": taus, "iso": iso}, open(ROOT / "models/taskB/ranker.pkl", "wb"))
    for sl in ("real", "synthetic", "real_indian"):
        m = res["test"]["metrics"][sl]; print(f"--- RANKER test {sl} micro {micro(m, list(m))}")
        for f in FIELDS:
            if f in m and m[f]["n_gt"] >= 3: print(f"  {f:20s} n={m[f]['n_gt']:4d} P={m[f]['precision']} R={m[f]['recall']} F1={m[f]['f1']}")
