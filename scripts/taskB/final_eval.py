"""Compare baseline / ranker / LayoutLMv3, build the per-field HYBRID (system chosen on validation), calibrate confidence, analyse errors.
Test is scored once per system here. Outputs reports/taskB/final_results.json, final_results.md, errors.json, models/taskB/hybrid.json."""
import json, pickle, collections, re
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from schema import *
from data import load, bbox_of
from evaluate import *
import baseline, decode_ie
from gating import gate, EVID
REP = ROOT / "reports/taskB"; MOD = ROOT / "models/taskB"
CRIT = ["merchant_name", "invoice_date", "total_amount"]
def apply_tau(preds, taus): return {k: {f: p for f, p in v.items() if p["score"] is None or p["score"] >= taus.get(f, 0.0)} for k, v in preds.items()}

def load_ranker(sp, docs):
    raw = json.load(open(REP / f"ranker_predictions_{sp}.json")); byid = {d["document_id"]: d for d in docs}
    return {k: {f: dict(p, bbox=bbox_of(byid[k], p["word_idx"]) if p["word_idx"] else None, source="ranker") for f, p in v.items()} for k, v in raw.items()}

def layoutlm_preds(docs, probs, sp):
    return {d["document_id"]: decode_ie.decode(d, probs[sp][d["document_id"]]) if d["document_id"] in probs[sp] else {} for d in docs}

def calibrate(va, pv_raw, f, info):
    def collect(docs):
        xs, ys = [], []
        for d in docs:
            p = pv_raw[d["document_id"]].get(f)
            if p is None or f not in d["supported_fields"] or p["score"] is None: continue
            g = d["fields"].get(f); xs.append(p["score"]); ys.append(int(g is not None and values_match(f, p["norm"], g["norm"])))
        return xs, ys
    xs, ys = collect([d for d in va if not d["synthetic"]]); info["calibration"] = "real validation predictions"
    if len(xs) < MIN_REAL_CAL or not (0 < sum(ys) < len(ys)): xs, ys = collect(va); info["calibration"] = "all validation predictions (incl. synthetic): LOW-SUPPORT"
    return IsotonicRegression(y_min=0.01, y_max=0.995, out_of_bounds="clip").fit(xs, ys) if len(xs) >= 30 and 0 < sum(ys) < len(ys) else None

def tune_tau(va, raw, f):
    """tuned on REAL validation docs only when >= MIN_REAL_VAL of them carry the field, else DEFAULT_TAU (flagged). Test never used."""
    vreal = [d for d in va if not d["synthetic"]]; n = score_field(vreal, raw, f)["n_gt"]
    if n < MIN_REAL_VAL: return DEFAULT_TAU, dict(tau=DEFAULT_TAU, source="default (insufficient real validation support)", real_val_gt=n, low_support=True)
    best = (-1, 0.0)
    for t in [0.0] + list(np.arange(0.05, 0.96, 0.05)):
        r = score_field(vreal, apply_tau(raw, {f: t}), f); f1 = -1 if r["f1"] is None else r["f1"]
        if f1 > best[0] + 1e-9: best = (f1, float(t))
    return best[1], dict(tau=best[1], source="tuned on real validation", real_val_gt=n, low_support=False)

def correctness(docs, preds, fields):
    rows = []
    for d in docs:
        for f in fields:
            if f not in d["supported_fields"]: continue
            p = (preds.get(d["document_id"]) or {}).get(f)
            if p is None: continue
            g = d["fields"].get(f); rows.append((f, p.get("confidence"), p.get("score"), int(g is not None and values_match(f, p["norm"], g["norm"])), d))
    return rows

def calib_report(rows, key=1):
    r = [(x[key], x[3]) for x in rows if x[key] is not None]
    if len(r) < 20: return None
    c = np.array([a for a, _ in r]); y = np.array([b for _, b in r]); e = 0.0
    for lo in np.linspace(0, 1, 10, endpoint=False):
        m = (c > lo) & (c <= lo + 0.1)
        if m.any(): e += m.mean() * abs(y[m].mean() - c[m].mean())
    out = {"n": len(r), "accuracy_of_predictions": round(float(y.mean()), 4), "ece": round(float(e), 4), "auroc": round(float(roc_auc_score(y, c)), 4) if 0 < y.sum() < len(y) else None, "curve": []}
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        m = c >= t
        if m.sum(): out["curve"].append(dict(conf_ge=t, coverage=round(float(m.mean()), 3), precision=round(float(y[m].mean()), 3)))
    return out

PHONE = re.compile(r"^\d{10,12}$")
def categorize_errors(docs, preds, raw, fields):
    """real test documents only. each (doc, field) error -> ONE category. raw = chosen system's predictions BEFORE threshold/gating (to detect incorrect abstention)."""
    cats = collections.defaultdict(list); per_field = collections.defaultdict(collections.Counter)
    for d in docs:
        if d["synthetic"]: continue
        gtall = {k: v for k, v in d["fields"].items()}
        for f in fields:
            if f not in d["supported_fields"]: continue
            g = d["fields"].get(f); p = (preds.get(d["document_id"]) or {}).get(f); r = (raw.get(d["document_id"]) or {}).get(f)
            gn = g["norm"] if g else None; pn = p["norm"] if p else None
            if gn is None and pn is None: continue
            if gn is not None and pn is not None and values_match(f, pn, gn): continue
            T = FIELDS[f]; ctx = dict(doc=d["document_id"], field=f, gt=str(gn)[:40], pred=str(pn)[:40], country=d["country"], category=d["expense_category"])
            def other_field_match():
                for k, v in gtall.items():
                    if k == f or v["norm"] is None: continue
                    if FIELDS[k] == T and (values_match(f, pn, v["norm"], soft=T in ("text", "place"))): return k
                return None
            if gn is not None and pn is None:
                if not g["aligned"]: c = "GT not locatable in OCR text (OCR error, unusual format, or value not printed on the page)"
                elif r is not None and values_match(f, r["norm"], gn): c = "incorrect abstention: a correct value was produced but removed by the threshold/gating"
                elif r is not None: c = "missed (only a wrong value scored above the floor, and it was abstained)"
                else: c = "missed: no candidate / span produced"
            elif gn is None:
                oth = other_field_match()
                c = "false positive: GSTIN-field output on a document without GSTIN annotation" if f == "gstin" else (f"false positive: value belongs to {oth}" if oth else "false positive (field not annotated / hallucinated)")
            else:
                oth = other_field_match()
                if oth: c = f"wrong field: {f} <- value of {oth}" + ("  [merchant_name vs merchant_address]" if {f, oth} == {"merchant_name", "merchant_address"} else "")
                elif f == "tax_total" and any(k in gtall and gtall[k]["norm"] is not None and (amount_eq(pn, gtall[k]["norm"]) or ("cgst" in gtall and "sgst" in gtall and amount_eq(pn, gtall["cgst"]["norm"] + gtall["sgst"]["norm"]))) for k in ("cgst", "sgst", "igst")): c = "tax_total vs CGST/SGST/IGST confusion"
                elif f == "invoice_number" and ((isinstance(pn, str) and PHONE.match(pn)) or (gtall.get("gstin") and pn == gtall["gstin"]["norm"]) or parse_date(str(p["value"]))): c = "invoice_number vs other ID (phone / GSTIN / date)"
                elif f == "currency": c = f"wrong currency code ({gn} read as {pn}; '$' is ambiguous)"
                elif not g["aligned"]: c = "GT not locatable in OCR text (OCR error, unusual format, or value not printed on the page)"
                elif T == "date" and isinstance(pn, str) and isinstance(gn, str) and pn[:4] == gn[:4] and sorted([pn[5:7], pn[8:]]) == sorted([gn[5:7], gn[8:]]): c = "invoice_date: day/month order swapped"
                elif T == "date": c = "wrong date (another date on the document)"
                elif T in ("text", "place") and (pn in gn or gn in pn): c = "partial / over-extended span"
                elif p.get("word_idx") and g["word_idx"] and set(p["word_idx"]) & set(g["word_idx"]): c = "wrong bounding box (overlapping but different span)"
                elif f in ("gstin", "cgst", "sgst", "igst", "hsn_sac") and d["country"] == "IN": c = "Indian-format failure (wrong value)"
                else: c = "wrong value (incorrect candidate / context)"
            cats[c].append(ctx); per_field[f][c] += 1
    return cats, per_field

def ranker_candidate_analysis(te, ranker_pkl):
    """for the candidate ranker on REAL test docs: was the GT among the generated candidates (generation) and was it ranked first (ranking)?"""
    import ranker as R
    models = pickle.load(open(ranker_pkl, "rb"))["models"]; CD = R.build_cands([d for d in te if not d["synthetic"]]); out = {}
    for f, clf in models.items():
        n = inc = top = 0
        for d in te:
            if d["synthetic"] or f not in d["supported_fields"] or d["document_id"] not in CD: continue
            g = d["fields"].get(f)
            if not g: continue
            cs = CD[d["document_id"]].get(R.TYPE_OF[f], []); n += 1
            lab = [R.label_of(f, c, g["norm"]) for c in cs]
            if any(lab): inc += 1
            if cs:
                pr = clf.predict_proba(R.featurize(cs))[:, 1]; top += lab[int(np.argmax(pr))]
        if n: out[f] = dict(n_gt=n, gt_in_candidates=inc, top1_correct=top)
    return out

def bootstrap_systems(docs, preds_by_sys, groups, n=400, seed=SEED):
    """doc-level (grouped) bootstrap of REAL test micro-F1 per system and paired differences; returns CIs"""
    rng = np.random.default_rng(seed); real = [d for d in docs if not d["synthetic"]]
    def counts(d, preds):
        tp = fp = fn = 0
        for f in d["supported_fields"]:
            g = d["fields"].get(f); g = g["norm"] if g else None; p = (preds.get(d["document_id"]) or {}).get(f); p = p["norm"] if p else None
            if p is not None and g is not None and values_match(f, p, g): tp += 1
            else: fp += p is not None; fn += g is not None
        return tp, fp, fn
    C = {s: np.array([counts(d, pr) for d in real]) for s, pr in preds_by_sys.items()}; gid = {}
    for i, d in enumerate(real): gid.setdefault(groups.get(d["document_id"], d["document_id"]), []).append(i)
    keys = list(gid); samples = {s: [] for s in C}
    for _ in range(n):
        idx = [i for k in rng.choice(len(keys), len(keys)) for i in gid[keys[k]]]
        for s, c in C.items():
            tp, fp, fn = c[idx].sum(0); pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); samples[s].append(2 * pr * rc / max(pr + rc, 1e-9))
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    out = {"micro_f1_ci95": {s: ci(v) for s, v in samples.items()}, "paired_difference_ci95": {}}
    names = list(C)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]; diff = np.array(samples[b]) - np.array(samples[a]); out["paired_difference_ci95"][f"{b} - {a}"] = dict(ci=ci(diff), significant=bool(np.percentile(diff, 2.5) > 0 or np.percentile(diff, 97.5) < 0))
    return out

if __name__ == "__main__":
    va, te = load("val"), load("test"); S = {"val": {}, "test": {}}; raw_l = None
    S["val"]["baseline"] = {d["document_id"]: baseline.extract(d) for d in va}; S["test"]["baseline"] = {d["document_id"]: baseline.extract(d) for d in te}
    S["val"]["ranker"] = load_ranker("val", va); S["test"]["ranker"] = load_ranker("test", te)
    if (MOD / "ie_probs.pkl").exists():
        pr = pickle.load(open(MOD / "ie_probs.pkl", "rb")); rv, rt = layoutlm_preds(va, pr, "val"), layoutlm_preds(te, pr, "test"); taus = {}; iso = {}; ll_info = {}
        for f in FIELDS:
            if FIELDS[f] == "code": continue
            taus[f], ll_info[f] = tune_tau(va, rv, f); c = calibrate(va, rv, f, ll_info[f])
            if c is not None: iso[f] = c
        for r in (rv, rt):
            for k, v in r.items():
                for f, p in v.items(): p["confidence"] = float(iso[f].predict([p["score"]])[0]) if f in iso else None
        RAW_LL = {"val": rv, "test": rt}; S["val"]["layoutlm"], S["test"]["layoutlm"] = apply_tau(rv, taus), apply_tau(rt, taus); LL_TAUS = taus; LL_INFO = ll_info
        pickle.dump({"taus": taus, "iso": iso}, open(MOD / "layoutlm_calib.pkl", "wb"))
    systems = list(S["val"]); fields = list(FIELDS); R = {"systems": systems, "by_system": {}}
    for sysn in systems: R["by_system"][sysn] = {sp: evaluate(va if sp == "val" else te, S[sp][sysn]) for sp in ("val", "test")}
    # hybrid: per field choose the system by REAL validation F1 only when >= MIN_REAL_VAL real val docs carry the field; otherwise the pre-defined DEFAULT system (= best real-val micro-F1 over the well-supported fields). Test never used.
    order = [s for s in ("layoutlm", "ranker", "baseline") if s in systems]; vreal = [d for d in va if not d["synthetic"]]
    well = [f for f in fields if score_field(vreal, S["val"][order[0]], f)["n_gt"] >= MIN_REAL_VAL]
    def vmicro(sn):
        tp = fp = fn = 0
        for f in well: r = score_field(vreal, S["val"][sn], f); tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
        p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); return 2 * p * rc / max(p + rc, 1e-9)
    vm = {sn: vmicro(sn) for sn in order}; default_system = max(order, key=lambda sn: (round(vm[sn], 4), -order.index(sn))); choice = {}; choice_info = {}
    for f in fields:
        n_real = score_field(vreal, S["val"][order[0]], f)["n_gt"]
        if n_real >= MIN_REAL_VAL:
            sc = {sn: (lambda r: -1 if r["f1"] is None else r["f1"])(score_field(vreal, S["val"][sn], f)) for sn in order}
            choice[f] = max(order, key=lambda sn: (round(sc[sn], 3), -order.index(sn))); choice_info[f] = dict(system=choice[f], source="best real-validation F1", real_val_gt=n_real, low_support=False, val_f1=sc)
        else:
            choice[f] = default_system; choice_info[f] = dict(system=default_system, source="default system (insufficient real validation support)", real_val_gt=n_real, low_support=True)
    R["default_system"] = default_system; R["default_system_real_val_micro_f1"] = vm; R["choice_info"] = choice_info
    H = {sp: {d["document_id"]: {f: S[sp][choice[f]][d["document_id"]][f] for f in fields if f in S[sp][choice[f]].get(d["document_id"], {})} for d in (va if sp == "val" else te)} for sp in ("val", "test")}
    for sp in H:   # derived field: tax_total = IGST when no explicit tax total is extracted; source 'derived', confidence = min of the parts' confidences (null if any is null)
        for k, v in H[sp].items():
            if "tax_total" not in v and "igst" in v:      # IGST-only bills: the single tax line IS the total tax (annotation convention); CGST+SGST sums are NOT derived (not annotated as tax_total)
                parts = [v["igst"]]; val = round(sum(x["norm"] for x in parts), 2)
                cf = [x.get("confidence") for x in parts]; v["tax_total"] = dict(value=str(val), norm=val, word_idx=[i for x in parts for i in x["word_idx"]], score=None, confidence=None if None in cf else min(cf), source="derived")
    R["hybrid_choice"] = choice; R["by_system"]["hybrid_ungated"] = {sp: evaluate(va if sp == "val" else te, H[sp]) for sp in ("val", "test")}
    HU = H; H = {sp: {d["document_id"]: gate(d["words"], HU[sp][d["document_id"]]) for d in (va if sp == "val" else te)} for sp in ("val", "test")}      # FINAL system = hybrid + applicability gating
    R["by_system"]["hybrid"] = {sp: evaluate(va if sp == "val" else te, H[sp]) for sp in ("val", "test")}
    def emission(preds):   # how often a category-specific field is emitted on documents of sources that do NOT annotate it (hallucination on non-applicable docs), real test docs
        out = {}
        for f in EVID:
            docs = [d for d in te if not d["synthetic"] and f not in d["supported_fields"]]
            if docs: out[f] = dict(n_docs=len(docs), emitted=sum(f in preds[d["document_id"]] for d in docs))
        return out
    R["emission_on_unsupported_real"] = {"ungated": emission(HU["test"]), "gated": emission(H["test"])}
    R["by_category_real_hybrid"] = by_category([d for d in te if not d["synthetic"]], H["test"], ["merchant_name", "invoice_number", "invoice_date", "subtotal", "tax_total", "total_amount", "gstin", "cgst", "sgst", "igst"])
    # critical-field document success
    def doc_ok(docs, preds):
        n = ok = 0
        for d in docs:
            need = [f for f in CRIT if f in d["fields"] and f in d["supported_fields"]]
            if not need: continue
            n += 1; ok += all(f in preds.get(d["document_id"], {}) and values_match(f, preds[d["document_id"]][f]["norm"], d["fields"][f]["norm"]) for f in need)
        return dict(docs=n, all_critical_correct=round(ok / n, 4) if n else None)
    R["doc_level_critical"] = {s: {sl: doc_ok([d for d in te if doc_flags(d)[sl]], (H["test"] if s == "hybrid" else S["test"][s])) for sl in ("real", "synthetic", "real_indian")} for s in systems + ["hybrid"]}
    # calibration (hybrid + per-system) ; only calibrated confidences, never invented
    R["calibration"] = {}
    for s in systems + ["hybrid"]:
        preds = H["test"] if s == "hybrid" else S["test"][s]
        R["calibration"][s] = {sl: calib_report(correctness([d for d in te if doc_flags(d)[sl]], preds, fields)) for sl in ("real", "real_indian")}
    json.dump(R, open(REP / "final_results.json", "w"), indent=1)
    man = {}
    for l in open(META / "manifest.jsonl"): m_ = json.loads(l); man[m_["document_id"]] = m_["group_id"]
    R["significance_real_test"] = bootstrap_systems(te, {**{sn: S["test"][sn] for sn in systems}, "hybrid": H["test"]}, man)
    rb = json.load(open(REP / "ranker_raw_test.json")); RAWH = {d["document_id"]: {} for d in te}
    for f in fields:
        sn = choice[f]
        for d in te:
            k = d["document_id"]; src = {"baseline": S["test"]["baseline"].get(k, {}), "ranker": rb.get(k, {}), "layoutlm": (RAW_LL["test"].get(k, {}) if "layoutlm" in systems else {})}[sn]
            if f in src: RAWH[k][f] = src[f]
    cats, per_field = categorize_errors(te, H["test"], RAWH, fields); summ = {c: len(v) for c, v in sorted(cats.items(), key=lambda t: -len(t[1]))}
    R["ranker_candidate_analysis_real_test"] = ranker_candidate_analysis(te, MOD / "ranker.pkl"); json.dump(R, open(REP / "final_results.json", "w"), indent=1)
    json.dump({"counts": summ, "per_field": {f: dict(c) for f, c in per_field.items()}, "examples": {c: v[:6] for c, v in cats.items()}}, open(REP / "errors.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"choice": choice, "choice_info": choice_info, "default_system": default_system, "layoutlm_taus": LL_TAUS if "layoutlm" in S["val"] else None, "layoutlm_tau_info": LL_INFO if "layoutlm" in S["val"] else None, "ranker_taus": json.load(open(REP / "ranker.json"))["taus"], "ranker_tau_info": json.load(open(REP / "ranker.json"))["tau_info"],
          "policy": dict(MIN_REAL_VAL=MIN_REAL_VAL, DEFAULT_TAU=DEFAULT_TAU, MIN_REAL_CAL=MIN_REAL_CAL)}, open(MOD / "hybrid.json", "w"), indent=1)
    pickle.dump({"preds": H, "S": S}, open(REP / "all_predictions.pkl", "wb"))
    for s in systems + ["hybrid"]:
        for sl in ("real", "real_indian", "synthetic"):
            m = R["by_system"][s]["test"][sl]; print(f"{s:9s} test {sl:11s} micro {micro(m, list(m))['f1']}  macro {macro(m, list(m))}")
    print(summ)
