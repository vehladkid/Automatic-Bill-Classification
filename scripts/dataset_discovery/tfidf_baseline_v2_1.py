"""Controlled development baseline: TF-IDF (char_wb 2-5 + word 1-2) + logistic regression document-type classifier on OCR text of the APPROVED v2.1 manifest. Not a production model; nothing is saved except metrics.
Hyper-parameters are FIXED a priori (C=10, class_weight=balanced; same as the v1 probe) - nothing is tuned on any split. Inputs: datasets/metadata/doc_type_v2_1_approved/*.jsonl; OCR text = existing OCR words (real, synthetic_in) or datasets/interim/doctype_v2_1_ocr (synthgen_v1, fatura, agami).
Experiments (each reports per-class P/R/F1/support, macro-F1 over classes with support, confusion matrix):
  real_only      train REAL train          -> REAL val
  synthetic_only train SYNTHETIC train     -> SYNTHETIC val (held-out layout/template), SYNTHETIC test slices (layout D / fatura templates / agami Digital_Type2 / older synthetic_in), REAL val (cross-domain)
  combined       train REAL+SYNTHETIC      -> REAL val, SYNTHETIC val, SYNTHETIC test slices
Diagnostics on the combined model: footer/watermark kept vs stripped, class-title masking, digit masking, source-name masking, label-permutation control, seen-template grouped CV vs held-out template, source-only majority baseline, real->synthetic-class shortcut rate, top features.
Primary model strips the generator watermark ('SYNTHETIC SAMPLE ... not a real document') from ALL texts: it is an artefact that never occurs in real documents. REAL TEST is read only with --final-test (once, after the configuration is frozen).
Output: reports/doc_type/experiments_v3/tfidf_baseline_v2_1.json (+ .md)."""
import sys, json, re, argparse, collections, random
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import ROOT, DS, META, read_jsonl
OUTD = ROOT / "reports/doc_type/experiments_v3"
FOOTER = re.compile(r"synthetic\s*sample[^\n]*", re.I); FOOTER2 = re.compile(r"(generated\s*for\s*ml\s*dataset\s*development|not\s*a\s*real\s*document|ml\s*dataset\s*development)", re.I)  # OCR sometimes drops spaces: "SYNTHETICSAMPLE"
def load_texts():
    cache = {}
    for f in (DS / "interim/doctype_v2_1_ocr").glob("*.jsonl"):
        for l in open(f): j = json.loads(l); cache[j["doc_id"]] = j["text"]
    return cache
def text_of(r, cache):
    if r["doc_id"] in cache: return cache[r["doc_id"]]
    try: return " ".join(w["text"] for w in json.loads((DS / r["ann_path"]).read_text())["words"])
    except Exception: return None
def clean(t, strip_footer=True, mask_digits=False, mask_titles=None, mask_source=None):
    t = t[:6000]
    if strip_footer: t = FOOTER2.sub(" ", FOOTER.sub(" ", t))
    t = t.lower()
    if mask_titles:
        for p in mask_titles: t = t.replace(p, " ")
    if mask_source:
        for p in mask_source: t = re.sub(re.escape(p), " ", t)
    if mask_digits: t = re.sub(r"\d", "0", t)
    return re.sub(r"\s+", " ", t)[:4000]
def make_vec(): return make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=150000), TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100000))
def fit(tr_t, tr_y, seed=0):
    vec = make_vec(); X = vec.fit_transform(tr_t); clf = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=seed).fit(X, tr_y); return vec, clf
def metrics(y, p, labels):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0); cm = confusion_matrix(y, p, labels=labels); has = sup > 0
    return dict(n=int(len(y)), accuracy=round(float(np.mean(np.array(y) == np.array(p))), 4) if len(y) else None, macro_f1_classes_with_support=round(float(f1[has].mean()), 4) if has.any() else None, n_classes_with_support=int(has.sum()),
                per_class={l: dict(precision=round(float(a), 3), recall=round(float(b), 3), f1=round(float(c), 3), support=int(s)) for l, a, b, c, s in zip(labels, pr, rc, f1, sup) if s > 0 or a > 0},
                confusion=dict(labels=labels, rows_true_cols_pred=cm.tolist()), top_confusions=[[a, b, n] for (a, b), n in collections.Counter((a, b) for a, b in zip(y, p) if a != b).most_common(8)])
def group_summary(m, real_backed, synth_only):
    pc = m["per_class"]; f = lambda g: [v["f1"] for c, v in pc.items() if c in g and v["support"] > 0]; n = lambda g: sum(v["support"] for c, v in pc.items() if c in g)
    return dict(real_backed_classes=dict(macro_f1=round(float(np.mean(f(real_backed))), 4) if f(real_backed) else None, n_classes=len(f(real_backed)), n_docs=n(real_backed)), synthetic_only_classes=dict(macro_f1=round(float(np.mean(f(synth_only))), 4) if f(synth_only) else None, n_classes=len(f(synth_only)), n_docs=n(synth_only)))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final-test", action="store_true", help="also read the REAL test split once (frozen configuration, no feedback)"); a = ap.parse_args(); OUTD.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(META / "doc_type_v2_1_approved/manifest.jsonl"); cache = load_texts(); T = {}
    for r in rows:
        t = text_of(r, cache)
        if t and len(t.strip()) > 10: T[r["doc_id"]] = t
    dropped = collections.Counter((r["source"], r["split"]) for r in rows if r["doc_id"] not in T); rows = [r for r in rows if r["doc_id"] in T]
    def sel(syn, sp, pred=lambda r: True): return [r for r in rows if r["synthetic"] == syn and r["split"] == sp and pred(r)]
    RT, RV, ST, SV_all = sel(False, "train"), sel(False, "val"), sel(True, "train"), sel(True, "val"); SV = [r for r in SV_all if r["source"] != "synthetic_in"]  # older synthetic_in is template-ungrouped (same template family in every split): reported separately as optimistic
    slices = {"synthgen_layout_D": sel(True, "test", lambda r: r["source"] == "synthgen_v1"), "fatura_unseen_templates": sel(True, "test", lambda r: r["source"] == "fatura"), "agami_Digital_Type2": sel(True, "test", lambda r: r["source"] == "agami"), "older_synthetic_in_(template-ungrouped)": sel(True, "test", lambda r: r["source"] == "synthetic_in")}
    slices_val = {"synthgen_heldout_val_layout": [r for r in SV if r["source"] == "synthgen_v1"], "fatura_val_templates": [r for r in SV if r["source"] == "fatura"], "agami_Scanned_Type2": [r for r in SV if r["source"] == "agami"], "older_synthetic_in_val_(same_template_optimistic)": [r for r in SV_all if r["source"] == "synthetic_in"]}
    classes_real = sorted({r["doc_type"] for r in RT}); labels = sorted({r["doc_type"] for r in RT + ST}); synth_only = [l for l in labels if l not in classes_real]
    out = dict(config=dict(model="TF-IDF char_wb(2-5)+word(1-2), LogisticRegression C=10 class_weight=balanced, fixed a priori", text="OCR text (first page for generated docs), footer stripped for the primary model", seed=0),
               counts=dict(train_real=len(RT), val_real=len(RV), train_synthetic=len(ST), val_synthetic=len(SV), test_synthetic_slices={k: len(v) for k, v in slices.items()}, dropped_no_text={f"{k[0]}|{k[1]}": v for k, v in dropped.items()},
                           real_train_by_source=dict(collections.Counter(r["source"] for r in RT)), synthetic_train_by_source=dict(collections.Counter(r["source"] for r in ST)),
                           per_class={l: dict(real_train=sum(r["doc_type"] == l for r in RT), real_val=sum(r["doc_type"] == l for r in RV), synth_train=sum(r["doc_type"] == l for r in ST), synth_val=sum(r["doc_type"] == l for r in SV), synth_test=sum(r["doc_type"] == l for v in slices.values() for r in v)) for l in labels}),
               classes_with_real_training_docs=classes_real, classes_without_any_approved_real_docs=synth_only,
               classes_without_real_VAL_examples=[l for l in labels if sum(r["doc_type"] == l for r in RV) == 0], excluded_classes_no_approved_data=["cheque"])
    X = lambda rs, **kw: [clean(T[r["doc_id"]], **kw) for r in rs]; Y = lambda rs: [r["doc_type"] for r in rs]; E = {}
    def run(name, tr, evals, **kw):
        vec, clf = fit(X(tr, **kw), Y(tr)); res = {}
        for en, er in evals.items():
            if not er: continue
            p = clf.predict(vec.transform(X(er, **kw))); res[en] = metrics(Y(er), list(p), labels)
        E[name] = res; return vec, clf
    # primary experiments (footer stripped)
    evalsS = {"synthetic_val_heldout_templates": SV, **{"test:" + k: v for k, v in slices.items()}}
    run("real_only", RT, {"real_val": RV, "synthetic_val_CROSS_DOMAIN": SV, **{"test:" + k + "_CROSS_DOMAIN": v for k, v in slices.items()}}); run("synthetic_only", ST, {"real_val_CROSS_DOMAIN": RV, **evalsS}); vec, clf = run("combined", RT + ST, {"real_val": RV, **evalsS})
    E["combined"].update({f"val_slice:{k}": metrics(Y(v), list(clf.predict(vec.transform(X(v)))), labels) for k, v in slices_val.items() if v})
    fl = [r for r in RV if r["review_status"] == "flagged_for_review"]; E["combined"]["real_val_excluding_flagged_for_review"] = metrics(Y([r for r in RV if r not in fl]), list(clf.predict(vec.transform(X([r for r in RV if r not in fl])))), labels); out["real_val_flagged_for_review"] = len(fl)
    # real_val by origin: Indian vs not
    ind = [r for r in RV if r["country"] == "IN"]; E["combined"]["real_val_Indian_only"] = metrics(Y(ind), list(clf.predict(vec.transform(X(ind)))), labels) if ind else None
    # D. diagnostics
    D = {}
    # D1 footer kept vs stripped
    vk, ck = fit(X(RT + ST, strip_footer=False), Y(RT + ST))
    d1 = {}
    for en, er in {"real_val": RV, "synthgen_layout_D": slices["synthgen_layout_D"], "synthetic_val": SV}.items():
        pk = ck.predict(vk.transform(X(er, strip_footer=False))); pn = clf.predict(vec.transform(X(er)))
        d1[en] = dict(macro_f1_footer_kept=metrics(Y(er), list(pk), labels)["macro_f1_classes_with_support"], macro_f1_footer_stripped=metrics(Y(er), list(pn), labels)["macro_f1_classes_with_support"], accuracy_kept=metrics(Y(er), list(pk), labels)["accuracy"], accuracy_stripped=metrics(Y(er), list(pn), labels)["accuracy"])
    # real docs predicted into synthetic-only classes (shortcut) when footer kept vs stripped, and footer-added to real text (does the marker alone drag a real doc into a synthetic class?)
    FT = "SYNTHETIC SAMPLE - generated for ML dataset development - not a real document"
    pr_kept = ck.predict(vk.transform([clean(T[r["doc_id"]] + " " + FT, strip_footer=False) for r in RV])); pr_plain = ck.predict(vk.transform(X(RV, strip_footer=False)))
    d1["real_val_into_synthetic_only_classes"] = dict(footer_kept_model_plain_real_text=round(float(np.mean([p in synth_only for p in pr_plain])), 4), footer_kept_model_real_text_plus_synthetic_footer=round(float(np.mean([p in synth_only for p in pr_kept])), 4),
                                                       primary_footer_stripped_model=round(float(np.mean([p in synth_only for p in clf.predict(vec.transform(X(RV)))])), 4))
    d1["note"] = "if appending the watermark moves real documents into synthetic-only classes, the model uses the watermark as a class shortcut"; D["footer_kept_vs_stripped"] = d1
    # D2 mask class-defining titles/phrases (from generator key_phrases + titles)
    ph = set()
    for f in (DS / "synthetic_doctype/v1/ann").glob("*.json"):
        j = json.loads(f.read_text()); ph.add(j["title"].lower()); ph.update(k.lower() for k in j["key_phrases"])
    ph = sorted(ph, key=len, reverse=True); D["mask_class_titles_and_key_phrases"] = dict(n_phrases=len(ph), phrases_sample=ph[:12])
    vm, cm_ = fit(X(RT + ST, mask_titles=ph), Y(RT + ST)); ref = {"synthetic_val": E["combined"]["synthetic_val_heldout_templates"], "synthgen_layout_D": E["combined"]["test:synthgen_layout_D"], "real_val": E["combined"]["real_val"]}
    for en, er in {"synthgen_layout_D": slices["synthgen_layout_D"], "synthetic_val": SV, "real_val": RV}.items():
        D["mask_class_titles_and_key_phrases"][en] = dict(macro_f1_masked=metrics(Y(er), list(cm_.predict(vm.transform(X(er, mask_titles=ph)))), labels)["macro_f1_classes_with_support"], macro_f1_unmasked=ref[en]["macro_f1_classes_with_support"])
    # D3 digits masked
    v_, c_ = fit(X(RT + ST, mask_digits=True), Y(RT + ST)); D["mask_digits"] = {en: dict(macro_f1_masked=metrics(Y(er), list(c_.predict(v_.transform(X(er, mask_digits=True)))), labels)["macro_f1_classes_with_support"]) for en, er in {"real_val": RV, "synthetic_val": SV, "synthgen_layout_D": slices["synthgen_layout_D"]}.items()}
    # D4 source/issuer name masking: remove the first 3 OCR lines worth of text? use generator issuer names on synthetic; for real remove top-of-document 25 chars... implemented as removing capitalised issuer strings from generator ann
    iss = set()
    for f in (DS / "synthetic_doctype/v1/ann").glob("*.json"):
        j = json.loads(f.read_text()); 
        if j.get("issuer"): iss.add(j["issuer"].lower())
    iss = sorted(iss, key=len, reverse=True); v_, c_ = fit(X(RT + ST, mask_source=iss), Y(RT + ST)); D["mask_generator_issuer_names"] = dict(n_names=len(iss), synthgen_layout_D=metrics(Y(slices["synthgen_layout_D"]), list(c_.predict(v_.transform(X(slices["synthgen_layout_D"], mask_source=iss)))), labels)["macro_f1_classes_with_support"], unmasked=E["combined"]["test:synthgen_layout_D"]["macro_f1_classes_with_support"])
    # D5 label permutation control
    rng = random.Random(0); yp = Y(RT + ST); yp = rng.sample(yp, len(yp)); v_, c_ = fit(X(RT + ST), yp); D["label_permutation_control"] = dict(real_val_accuracy=metrics(Y(RV), list(c_.predict(v_.transform(X(RV)))), labels)["accuracy"], synthetic_val_accuracy=metrics(Y(SV), list(c_.predict(v_.transform(X(SV)))), labels)["accuracy"], chance_approx=round(1 / len(labels), 3))
    # D6 source-only baseline: predict the majority doc_type of the document's source (+ doc layout unknown)
    maj = {}
    for s in {r["source"] for r in RT + ST}: cnt = collections.Counter(r["doc_type"] for r in RT + ST if r["source"] == s); maj[s] = cnt.most_common(1)[0][0]
    D["source_only_majority_baseline"] = {en: dict(accuracy=round(float(np.mean([maj.get(r["source"]) == r["doc_type"] for r in er])), 4)) for en, er in {"real_val": RV, "synthetic_val": SV, "synthgen_layout_D": slices["synthgen_layout_D"]}.items()}
    D["source_only_majority_baseline"]["note"] = "within synthgen_v1 the source alone cannot separate its 11 types (~1/11 accuracy), but source separates synthetic-only classes from real-backed classes"
    # D7 seen-template (grouped by issuer within TRAIN) vs held-out template
    sg = [r for r in ST if r["source"] == "synthgen_v1"]; fold = lambda r: int(__import__("hashlib").sha1(r["original"]["issuer"].encode()).hexdigest()[:8], 16) % 5; te = [r for r in sg if fold(r) == 0]; tr_ = [r for r in sg if fold(r) != 0]
    v_, c_ = fit(X(tr_), Y(tr_)); sm = metrics(Y(te), list(c_.predict(v_.transform(X(te)))), labels)
    v_, c_ = fit(X(tr_), Y(tr_)); hm = metrics(Y(slices["synthgen_layout_D"]), list(c_.predict(v_.transform(X(slices["synthgen_layout_D"])))), labels)
    D["seen_template_vs_heldout_template_synthgen_only_model"] = dict(seen_layouts_issuer_grouped_holdout=dict(n=sm["n"], macro_f1=sm["macro_f1_classes_with_support"], accuracy=sm["accuracy"]), unseen_layout_D=dict(n=hm["n"], macro_f1=hm["macro_f1_classes_with_support"], accuracy=hm["accuracy"]), note="the difference is the optimism of the v2 split")
    # D8 top features per class (combined model, word features only)
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100000); Xw = wv.fit_transform(X(RT + ST)); cw = LogisticRegression(C=10, class_weight="balanced", max_iter=2000).fit(Xw, Y(RT + ST)); fn = np.array(wv.get_feature_names_out())
    SUSP = re.compile(r"synthetic|sample|fictional|generated|not a real|ml dataset", re.I); D["top_word_features_per_class"] = {c: list(fn[np.argsort(-cw.coef_[i])[:8]]) for i, c in enumerate(cw.classes_)}
    D["suspect_generator_marker_features_in_top20"] = {c: [f for f in fn[np.argsort(-cw.coef_[i])[:20]] if SUSP.search(f)] for i, c in enumerate(cw.classes_) if any(SUSP.search(f) for f in fn[np.argsort(-cw.coef_[i])[:20]])}
    out["group_summary_real_backed_vs_synthetic_only_classes"] = {f"{k}|{kk}": group_summary(m, classes_real, synth_only) for k, v in E.items() for kk, m in v.items() if m}
    resid = sum("synthetic" in clean(T[r["doc_id"]]) for r in RT + ST + SV); out["footer_residue_after_stripping_docs"] = resid
    out["experiments"] = E; out["diagnostics"] = D
    if a.final_test:
        RTE = sel(False, "test"); vr, cr = fit(X(RT), Y(RT)); out["FINAL_real_test_single_read"] = dict(n=len(RTE), note="real test = the protected Task A/B test split; configuration frozen before this read; not used for any decision", real_only=metrics(Y(RTE), list(cr.predict(vr.transform(X(RTE)))), labels), combined=metrics(Y(RTE), list(clf.predict(vec.transform(X(RTE)))), labels))
    (OUTD / "tfidf_baseline_v2_1.json").write_text(json.dumps(out, indent=1)); 
    def line(n, m): return f"{n}: n={m['n']} acc={m['accuracy']} macroF1={m['macro_f1_classes_with_support']} ({m['n_classes_with_support']} classes)"
    for k, v in E.items():
        for kk, m in v.items():
            if m: print(k, "|", line(kk, m))
    print(json.dumps({k: v for k, v in D.items() if k not in ("top_word_features_per_class",)}, indent=1)[:5000])
if __name__ == "__main__": main()
