"""Materialise task views (datasets/processed/*) from the final manifest, run integrity/leakage checks, write stats.
Exit code != 0 if any hard check fails. Generated: docs/DATASET_STATS.md, datasets/metadata/stats.json."""
import json, re, sys, collections
import pandas as pd
from common import *

SPL = {"train": "train", "val": "validation", "test": "test"}
LEX = {  # crude keyword lexicon used ONLY to flag possible label noise in human-labelled folders (not for training)
    "food": r"restaurant|menu|biryani|thali|dosa|paneer|coffee|cafe|tea\b|swiggy|zomato|dine|\bkot\b|table", "fuel": r"petrol|diesel|\bhpcl|\bbpcl|indian ?oil|nozzle|\bltr\b|litre|fuel",
    "train": r"irctc|railway|\bpnr\b|train|coach|berth|\bsl\b|3a|2a", "electricity": r"electricity|kwh|units|discom|bescom|meter|tariff",
    "water": r"water|jal|kl\b|sewer", "internet": r"broadband|fibre|fiber|jio|airtel|wifi|mbps|data|postpaid",
    "parking": r"parking|park\b|vehicle", "medical": r"pharmacy|medical|hospital|tablet|batch|exp\b|doctor", "hotel": r"hotel|room|check.?in|tariff|stay"}

def text_of(r):
    if not r["ann_path"]: return ""
    return " ".join(w["text"] for w in json.loads((DS / r["ann_path"]).read_text()).get("words", []))

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(META / "manifest.jsonl")]; df = pd.DataFrame(rows); fail = []
    # ---- hard checks
    if df.document_id.duplicated().any(): fail.append("duplicate document_id")
    miss = [p for p in df.file_path if not (DS / p).exists()]; 
    if miss: fail.append(f"{len(miss)} missing files")
    bad = df[~df.document_type.isin(DOC_TYPES)]; 
    if len(bad): fail.append("invalid document_type")
    bad = df[~df.expense_category.isin(list(CATEGORIES) + ["unlabeled"])]
    if len(bad): fail.append("invalid expense_category")
    hn = df[df.get("hard_negative", False) == True]
    if len(hn) and (hn.expense_category != "non_expense").any(): fail.append("hard negative labelled as an expense category")
    ex = df[(df.expense_category == "non_expense") & (df.document_type != "non_expense_document")]
    if len(ex): fail.append("non_expense with expense document_type")
    leak = df.groupby("group_id").split.nunique(); 
    if (leak > 1).any(): fail.append(f"{(leak > 1).sum()} groups span >1 split")
    leak = df.groupby("sha256").split.nunique(); 
    if (leak > 1).any(): fail.append("identical image bytes in >1 split")
    # ---- views
    for d in ("classification", "document_type", "information_extraction", "anomaly"): (PROCESSED / d).mkdir(parents=True, exist_ok=True)
    outs = collections.defaultdict(list)
    for r in rows:
        s = SPL[r["split"]]; base = dict(document_id=r["document_id"], image=r["file_path"], source=r["source_dataset"], synthetic=r["synthetic"], country=r["country"])
        txt = text_of(r)
        if r["expense_category"] != "unlabeled" and r["label_quality"] in TRUSTED:
            outs[("classification", s)].append({**base, "label": r["expense_category"], "parent_group": r["parent_group"], "ocr_text": txt, "ann_path": r["ann_path"]})
        outs[("document_type", s)].append({**base, "label": r["document_type"], "label_quality": r["label_quality"], "ocr_text": txt, "ann_path": r["ann_path"]})
        if r["field_annotations_available"]:
            outs[("information_extraction", s)].append({**base, "ann_path": r["ann_path"], "ocr_source": r["ocr_source"], "currency": r["currency"]})
    for (task, s), v in outs.items():
        (PROCESSED / task / f"{s}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in v) + "\n")
    # ---- stats
    clf = pd.DataFrame(outs[("classification", "train")] + outs[("classification", "validation")] + outs[("classification", "test")])
    clf["split"] = [s for s in ("train", "validation", "test") for _ in outs[("classification", s)]]
    tab = clf.pivot_table(index="label", columns="split", values="document_id", aggfunc="count", fill_value=0).reindex(columns=["train", "validation", "test"], fill_value=0)
    tab["total"] = tab.sum(axis=1); tab["pct"] = (100 * tab.total / tab.total.sum()).round(1)
    tab["real"] = clf[~clf.synthetic].groupby("label").size().reindex(tab.index, fill_value=0); tab["synthetic"] = tab.total - tab.real
    tab["real_test"] = clf[(~clf.synthetic) & (clf.split == "test")].groupby("label").size().reindex(tab.index, fill_value=0)
    tab["sources"] = clf.groupby("label").source.agg(lambda x: ", ".join(f"{k}:{v}" for k, v in x.value_counts().items()))
    tr = tab["train"].clip(lower=1); tab["class_weight"] = (tr.sum() / (len(tr) * tr)).round(2)
    tab.to_csv(META / "class_distribution.csv")
    # label-noise flags on human-labelled data
    flags = []
    for r in rows:
        if r["label_quality"] in ("manual_folder",) and r["ocr_available"]:
            t = text_of(r).lower(); hits = {c: len(re.findall(rx, t)) for c, rx in LEX.items()}
            best = max(hits, key=hits.get)
            if r["expense_category"] in LEX and hits[best] >= 3 and hits[best] > hits.get(r["expense_category"], 0) + 1 and best != r["expense_category"]:
                flags.append({"document_id": r["document_id"], "labelled": r["expense_category"], "text_suggests": best})
            if len(t) < 30: flags.append({"document_id": r["document_id"], "labelled": r["expense_category"], "text_suggests": "ocr_empty_or_tiny"})
    dump(META / "label_review_flags.json", flags)
    ts = pd.crosstab(df.source_dataset, df.split); ts["total"] = ts.sum(axis=1)
    st = {"total_docs_after_dedup": len(df), "by_source": df.source_dataset.value_counts().to_dict(), "real_vs_synthetic": df.synthetic.value_counts().rename({True: "synthetic", False: "real"}).to_dict(),
          "by_split": df.split.value_counts().to_dict(), "unlabeled_for_task_A": int((df.expense_category == "unlabeled").sum()), "label_review_flags": len(flags),
          "ocr_available": int(df.ocr_available.sum()), "task_A_samples": len(clf), "hard_check_failures": fail}
    dump(META / "stats.json", st)
    dd = json.loads((META / "dedup_stats.json").read_text())
    md = ["# Dataset statistics (generated by scripts/validate_dataset.py)\n", "## Task A: expense_category (trusted labels only)\n", tab.drop(columns="sources").to_markdown(), "\n### Sources per class\n",
          "\n".join(f"- **{k}**: {v}" for k, v in tab.sources.items()), "\n## Documents per source / split (all tasks)\n", ts.to_markdown(),
          "\n## Totals\n```json\n" + json.dumps(st, indent=1) + "\n```\n", "## Dedup\n```json\n" + json.dumps(dd, indent=1) + "\n```\n",
          f"## Label-review flags ({len(flags)})\nSee datasets/metadata/label_review_flags.json\n"]
    # ---- coverage report (real vs synthetic, before/after, provenance)
    lab = df[(df.expense_category != "unlabeled") & df.label_quality.isin(TRUSTED)].copy()
    before = pd.read_csv(META / "class_distribution_before_2026-10-04.csv").set_index("label")
    realdf = lab[~lab.synthetic]; cov = pd.DataFrame({"real_before": before["real"], "real_now": realdf.groupby("expense_category").size(), "synthetic": lab[lab.synthetic].groupby("expense_category").size()}).fillna(0).astype(int)
    cov["real_added"] = cov.real_now - cov.real_before; cov["real_indian"] = realdf[realdf.country == "IN"].groupby("expense_category").size().reindex(cov.index, fill_value=0)
    cov["real_foreign"] = cov.real_now - cov.real_indian
    cov["countries"] = realdf.groupby("expense_category").country.nunique().reindex(cov.index, fill_value=0)
    cov["independent_groups"] = realdf.groupby("expense_category").group_id.nunique().reindex(cov.index, fill_value=0)
    cov["merchants_known"] = realdf.dropna(subset=["merchant"]).groupby("expense_category").merchant.nunique().reindex(cov.index, fill_value=0)
    cov["real_test"] = realdf[realdf.split == "test"].groupby("expense_category").size().reindex(cov.index, fill_value=0)
    cov["real_val"] = realdf[realdf.split == "val"].groupby("expense_category").size().reindex(cov.index, fill_value=0)
    cov = cov[["real_before", "real_added", "real_now", "real_indian", "real_foreign", "synthetic", "independent_groups", "merchants_known", "countries", "real_val", "real_test"]]
    cov.to_csv(META / "coverage_before_after.csv")
    src_cat = realdf.pivot_table(index="expense_category", columns="source_dataset", values="document_id", aggfunc="count", fill_value=0)
    ctry = df[~df.synthetic].country.fillna("unknown").value_counts(); ctry_lab = realdf.country.fillna("unknown").value_counts()
    lang = df.language.fillna("unknown").value_counts(); dtp = df.document_type.value_counts(); dtp_real = df[~df.synthetic].document_type.value_counts()
    hnc = df[df.get("hard_negative", False) == True].groupby(["synthetic", "source_dataset"]).size()
    st.update({"real_labelled_task_A": len(realdf), "synthetic_labelled_task_A": int(lab.synthetic.sum()), "countries_real_all": ctry.to_dict(), "languages_all": lang.to_dict(), "document_types_all": dtp.to_dict(), "hard_negatives": {f"{'synthetic' if a else 'real'}:{b}": int(v) for (a, b), v in hnc.items()}})
    dump(META / "stats.json", st)
    md += ["\n## Coverage before/after (Task A trusted labels)\n", cov.to_markdown(), "\nreal_indian = country IN. independent_groups = near-duplicate/merchant groups (diversity proxy). merchants_known = distinct merchant strings where the source provides one.\n",
           "\n## Real documents per category x source\n", src_cat.to_markdown(), "\n## Country distribution (all real documents)\n", ctry.to_markdown(), "\n## Country distribution (real, Task-A labelled)\n", ctry_lab.to_markdown(),
           "\n## Language (all docs; detected where not stated, see language_source)\n", lang.to_markdown(), "\n## document_type (all docs)\n", dtp.to_markdown(), "\n## document_type (real docs only)\n", dtp_real.to_markdown(),
           "\n## Hard negatives (non_expense carrying amounts/GST/dates)\n", hnc.to_markdown() if len(hnc) else "none"]
    (ROOT / "docs").mkdir(exist_ok=True); (ROOT / "docs/DATASET_STATS.md").write_text("\n".join(md))
    print(tab.drop(columns="sources").to_string()); print(json.dumps(st, indent=1))
    if fail: print("FAILED:", fail); sys.exit(1)
    print("ALL HARD CHECKS PASSED")
