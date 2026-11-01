"""FEASIBILITY PROBE (not a production model, writes no model file): TF-IDF (char 2-5 + word 1-2) + logistic regression on OCR text for document_type, trained on the TRAIN split of doc_type_manifest_v1
(real documents only) and scored on the VAL split only. The TEST split is never read. Purpose: are the available classes separable from text, and are the val counts large enough to report per-class metrics?
Output: reports/doc_type/text_probe_val_v1.json."""
import sys, json, collections
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
ROOT = Path(__file__).resolve().parents[2]; DS = ROOT / "datasets"
def text(r):
    try: return " ".join(w["text"] for w in json.loads((DS / r["ann_path"]).read_text())["words"])[:4000]
    except Exception: return None
rows = [json.loads(l) for l in open(DS / "metadata/doc_type_manifest_v1.jsonl")]; rows = [r for r in rows if not r["synthetic"] and r["ann_path"] and r["split"] in ("train", "val")]
T = {r["doc_id"]: text(r) for r in rows}; rows = [r for r in rows if T[r["doc_id"]]]
tr = [r for r in rows if r["split"] == "train"]; va = [r for r in rows if r["split"] == "val"]; labels = sorted({r["doc_type"] for r in tr})
vec = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=150000), TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100000))
X = vec.fit_transform([T[r["doc_id"]] for r in tr]); clf = LogisticRegression(C=10, class_weight="balanced", max_iter=2000).fit(X, [r["doc_type"] for r in tr])
Xv = vec.transform([T[r["doc_id"]] for r in va]); yv = [r["doc_type"] for r in va]; pv = clf.predict(Xv); conf = clf.predict_proba(Xv).max(1)
p, rc, f1, sup = precision_recall_fscore_support(yv, pv, labels=labels, zero_division=0); cm = confusion_matrix(yv, pv, labels=labels)
res = dict(split="val (TEST NOT USED)", n_train=len(tr), n_val=len(va), classes=labels, per_class={l: dict(precision=round(float(a), 3), recall=round(float(b), 3), f1=round(float(c), 3), val_support=int(s), reliable=bool(s >= 30)) for l, a, b, c, s in zip(labels, p, rc, f1, sup)},
           macro_f1_all_classes=round(float(f1.mean()), 3), macro_f1_classes_with_val_support_ge30=round(float(np.mean([c for c, s in zip(f1, sup) if s >= 30])), 3), accuracy=round(float((pv == np.array(yv)).mean()), 3), confusion_matrix=dict(labels=labels, rows_true_cols_pred=cm.tolist()))
ind = [i for i, r in enumerate(va) if r["country"] == "IN"]; res["indian_val"] = dict(n=len(ind), accuracy=round(float(np.mean([pv[i] == yv[i] for i in ind])), 3) if ind else None, classes_present=dict(collections.Counter(yv[i] for i in ind)))
res["manual_review_rate_at_conf_lt_0.5"] = round(float((conf < 0.5).mean()), 3)
(ROOT / "reports/doc_type/text_probe_val_v1.json").write_text(json.dumps(res, indent=1)); print(f"train {len(tr)} / val {len(va)} real docs; accuracy {res['accuracy']}; macro-F1 all {res['macro_f1_all_classes']} | only classes with val>=30: {res['macro_f1_classes_with_val_support_ge30']}")
for l, v in res["per_class"].items(): print(f"  {l:18s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} val_n={v['val_support']:3d} {'' if v['reliable'] else '(too few val docs to trust)'}")
print("Indian val:", res["indian_val"]); print("review rate (conf<0.5):", res["manual_review_rate_at_conf_lt_0.5"])
mis = collections.Counter((a, b) for a, b in zip(yv, pv) if a != b); print("top confusions (true->pred):", mis.most_common(6))
