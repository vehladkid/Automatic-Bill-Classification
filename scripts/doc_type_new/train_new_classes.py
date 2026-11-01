"""Train a document-type classifier for the classes the existing bill classifier (Task A / legacy doc types) does NOT cover.
SYNTHETIC DATA ONLY (synthgen_v1 for 11 employee/tax types, agami for bank_statement). Split = template held-out (v2.1 manifest): train on 2 layouts, val = 1 rotating layout, test = unseen layout D.
Fixed hyper-parameters (no tuning). Scores here measure synthetic-template generalization only, NOT real-world accuracy."""
import sys, json, collections, joblib
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts/dataset_discovery")
from tfidf_baseline_v2_1 import load_texts, clean, make_vec, metrics
from sklearn.linear_model import LogisticRegression
ROOT = Path(".")
NEW = ["rent_receipt","rental_agreement","insurance_premium_receipt","school_fee_receipt","loan_interest_certificate","loan_statement","investment_certificate","meal_voucher_statement","ev_document","form_16","salary_slip","bank_statement"]
rows = [json.loads(l) for l in open("datasets/metadata/doc_type_manifest_v2_1.jsonl")]
rows = [r for r in rows if r["synthetic"] and r["doc_type"] in NEW and r["source"] in ("synthgen_v1","agami")]
T = load_texts(); rows = [r for r in rows if r["doc_id"] in T and len(T[r["doc_id"]].strip()) > 10]
S = {sp: [r for r in rows if r["split"] == sp] for sp in ("train","val","test")}
X = lambda rs: [clean(T[r["doc_id"]]) for r in rs]; Y = lambda rs: [r["doc_type"] for r in rs]
vec = make_vec(); Xtr = vec.fit_transform(X(S["train"]))
clf = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=0).fit(Xtr, Y(S["train"]))
res = {sp: metrics(Y(S[sp]), list(clf.predict(vec.transform(X(S[sp])))), NEW) for sp in ("val","test")}
for sp in res:
    print(sp, "n=", res[sp]["n"], "acc=", res[sp]["accuracy"], "macroF1=", res[sp]["macro_f1_classes_with_support"])
    for c, v in res[sp]["per_class"].items(): print("   %-28s f1=%.3f P=%.3f R=%.3f n=%d" % (c, v["f1"], v["precision"], v["recall"], v["support"]))
    print("   top confusions:", res[sp]["top_confusions"])
# final model: refit on train+val (all 3 layouts seen), test stays unseen layout
trv = S["train"] + S["val"]; vec2 = make_vec(); clf2 = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=0).fit(vec2.fit_transform(X(trv)), Y(trv))
res["final_refit_test"] = metrics(Y(S["test"]), list(clf2.predict(vec2.transform(X(S["test"])))), NEW)
print("final(train+val) test acc", res["final_refit_test"]["accuracy"], "macroF1", res["final_refit_test"]["macro_f1_classes_with_support"])
joblib.dump(dict(vec=vec2, clf=clf2, labels=NEW), "models/doc_type_new/tfidf_lr_synthetic_v1.joblib")
json.dump(dict(counts={k: len(v) for k, v in S.items()}, per_class_train=dict(collections.Counter(Y(trv))), results=res, caveat="synthetic-only; saturated/optimistic; not real-world accuracy"), open("reports/doc_type/new_classes/metrics.json","w"), indent=1)
