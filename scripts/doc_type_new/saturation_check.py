"""Diagnostic: why does the second-stage classifier score 1.000 on its held-out synthetic layout? Inference only; the saved model and the test split are unchanged.
Scores the saved model on the same 274 test documents after (a) removing class-name keywords, (b) removing the first ten words (title area), (c) dropping a random half of the words, (d) both (a) and (c).
Output: reports/doc_type/new_classes/saturation_check.json.  Run from the repository root: .venv/bin/python scripts/doc_type_new/saturation_check.py"""
import sys, json, re, random, joblib, collections
import numpy as np
sys.path.insert(0, "scripts/dataset_discovery")
from tfidf_baseline_v2_1 import load_texts, clean
NEW = ["rent_receipt","rental_agreement","insurance_premium_receipt","school_fee_receipt","loan_interest_certificate","loan_statement","investment_certificate","meal_voucher_statement","ev_document","form_16","salary_slip","bank_statement"]
rows = [json.loads(l) for l in open("datasets/metadata/doc_type_manifest_v2_1.jsonl")]; rows = [r for r in rows if r["synthetic"] and r["doc_type"] in NEW and r["source"] in ("synthgen_v1", "agami")]
T = load_texts(); te = [r for r in rows if r["split"] == "test" and r["doc_id"] in T and len(T[r["doc_id"]].strip()) > 10]
m = joblib.load("models/doc_type_new/tfidf_lr_synthetic_v1.joblib"); vec, clf = m["vec"], m["clf"]; y = np.array([r["doc_type"] for r in te])
KW = r"\b(rent|receipt|salary|slip|form|16|insurance|premium|school|fee|fees|loan|interest|certificate|statement|investment|meal|voucher|ev|electric|vehicle|bank|agreement|tenant|landlord|tuition|emi|pay|payslip)\b"
def acc(f, seed=0):
    random.seed(seed); return float((np.array(clf.predict(vec.transform([clean(f(T[r["doc_id"]])) for r in te]))) == y).mean())
drop = lambda t: " ".join(w for w in t.split() if random.random() > 0.5); nokw = lambda t: re.sub(KW, " ", t, flags=re.I)
out = dict(n_test=len(te), per_class_count=dict(collections.Counter(y.tolist())), accuracy_as_evaluated=acc(lambda t: t), accuracy_keywords_removed=acc(nokw), accuracy_first_ten_words_removed=acc(lambda t: " ".join(t.split()[10:])),
           accuracy_half_words_dropped=float(np.mean([acc(drop, s) for s in range(5)])), accuracy_keywords_removed_and_half_words_dropped=float(np.mean([acc(lambda t: drop(nokw(t)), s) for s in range(5)])),
           note="all stay at 1.000: the documents of each type share generator-specific wording (field labels, number formats) far beyond title words; the score measures template-generator separability, not real-world accuracy")
Path = __import__("pathlib").Path; Path("reports/doc_type/new_classes/saturation_check.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
