"""Tiny sanity check (NOT the project baseline): TF-IDF(char+word) + LogisticRegression on OCR text, Task A.
Proves labels are learnable & splits are consistent. Reports macro-F1 on val/test and on the REAL-only test subset."""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import make_union
from common import *

def load(s): return [json.loads(l) for l in open(PROCESSED / "classification" / f"{s}.jsonl")]
if __name__ == "__main__":
    tr, va, te = load("train"), load("validation"), load("test")
    vec = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True), TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True))
    X = vec.fit_transform([r["ocr_text"].lower() for r in tr]); clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, [r["label"] for r in tr])
    for name, d in (("val", va), ("test", te), ("test REAL only", [r for r in te if not r["synthetic"]]), ("test SYNTHETIC only", [r for r in te if r["synthetic"]])):
        p = clf.predict(vec.transform([r["ocr_text"].lower() for r in d])); y = [r["label"] for r in d]
        print(f"{name:22s} n={len(d):4d} acc={sum(a == b for a, b in zip(p, y)) / len(y):.3f} macroF1={f1_score(y, p, average='macro'):.3f}")
    r = [x for x in te if not x["synthetic"]]; p = clf.predict(vec.transform([x["ocr_text"].lower() for x in r]))
    print(classification_report([x["label"] for x in r], p, zero_division=0))
