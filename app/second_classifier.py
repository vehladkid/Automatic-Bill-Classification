"""Second-stage document-type classifier (models/doc_type_new): covers the document types Task A has no class for
(rent receipt, rental agreement, insurance premium, school fee, loan certificate/statement, investment certificate, meal voucher, EV document, Form 16, salary slip, bank statement).
Trained on SYNTHETIC data only (scripts/doc_type_new/train_new_classes.py) - treat its output as a suggestion that needs review. Runs on OCR text only.
It has no reject class, so a minimum probability (MIN_PROB) is required; on synthetic bills of the existing types the max probability was <=0.47 at the 99th percentile."""
import re
from pathlib import Path
import joblib
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/doc_type_new/tfidf_lr_synthetic_v1.joblib"
MIN_PROB = 0.60
_FOOT1 = re.compile(r"synthetic\s*sample[^\n]*", re.I); _FOOT2 = re.compile(r"(generated\s*for\s*ml\s*dataset\s*development|not\s*a\s*real\s*document|ml\s*dataset\s*development)", re.I)

def _clean(t):   # identical to scripts/dataset_discovery/tfidf_baseline_v2_1.clean (defaults)
    t = t[:6000]; t = _FOOT2.sub(" ", _FOOT1.sub(" ", t)).lower()
    return re.sub(r"\s+", " ", t)[:4000]

class SecondClassifier:
    def __init__(self):
        m = joblib.load(MODEL); self.vec, self.clf = m["vec"], m["clf"]; self.labels = list(m["labels"])
    def predict(self, text):
        """-> {'label': str|None, 'confidence': float, 'top3': [...], 'accepted': bool}; label is None when below MIN_PROB or text is empty"""
        if not text or len(text.strip()) < 10: return {"label": None, "confidence": 0.0, "top3": [], "accepted": False, "min_prob": MIN_PROB}
        p = self.clf.predict_proba(self.vec.transform([_clean(text)]))[0]; order = p.argsort()[::-1][:3]
        top = [(str(self.clf.classes_[i]), round(float(p[i]), 4)) for i in order]
        ok = top[0][1] >= MIN_PROB
        return {"label": top[0][0] if ok else None, "confidence": top[0][1], "top3": top, "accepted": ok, "min_prob": MIN_PROB}
