"""Validate LLM-proposed values against the OCR text. Core anti-hallucination rule: a value is usable only if it is GROUNDED in the OCR output
(found verbatim, or reachable by a small confusable-character correction, or a checksum-valid repair). Otherwise it is rejected, never used."""
import re, sys, datetime as dt
from pathlib import Path
_T = Path(__file__).resolve().parents[2] / "scripts" / "taskB"
for _p in (_T, _T.parent):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import schema as S
import doc_fields as DF

CONF = {"O": "0", "0": "O", "I": "1", "1": "I", "L": "1", "S": "5", "5": "S", "B": "8", "8": "B", "Z": "2", "2": "Z", "G": "6", "6": "G", "Q": "0", "D": "0", "T": "7", "A": "4"}
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d{1,2})?(?![\w])")
DATE_RE = re.compile(r"\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}\b|\b\d{1,2}[ -](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ,\-]*\d{2,4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2},? \d{2,4}\b", re.I)

ZERO_OK = {"cgst", "sgst", "igst", "tax_total", "discount", "round_off", "taxable_value", "principal_amount", "interest_amount"}
_LABELS = {"cgst": ("cgst",), "sgst": ("sgst",), "igst": ("igst",), "tax_total": ("tax", "gst"), "discount": ("discount", "disc"), "round_off": ("round",), "taxable_value": ("taxable",), "principal_amount": ("principal",), "interest_amount": ("interest",)}
def _zero_labelled(field, ocr):
    """a zero value is only believable when a zero figure sits in a segment that names the field (e.g. 'CGST 0.00')"""
    for s in ocr.segs:
        t = s["text"].lower()
        if any(w in t for w in _LABELS.get(field, ())) and any(S.parse_amount(m) == 0 for m in NUM_RE.findall(s["text"])): return True
    return False

class OCR:
    """Searchable view of OCR segments [{'text','bbox','conf'}]."""
    def __init__(self, segs):
        self.segs = [s for s in (segs or []) if str(s.get("text", "")).strip()]
        self.text = "\n".join(s["text"] for s in self.segs); self.upper = self.text.upper()
        self.norm = S.norm_text(self.text); self.idnorm = S.norm_id(self.text)
        self.amounts = {}
        for s in self.segs:
            for m in NUM_RE.findall(s["text"]):
                v = S.parse_amount(m)
                if v is not None: self.amounts.setdefault(round(v, 2), s)
        self.dates = []
        for s in self.segs:
            for m in DATE_RE.findall(s["text"]):
                v = S.parse_date(m)
                if v: self.dates.append((v, s))
    def mean_conf(self):
        c = [float(s.get("conf", 1.0)) for s in self.segs]; return sum(c) / len(c) if c else 0.0
    def find_text(self, value, thr=0.85):
        """best (similarity, segment) of normalised value vs each segment and each pair of adjacent segments"""
        v = S.norm_text(value)
        if not v: return 0.0, None
        best = (0.0, None)
        for i, s in enumerate(self.segs):
            for cand, seg in ((S.norm_text(s["text"]), s), *(((S.norm_text(s["text"] + self.segs[i + 1]["text"]), s),) if i + 1 < len(self.segs) else ())):
                if not cand: continue
                sim = 1.0 if (v == cand or (len(v) >= 5 and v in cand)) else S.similarity(v, cand)
                if sim > best[0]: best = (sim, seg)
        return best

def windows(t, n):
    """all length-n substrings (tokens are glued after id-normalisation, so scan every offset)"""
    return [t[i:i + n] for i in range(0, max(0, len(t) - n + 1))]

def confusable_distance(a, b):
    """Edit distance counting only substitutions between visually confusable characters; None if any other difference exists."""
    if len(a) != len(b): return None
    d = 0
    for x, y in zip(a, b):
        if x == y: continue
        if CONF.get(x) == y: d += 1
        else: return None
    return d

def ground(field, value, ocr):
    """Returns dict(status in grounded|corrected|ungrounded|invalid, norm, seg, detail)."""
    t = DF.ALL_FIELDS.get(field, "text"); bad = lambda why: dict(status="ungrounded", norm=None, seg=None, detail=why)
    if value in (None, "", []): return dict(status="absent", norm=None, seg=None, detail="no value")
    if t == "amount":
        v = S.parse_amount(value)
        if v is None: return dict(status="invalid", norm=None, seg=None, detail="not a number")
        if v == 0 and field not in ZERO_OK: return bad("an amount of 0 is not accepted for this field (a missing value is null, not 0)")
        if v == 0 and not _zero_labelled(field, ocr): return bad("0 is printed in the document but not next to a label for this field")
        seg = ocr.amounts.get(round(v, 2))
        return dict(status="grounded", norm=v, seg=seg, detail="number found in OCR") if seg else bad(f"{v:g} does not appear in the OCR text")
    if t == "gstin":
        g = S.norm_id(value)
        if not S.gstin_valid_format(g): return dict(status="invalid", norm=None, seg=None, detail="not a valid GSTIN format")
        ck = S.gstin_checksum_ok(g)
        if g in ocr.idnorm:
            seg = next((s for s in ocr.segs if g in S.norm_id(s["text"])), None)
            return dict(status="grounded", norm=g, seg=seg, detail="in OCR; checksum " + ("valid" if ck else "INVALID"), checksum=ck)
        for s in ocr.segs:                       # near-match to an OCR token via confusable chars only
            for tok in windows(S.norm_id(s["text"]), 15):
                dd = confusable_distance(tok, g)
                if dd is not None and dd <= 2 and ck: return dict(status="corrected", norm=g, seg=s, detail=f"OCR token {tok} -> {g} ({dd} confusable char(s)); checksum valid", checksum=True)
        return bad("GSTIN not in OCR and not a checksum-valid confusable-character repair of any OCR token")
    if t == "date":
        v = S.parse_date(value)
        if not v: return dict(status="invalid", norm=None, seg=None, detail="not a valid date")
        try: y = int(v[:4]); ok = 2000 <= y <= dt.date.today().year + 1
        except ValueError: ok = False
        if not ok: return dict(status="invalid", norm=None, seg=None, detail="date outside plausible range")
        for dv, s in ocr.dates:
            if dv == v: return dict(status="grounded", norm=v, seg=s, detail="date string present in OCR")
        return bad("that date does not appear in the OCR text")
    if t == "id":
        v = S.norm_id(value)
        if len(v) < 3: return dict(status="invalid", norm=None, seg=None, detail="too short")
        if v in ocr.idnorm:
            seg = next((s for s in ocr.segs if v in S.norm_id(s["text"])), None); return dict(status="grounded", norm=v, seg=seg, detail="in OCR")
        for s in ocr.segs:
            for tok in windows(S.norm_id(s["text"]), len(v)):
                dd = confusable_distance(tok, v)
                if dd is not None and 0 < dd <= 1: return dict(status="corrected", norm=v, seg=s, detail=f"OCR token {tok} -> {v} (one confusable char)")
        return bad("identifier not found in the OCR text")
    # text / place / enum / code
    sim, seg = ocr.find_text(value)
    if sim >= 0.999: return dict(status="grounded", norm=S.norm_text(value), seg=seg, detail="text found in OCR")
    if sim >= 0.85: return dict(status="corrected", norm=S.norm_text(value), seg=seg, detail=f"close match to OCR text (similarity {sim:.2f})")
    return bad(f"text not found in OCR (best similarity {sim:.2f})")

def consistency(f):
    """f: {name: normalised value}. Returns list of (check, ok, detail) for relations that can be tested; amounts in rupees, tolerance max(1.0, 0.5%)."""
    g = lambda k: f.get(k); out = []; tol = lambda x: max(1.0, 0.005 * abs(x))
    tax_parts = [g(k) for k in ("cgst", "sgst", "igst") if g(k) is not None]
    if g("tax_total") is not None and tax_parts:
        s = sum(tax_parts); out.append(("tax_total = cgst+sgst+igst", abs(s - g("tax_total")) <= tol(s), f"{s:g} vs {g('tax_total'):g}"))
    if g("subtotal") is not None and g("total_amount") is not None:
        tax = g("tax_total") if g("tax_total") is not None else (sum(tax_parts) if tax_parts else None)
        if tax is not None:
            s = g("subtotal") + tax; out.append(("total = subtotal + tax (within round-off)", abs(s - g("total_amount")) <= max(tol(s), 1.0), f"{s:g} vs {g('total_amount'):g}"))
    if g("cgst") is not None and g("sgst") is not None: out.append(("cgst = sgst", abs(g("cgst") - g("sgst")) <= 0.02, f"{g('cgst'):g} vs {g('sgst'):g}"))
    return out

def validate_correction(c, ocr):
    """An OCR correction {'ocr','corrected','reason'} is accepted only if the original string exists in the OCR text AND the change is confusable-character-only,
    AND the corrected string is structurally valid for its kind (GSTIN checksum, date, amount). Everything else is rejected."""
    o, n = str(c.get("ocr", "")).strip(), str(c.get("corrected", "")).strip()
    if not o or not n or o == n: return False, "empty or unchanged"
    if o not in ocr.text and o.upper() not in ocr.upper: return False, "original string is not in the OCR text"
    a, b = re.sub(r"\s", "", o).upper(), re.sub(r"\s", "", n).upper()
    dd = confusable_distance(a, b)
    if dd is None or dd > max(2, len(a) // 5): return False, "change is not a small confusable-character fix"
    if re.fullmatch(r"[0-9A-Z]{15}", b) and S.gstin_valid_format(b): return (S.gstin_checksum_ok(b), "GSTIN checksum " + ("valid" if S.gstin_checksum_ok(b) else "invalid"))
    if S.parse_date(n) and (DATE_RE.search(o) or re.fullmatch(r"[\dO]{1,2}[/\-.][\dO]{1,2}[/\-.][\dO]{2,4}", o.upper())): return True, "valid date"
    if re.fullmatch(r"[\d,]+(?:\.\d+)?", b.replace("O", "0")): return (S.parse_amount(n) is not None, "valid number")
    return False, "no structural check available for this kind of text; kept as suggestion only"
