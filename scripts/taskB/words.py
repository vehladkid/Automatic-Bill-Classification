"""OCR segments -> words with sub-boxes (OCR 'words' in the manifest are line-level segments); text view with char offsets; regex candidate finders."""
import re, json
from pathlib import Path
from schema import *

def load_ann(rel):
    return json.loads((DS / rel).read_text()) if rel else {"words": [], "fields": {}}

LABEL_VOCAB = {"gstin", "gst", "gstno", "gstn", "gstin no", "no", "nos", "inv", "invoice", "bill", "billno", "date", "total", "tin", "ph", "tel", "phone", "fssai", "pan", "cin", "hsn", "sac", "room", "pnr", "id", "ref", "vat", "cgst", "sgst", "igst", "tax", "amount", "amt", "grand", "net", "sub", "subtotal"}
_LAB = re.compile(r"^([A-Za-z.()/&]{1,14})([:=\uff1a]+|[-\u2013]+)(?=[A-Za-z0-9\u20b9$])")
def split_label(tok):
    """'GSTIN-33AABPV3906B3Z3' / 'NO:29AAC..' / 'Total:500.00' -> ['GSTIN-', '33AAB..']; OCR engines glue label and value. '-' splits only after known label words."""
    m = _LAB.match(tok)
    if not m or len(tok) <= m.end() + 1: return [tok]
    lab, sep = m.group(1), m.group(2)
    if sep[0] in "-\u2013" and lab.lower().strip(".") not in LABEL_VOCAB: return [tok]
    if len(re.sub(r"\D", "", tok[m.end():])) == 0 and not re.match(r"^[A-Z0-9]{6,}$", tok[m.end():]): return [tok]
    return [tok[:m.end()], tok[m.end():]]

def split_all(tok):
    """apply split_label repeatedly ('Bill-No:123' -> 'Bill-','No:','123') and split '...%:12.00' style tax tokens"""
    m = re.match(r"^(.*[%)])[:=]*(\d[\d.,]*)$", tok)
    if m and len(m.group(1)) >= 3 and re.search(r"(?i)gst|tax|vat", m.group(1)): return [m.group(1) + (":" if ":" in tok[len(m.group(1)):len(m.group(1)) + 1] else ""), m.group(2)]
    p = split_label(tok)
    return p if len(p) == 1 else p[:1] + split_all(p[1])

def split_words(segments):
    """segments [{text,bbox}] -> list of dict(text, bbox, seg, line_h) in reading order; sub-boxes proportional to characters."""
    out = []
    for si, s in enumerate(segments):
        t = str(s.get("text", "")).strip(); b = s.get("bbox") or []
        if not t or len(b) != 4: continue
        x0, y0, x1, y1 = b; n = len(t); pos = 0
        for m in re.finditer(r"\S+", t):
            a0 = m.start(); parts = split_all(m.group())
            for part in parts:
                a, c = a0, a0 + len(part); a0 = c; wx0 = x0 + (x1 - x0) * a / n; wx1 = x0 + (x1 - x0) * c / n
                out.append(dict(text=part, bbox=[int(wx0), int(y0), int(max(wx1, wx0 + 1)), int(y1)], seg=si, h=int(y1 - y0)))
    return out

def reading_order(words, H=None):
    """Sort by line (y-centre bucket relative to median height) then x."""
    if not words: return words
    hs = sorted(w["h"] for w in words); med = max(hs[len(hs) // 2], 6)
    return sorted(words, key=lambda w: (round(((w["bbox"][1] + w["bbox"][3]) / 2) / (0.7 * med)), w["bbox"][0]))

def text_view(words):
    """'w0 w1 w2' string and (start,end) offsets per word."""
    offs = []; pos = 0; parts = []
    for w in words: offs.append((pos, pos + len(w["text"]))); parts.append(w["text"]); pos += len(w["text"]) + 1
    return " ".join(parts), offs

def span_words(offs, a, b):
    """word indices overlapping char range [a,b)."""
    return [i for i, (s, e) in enumerate(offs) if s < b and e > a]

AMT_RE = re.compile(r"(?<![\w.,])(?:[₹$€£]|rs\.?|inr|rm|rp\.?)?\s?-?\d{1,3}(?:[,.]\d{3})*(?:[.,]\d{1,3})?(?![\w])|(?<![\w.,])-?\d+(?:[.,]\d{1,3})?(?![\w])", re.I)
DATE_RES = [re.compile(p, re.I) for p in (
    r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b", r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b", r"\b\d{1,2}[-/. ]?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/., ]*\d{2,4}\b",
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/. ]+\d{1,2}(?:st|nd|rd|th)?[-/.,  ]+\d{2,4}\b")]
GSTIN_LIKE = re.compile(r"\b[0-9A-Z]{15}\b")

def amount_candidates(text, offs, locale="default"):
    """[(value, word_indices, char_span)] for every amount-looking token (decimals preferred)."""
    c = []
    for m in AMT_RE.finditer(text):
        raw = m.group().strip(); digits = re.sub(r"\D", "", raw)
        if len(digits) > 9 or not digits: continue
        v = parse_amount(raw, locale)
        if v is None: continue
        c.append((v, span_words(offs, m.start(), m.end()), (m.start(), m.end()), raw))
    return c

def date_candidates(text, offs, dayfirst=True):
    c = []
    for rx in DATE_RES:
        for m in rx.finditer(text):
            d = parse_date(m.group(), dayfirst)
            if d: c.append((d, span_words(offs, m.start(), m.end()), (m.start(), m.end()), m.group()))
    return c
