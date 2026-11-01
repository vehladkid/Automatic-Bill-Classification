"""Deterministic baseline: regex + keyword + position heuristics over OCR words/boxes. Deliberately plain (no per-dataset tuning, no checksum tricks);
it only knows generic bill vocabulary. NOTE: several travel/hotel/medical/employee patterns use generic label words ('Check-in', 'Patient Name', 'Rx No')
that the synthetic_b templates also use, so its synthetic_b scores are optimistic; real-document scores are the honest ones."""
import re
from schema import *
from words import AMT_RE, DATE_RES, GSTIN_LIKE
from data import bbox_of

def lines_of(words, boxes):
    """group words into text lines by y-centre proximity -> [{'idx': [...], 'text': str, 'y': float, 'h': float}]"""
    if not words: return []
    hs = sorted(b[3] - b[1] for b in boxes); med = max(hs[len(hs) // 2], 6); order = sorted(range(len(words)), key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
    L = []
    for i in order:
        c = (boxes[i][1] + boxes[i][3]) / 2
        if L and abs(c - L[-1]["y"]) <= 0.6 * med: L[-1]["idx"].append(i); L[-1]["y"] = (L[-1]["y"] * (len(L[-1]["idx"]) - 1) + c) / len(L[-1]["idx"])
        else: L.append({"idx": [i], "y": c})
    for l in L: l["idx"].sort(key=lambda i: boxes[i][0]); l["text"] = " ".join(words[i] for i in l["idx"]); l["h"] = max(boxes[i][3] - boxes[i][1] for i in l["idx"])
    return L

def line_amounts(l, words, locale):
    """amounts in a line as [(value, word_idx_list)] ignoring percentages ('2.5%', '@ 9') ."""
    out = []; text = ""; offs = []
    for i in l["idx"]: offs.append((len(text), len(text) + len(words[i]), i)); text += words[i] + " "
    for m in AMT_RE.finditer(text):
        raw = m.group().strip(); nxt = text[m.end():m.end() + 2].strip()
        if nxt.startswith("%") or re.search(r"@\s*$", text[:m.start()]): continue
        if not re.search(r"[.,]\d{1,3}$|\d{4,}", raw) and not re.search(r"[₹$€]|rs|rm|inr", raw, re.I) and len(re.sub(r"\D", "", raw)) < 3: continue
        v = parse_amount(raw, locale)
        if v is not None and float(v).is_integer() and 1990 <= v <= 2035 and not re.search(r"[.,]\d", raw) and not re.search(r"[\u20b9$\u20ac]|rs|rm", raw, re.I): continue   # a year, not an amount
        if v is not None: out.append((v, [i for s, e, i in offs if s < m.end() and e > m.start()]))
    return out

def infer_locale(text):
    t = text.lower()
    if re.search(r"\brp\.?\s?\d|idr|rupiah", t): return "idr"
    if "€" in t or re.search(r"\beur\b", t) or re.search(r"\d,\d{2}\b(?!\d)", t) and not re.search(r"\d\.\d{2}\b(?!\d)", t): return "eu"
    return "default"
def infer_dayfirst(text): return not re.search(r"\busd\b|\$", text.lower()) or bool(re.search(r"₹|rs\.?|inr", text.lower()))

KEYS = {"total_amount": [r"grand ?total", r"total amount", r"amount payable|net payable|payable amount|amount due|balance due|total due", r"total (?:rs|inr|rm|₹)", r"\btotal\b", r"jumlah|amount paid|bill amount|net amount"],
        "subtotal": [r"sub[ -]?total", r"taxable (?:value|amount)|total before tax|net before tax", r"total excl"],
        "cgst": [r"cgst|c\.g\.s\.t"], "sgst": [r"sgst|utgst|s\.g\.s\.t"], "igst": [r"igst|i\.g\.s\.t"]}
BAD_TOTAL = re.compile(r"sub[ -]?total|total (?:tax|qty|quantity|items?|discount|savings|gst|vat)|cgst|sgst|igst|round|change|cash|tender", re.I)

def pick_amount(L, words, field, locale, fallback_max=False):
    best = None
    for pri, rx in enumerate(KEYS[field]):
        cands = []
        for li, l in enumerate(L):
            if re.search(rx, l["text"], re.I) and not (field == "total_amount" and BAD_TOTAL.search(l["text"]) and not re.search(r"grand|amount payable|net payable|total amount|amount due", l["text"], re.I)):
                am = line_amounts(l, words, locale)
                if not am and li + 1 < len(L): am = line_amounts(L[li + 1], words, locale)[:1]
                if am: cands.append((am[-1], li))
        if cands:
            best = max(cands, key=lambda c: (c[0][0] if field == "total_amount" else 0, c[1])) if field == "total_amount" else cands[0]; break
    if best is None and fallback_max:
        allv = [a for l in L for a in line_amounts(l, words, locale)]
        if allv: best = (max(allv, key=lambda a: a[0]), None)
    return best[0] if best else None

ID_KW = re.compile(r"(?:invoice|inv|bill|receipt|rcpt|cash ?memo|doc(?:ument)?|order|check|folio|ref(?:erence)?)\s*(?:no|number|num|#|id)?\s*[:.#-]?\s*", re.I)
PM = [("upi", r"\bupi\b|phonepe|google ?pay|gpay|bhim"), ("wallet", r"paytm|wallet"), ("net_banking", r"net ?banking|netbanking"), ("credit_card", r"credit"), ("debit_card", r"debit"), ("card", r"\bcard\b|visa|master"), ("cash", r"\bcash\b")]
GENERIC_TOP = re.compile(r"tax invoice|invoice|receipt|cash ?memo|bill|welcome|thank|gstin|gst no|tel|phone|ph[:.]|www|http|copy|original|duplicate|customer|date|table|^\W*$", re.I)
ADDR_KW = re.compile(r"road|rd\b|street|st\b|nagar|sector|plot|floor|jalan|lane|colony|market|mall|avenue|ave\b|blvd|block|no\.|\d{5,6}|taman|bangalore|mumbai|delhi|chennai|kolkata|hyderabad|pune|india|\b[a-z]+,\s", re.I)
PLATE = r"[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{0,3}[ -]?\d{3,4}"

def kv(text, key_rx, val_rx):
    m = re.search(r"(?:" + key_rx + r")\s*[:.\-]?\s*(" + val_rx + ")", text, re.I)
    return m.group(1).strip() if m else None

def extract(doc):
    words, boxes = doc["words"], doc["boxes"]; preds = {}
    if not words: return preds
    L = lines_of(words, boxes); full = "\n".join(l["text"] for l in L); locale = infer_locale(full); dayfirst = infer_dayfirst(full)
    def put(f, val, idx=None, score=None, locale_=None):
        if val in (None, ""): return
        n = normalize(f, val, locale_ or locale, dayfirst)
        if n is None: return
        preds[f] = dict(value=val, norm=n, word_idx=idx or [], bbox=bbox_of(doc, idx) if idx else None, score=score, source="baseline")
    # amounts
    for f in ("total_amount", "subtotal"):
        a = pick_amount(L, words, f, locale, fallback_max=(f == "total_amount"))
        if a: put(f, a[0], a[1])
    for f in ("cgst", "sgst", "igst"):
        a = pick_amount(L, words, f, locale)
        if a and a[0] > 0: put(f, a[0], a[1])
    tl = [l for l in L if re.search(r"\b(?:total )?(?:tax|gst|vat|iva)\b", l["text"], re.I) and not re.search(r"cgst|sgst|igst|gstin|gst no|tax invoice|before tax|excl", l["text"], re.I)]
    if tl:
        am = line_amounts(tl[-1], words, locale)
        if am: put("tax_total", am[-1][0], am[-1][1])
    elif "cgst" in preds and "sgst" in preds: put("tax_total", round(preds["cgst"]["norm"] + preds["sgst"]["norm"], 2), preds["cgst"]["word_idx"] + preds["sgst"]["word_idx"])
    elif "igst" in preds: put("tax_total", preds["igst"]["norm"], preds["igst"]["word_idx"])
    # gstin: strict pattern, checksum-valid preferred, then first
    gs = [(i, g) for i, w in enumerate(words) for g in [repair_gstin(w) or (norm_id(w) if gstin_valid_format(norm_id(w)) else None)] if g]
    if gs:
        ok = [g for g in gs if gstin_checksum_ok(g[1])]; i, g = (ok or gs)[0]; put("gstin", g, [i])
    # date
    cand = []
    for l in L:
        for rx in DATE_RES:
            for m in rx.finditer(l["text"]):
                d = parse_date(m.group(), dayfirst)
                if d: cand.append((0 if re.search(r"\b(date|dt|dated)\b", l["text"], re.I) and not re.search(r"check|due|expiry|exp\b|valid|journey|travel|depart|arriv", l["text"], re.I) else 1, len(cand), d, m.group(), l))
    if cand:
        _, _, d, raw, l = min(cand, key=lambda c: (c[0], c[1])); idx = [i for i in l["idx"] if any(t in words[i] for t in re.split(r"\s+", raw) if t)]
        put("invoice_date", raw, idx)
    # invoice number
    for l in L:
        m = re.search(r"(?:invoice|inv|bill|receipt|rcpt|cash ?memo|doc(?:ument)?|folio)\s*(?:no|number|num|#|id)\b\.?\s*[:.#-]?\s*([A-Z0-9][A-Z0-9/\-_.]{2,24})", l["text"], re.I)
        if m and re.search(r"\d", m.group(1)) and not re.search(r"gst", m.group(1), re.I):
            tok = m.group(1); idx = [i for i in l["idx"] if tok.startswith(words[i].strip(":#")) or words[i] == tok]; put("invoice_number", tok, idx); break
    # currency
    cm = re.search(r"₹|\brs\b\.?|\binr\b|\brm\b|\bmyr\b|\brp\b\.?|\bidr\b|\$|\busd\b|€|\beur\b|£|\bgbp\b|\bcad\b|\baud\b|\bsgd\b|\baed\b", full, re.I)
    if cm: put("currency", {"₹": "INR", "rs": "INR", "inr": "INR", "rm": "MYR", "myr": "MYR", "rp": "IDR", "idr": "IDR", "$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP", "cad": "CAD", "aud": "AUD", "sgd": "SGD", "aed": "AED"}[cm.group().lower().strip(".")])
    # payment
    low = full.lower()
    for name, rx in PM:
        m = re.search(rx, low)
        if m: put("payment_method", name); break
    # merchant (tallest of the first lines that look like a name), address (following lines)
    top = [(li, l) for li, l in enumerate(L[:8]) if len(re.findall(r"[A-Za-z]", l["text"])) >= 3 and not GENERIC_TOP.search(l["text"]) and not re.search(r"\d{6,}", l["text"])]
    mi = None
    if top:
        li, l = max(top[:4], key=lambda t: (t[1]["h"], -t[0])); mi = li; put("merchant_name", l["text"], l["idx"])
    if mi is not None:
        ad = []
        for l in L[mi + 1: mi + 5]:
            if re.search(r"gstin|gst no|tel|phone|ph[:.]|invoice|bill no|date|tax invoice|\bcin\b|fssai|www", l["text"], re.I) and ad: break
            if ADDR_KW.search(l["text"]) or ad: ad.append(l)
            if len(ad) >= 3: break
        if ad: put("merchant_address", " ".join(l["text"] for l in ad), [i for l in ad for i in l["idx"]])
    # codes and travel / hotel / medical / employee (generic label words)
    m = re.search(r"\b(?:hsn|sac)(?:/(?:hsn|sac))?\s*(?:code|no)?\s*[:.\-]?\s*(\d{4,8})\b", full, re.I)
    if m: put("hsn_sac", m.group(1))
    for f, key, val in (("pnr", r"\bpnr(?: no)?", r"\d{8,12}"), ("ticket_number", r"ticket (?:no|number|#)", r"[A-Z0-9]{8,14}"), ("train_number", r"train(?: no)?\.?", r"\d{4,5}"),
                        ("flight_number", r"flight(?: no)?", r"[A-Z0-9]{2}[- ]?\d{2,4}"), ("bus_number", r"bus no", PLATE), ("vehicle_number", r"vehicle(?: no)?", PLATE),
                        ("room_number", r"room(?: no| number)?", r"\d{2,4}"), ("prescription_number", r"(?:rx|prescription)(?: no| number)?", r"[A-Z0-9]{5,12}"), ("employee_id", r"employee id|emp(?:loyee)? id|emp id", r"[A-Z]{0,4}\d{3,8}")):
        v = kv(full, key, val)
        if v: put(f, v)
    for f, key in (("employee_name", r"employee name|emp(?:loyee)? name"), ("passenger_name", r"passenger(?: name)?"), ("patient_name", r"patient(?: name)?"), ("doctor_name", r"doctor|consultant|dr\.")):
        v = kv(full, key, r"[A-Za-z][A-Za-z .]{2,40}(?=\n|$| {2,})")
        if v: put(f, v.split("  ")[0])
    m = re.search(r"from\s*[:\-]?\s*([A-Za-z .]{3,30}?)\s+to\s*[:\-]?\s*([A-Za-z .]{3,30})", full, re.I)
    if m: put("origin", m.group(1)); put("destination", m.group(2))
    for f, key in (("check_in", r"check[- ]?in|arrival"), ("check_out", r"check[- ]?out|departure"), ("travel_date", r"journey date|date of journey|travel date|date of travel")):
        m = re.search(r"(?:" + key + r")\s*(?:date)?\s*[:\-]?\s*([^\n]{6,22})", full, re.I)
        if m and parse_date(m.group(1), dayfirst): put(f, m.group(1))
    v = kv(full, r"nights?", r"\d{1,2}") or kv(full, r"(?:no\.? of )?nights?", r"\d{1,2}")
    if v: put("number_of_nights", v)
    if "merchant_name" in preds and re.search(r"pharmacy|chemist|medical|medicals|apotheke|medplus|apollo", preds["merchant_name"]["value"], re.I): put("pharmacy_name", preds["merchant_name"]["value"], preds["merchant_name"]["word_idx"])
    return preds
