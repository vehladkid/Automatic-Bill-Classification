"""Task B schema, normalisers and validators. Single source of truth for field names, types and value normalisation (used by dataset build, baseline,
ranker, LayoutLM and evaluation). No torch import."""
import re, sys, unicodedata, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, DS, META, PROCESSED, INTERIM, SEED

# name -> type ; groups used for reporting
FIELDS = {
    "merchant_name": "text", "merchant_address": "text", "invoice_number": "id", "invoice_date": "date", "subtotal": "amount", "tax_total": "amount",
    "total_amount": "amount", "currency": "code", "payment_method": "enum",
    "gstin": "gstin", "cgst": "amount", "sgst": "amount", "igst": "amount", "hsn_sac": "id",
    "employee_name": "text", "employee_id": "id",
    "origin": "place", "destination": "place", "travel_date": "date", "pnr": "id", "ticket_number": "id", "passenger_name": "text",
    "flight_number": "id", "train_number": "id", "bus_number": "id", "vehicle_number": "id",
    "check_in": "date", "check_out": "date", "room_number": "id", "number_of_nights": "int",
    "patient_name": "text", "prescription_number": "id", "pharmacy_name": "text", "doctor_name": "text",
}
GROUPS = {"universal": ["merchant_name", "merchant_address", "invoice_number", "invoice_date", "subtotal", "tax_total", "total_amount", "currency", "payment_method"],
          "indian_tax": ["gstin", "cgst", "sgst", "igst", "hsn_sac"], "employee": ["employee_name", "employee_id"],
          "travel": ["origin", "destination", "travel_date", "pnr", "ticket_number", "passenger_name", "flight_number", "train_number", "bus_number", "vehicle_number"],
          "hotel": ["check_in", "check_out", "room_number", "number_of_nights"], "medical": ["patient_name", "prescription_number", "pharmacy_name", "doctor_name"]}
PAYMENT = ["cash", "upi", "credit_card", "debit_card", "card", "net_banking", "wallet"]
AMOUNT_FIELDS = [f for f, t in FIELDS.items() if t == "amount"]

# ---------- text / ids ----------
def norm_text(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", s)
def norm_id(s): return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())
def norm_place(s):
    s = str(s or ""); s = s.split("-")[-1] if re.match(r"^[A-Za-z .]+-[A-Za-z .()]+$", s) else s   # 'India-Bengaluru' -> 'Bengaluru'
    return norm_text(s)
def norm_pm(s):
    t = str(s or "").casefold()
    for k, v in (("upi", "upi"), ("paytm", "wallet"), ("wallet", "wallet"), ("net banking", "net_banking"), ("netbanking", "net_banking"), ("credit", "credit_card"), ("debit", "debit_card"), ("card", "card"), ("visa", "card"), ("master", "card"), ("cash", "cash")):
        if k in t: return v
    return None

# ---------- amounts ----------
_CUR = re.compile(r"(?i)(?:rs\.?|inr|rm|myr|idr|rp\.?|usd|eur|gbp|cad|aud|sgd|aed|[₹$€£])")
def parse_amount(s, locale="default"):
    """-> float or None. locale: 'default' (1,234.50), 'eu' (1.234,50 / 6,14), 'idr' (thousand separators only, no decimals)."""
    if s is None: return None
    if isinstance(s, (int, float)): return round(float(s), 2)
    t = _CUR.sub("", str(s)).strip().replace(" ", "").replace(" ", "")
    neg = t.startswith("-") or (t.startswith("(") and t.endswith(")")); t = t.strip("-()")
    t = re.sub(r"[^0-9.,]", "", t)
    if not t or not re.search(r"\d", t): return None
    if locale == "idr":
        t = re.sub(r"[.,](?=\d{3}(\D|$))", "", t); t = t.replace(",", ".")
    elif "," in t and "." in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."; th = "." if dec == "," else ","; t = t.replace(th, "").replace(dec, ".")
    elif "," in t:
        if locale == "eu" or re.search(r",\d{1,2}$", t) and not re.search(r"^\d{1,3}(,\d{3})+$", t): t = t.replace(".", "").replace(",", ".")
        else: t = t.replace(",", "")
    elif "." in t and re.match(r"^\d{1,3}(\.\d{3})+$", t) and locale in ("eu",): t = t.replace(".", "")
    try: v = float(t)
    except ValueError: return None
    return round(-v if neg else v, 2)
def amount_eq(a, b, tol=0.005): return a is not None and b is not None and abs(a - b) <= tol

# ---------- dates ----------
_MON = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
def parse_date(s, dayfirst=True):
    """-> 'YYYY-MM-DD' or None. Conservative: only explicit numeric / month-name formats; 2-digit years -> 2000+."""
    if not s: return None
    t = str(s).strip().lower(); t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t); t = re.sub(r"[,]", " ", t); t = re.sub(r"\s+", " ", t)
    def ok(y, m, d):
        if y < 100: y += 2000
        try: return dt.date(y, m, d).isoformat()
        except ValueError: return None
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", t)
    if m: return ok(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"\b(\d{1,2})[-/. ]([a-z]{3})[a-z]*[-/. ]*(\d{2,4})\b", t)
    if m and m[2] in _MON: return ok(int(m[3]), _MON[m[2]], int(m[1]))
    m = re.search(r"\b([a-z]{3})[a-z]*[-/. ]+(\d{1,2})[-/. ]+(\d{2,4})\b", t)
    if m and m[1] in _MON: return ok(int(m[3]), _MON[m[1]], int(m[2]))
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", t)
    if m:
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        if a > 12: return ok(y, b, a)
        if b > 12: return ok(y, a, b)
        return ok(y, b, a) if dayfirst else ok(y, a, b)
    return None

# ---------- GSTIN ----------
GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b")
_ALPHA = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def gstin_checksum_ok(g):
    g = norm_id(g)
    if len(g) != 15: return False
    tot = 0
    for i, c in enumerate(g[:14]):
        v = _ALPHA.index(c) * (1 if i % 2 == 0 else 2); tot += v // 36 + v % 36
    return _ALPHA[(36 - tot % 36) % 36] == g[14]
def gstin_valid_format(g):
    g = norm_id(g); return bool(GSTIN_RE.fullmatch(g)) and 1 <= int(g[:2]) <= 38
def norm_gstin(s):
    """GSTIN from a value ('GSTIN: 27AAB...' ok). Full-string first, then a word-bounded search in the raw text; never from inside longer IDs."""
    g = norm_id(s)
    if len(g) == 15 and GSTIN_RE.fullmatch(g): return g
    m = GSTIN_RE.search(str(s or "").upper().replace("-", "").replace(" ", " ")); return m[1] if m else None

# ---------- generic value normalisation + matching ----------
def normalize(field, value, locale="default", dayfirst=True):
    if value in (None, ""): return None
    t = FIELDS[field]
    if t == "amount": return parse_amount(value, locale)
    if t == "date": return parse_date(value, dayfirst)
    if t == "gstin": return norm_gstin(value) or repair_gstin(re.sub(r"^.*?[:=\-]", "", str(value)) if re.search(r"[:=]", str(value)) else value)
    if t == "id": return norm_id(value) or None
    if t == "place": return norm_place(value) or None
    if t == "enum": return norm_pm(value)
    if t == "code": return str(value).strip().upper() or None
    if t == "int":
        try: return int(float(value))
        except (TypeError, ValueError): return None
    return norm_text(value) or None
def similarity(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0
def values_match(field, a, b, soft=False):
    """a, b are NORMALISED values. soft=True: 0.9 string similarity for text/place fields (reported separately from exact)."""
    if a is None or b is None: return False
    if FIELDS[field] == "amount": return amount_eq(a, b)
    if soft and FIELDS[field] in ("text", "place"): return a == b or similarity(a, b) >= 0.9 or (len(a) >= 6 and (a in b or b in a))
    return a == b

# ---------- OCR-aware GSTIN repair ----------
_D2L = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A", "7": "T"}
_L2D = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "A": "4", "T": "7"}
def repair_gstin(tok):
    """Position-aware repair of an OCR'd 15-char GSTIN candidate: slots 0-1 digits, 2-6 letters, 7-10 digits, 11 letter, 12 [1-9A-Z], 13 'Z', 14 alnum.
    Only digit<->letter confusions are fixed; returns a format-valid GSTIN or None. Prefers a checksum-valid repair when one exists."""
    t = re.sub(r"[^A-Z0-9]", "", str(tok or "").upper())
    if len(t) != 15: return None
    kind = "DDLLLLLDDDDLXZX"; out = []
    for ch, k in zip(t, kind):
        if k == "D": out.append(ch if ch.isdigit() else _L2D.get(ch, ch))
        elif k == "L": out.append(ch if ch.isalpha() else _D2L.get(ch, ch))
        elif k == "Z": out.append("Z" if ch in ("Z", "2") else ch)
        else: out.append(ch)
    g = "".join(out)
    return g if gstin_valid_format(g) else None

# ---------- threshold / calibration policy (documented in docs/TASK_B_REPORT.md) ----------
MIN_REAL_VAL = 30      # a field's score threshold / system choice is tuned on REAL validation documents only if >= this many real validation docs have a GT value for it
DEFAULT_TAU = 0.5      # otherwise: fixed, untuned default (a prediction is kept only if the model score >= 0.5); such fields are flagged LOW-SUPPORT everywhere
MIN_REAL_CAL = 30      # isotonic confidence calibration uses real validation predictions if >= this many exist, else all validation predictions (flagged low-support)
