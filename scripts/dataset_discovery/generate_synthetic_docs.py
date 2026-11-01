"""Synthetic Indian employee-expense / financial / tax-evidence document generator  (generator version: synthgen-v1).
Fills the categories for which NO public labelled data exists. Every document is rendered from an explicit fictional structured record (saved as the machine-readable label), through one of four layout
families, with optional degradation (scan / phone photo / low resolution) and, where feasible, a Hindi header. NOTHING here is a real person, issuer, identifier or transaction: names/issuers are random
combinations checked against a deny-list of real brands; PAN/TAN/GSTIN/account numbers are random strings in the right FORMAT (resemblance to a real one is coincidental). Every page carries a small
"SYNTHETIC SAMPLE" footer so the images cannot be mistaken for, or used as, real documents (the footer is identical across classes, so it carries no class information).
Deterministic: (base seed, type, index) -> document; re-running reproduces identical records and images.  Synthetic data supports DEVELOPMENT and pipeline checks only - never evidence of real-world accuracy.
  .venv/bin/python scripts/dataset_discovery/generate_synthetic_docs.py --out datasets/synthetic_doctype/v1 --n 110 --seed 20261019 [--types rent_receipt ...] [--limit-preview]
Layouts: A formal letterhead, B banded modern, C narrow thermal-slip (mono), D boxed form (RESERVED for the test split = unseen-layout check; also the layout carrying Hindi titles)."""
import sys, os, json, hashlib, random, math, glob, argparse, datetime as dt
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, features

VERSION = "synthgen-v1"
W, H = 1240, 1754                               # A4 @ 150 dpi
FOOTER = "SYNTHETIC SAMPLE - generated for ML dataset development - not a real document"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "taskB"))
try:
    import schema as _S
except Exception:
    _S = None

# ------------------------------------------------------------------ fonts
def _find(*names):
    for n in names:
        g = glob.glob(f"/usr/share/fonts/**/{n}", recursive=True) + glob.glob(os.path.expanduser(f"~/.fonts/{n}"))
        if g: return g[0]
    return None
FONT_FILES = {
    "sans": (_find("LiberationSans-Regular.ttf", "DejaVuSans.ttf"), _find("LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf")),
    "dejavu": (_find("DejaVuSans.ttf"), _find("DejaVuSans-Bold.ttf")),
    "serif": (_find("LiberationSerif-Regular.ttf", "DejaVuSerif.ttf"), _find("LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf")),
    "mono": (_find("LiberationMono-Regular.ttf", "DejaVuSansMono.ttf"), _find("LiberationMono-Bold.ttf", "DejaVuSansMono-Bold.ttf")),
    "hindi": (_find("NotoSansDevanagari-Regular.ttf"), _find("NotoSansDevanagari-Bold.ttf")),
}
HINDI_OK = bool(FONT_FILES["hindi"][0]) and features.check("raqm")
_fc = {}
def font(fam, size, bold=False):
    key = (fam, size, bold)
    if key not in _fc:
        p = FONT_FILES[fam][1 if bold else 0] or FONT_FILES["dejavu"][1 if bold else 0]
        _fc[key] = ImageFont.truetype(p, size) if p else ImageFont.load_default()
    return _fc[key]

# ------------------------------------------------------------------ fictional data pools
FIRST_M = "Aarav Vivaan Aditya Arjun Rohan Karthik Siddharth Manish Rajesh Suresh Anil Vikram Nikhil Harish Pranav Imran Faisal Gurpreet Tarun Deepak Mohan Ramesh Sanjay Naveen Kunal Yash Dev Ishaan Varun".split()
FIRST_F = "Ananya Diya Kavya Meera Priya Sneha Pooja Neha Anjali Shreya Divya Lakshmi Radhika Sunita Rekha Nandini Isha Tanvi Mansi Farah Simran Harleen Swati Aditi Bhavna Kiran Jyoti Pallavi Rashmi Sana".split()
LAST = "Sharma Verma Iyer Nair Reddy Patel Shah Mehta Kulkarni Deshpande Joshi Gupta Agarwal Menon Pillai Naidu Rao Khan Ansari Singh Kaur Chatterjee Banerjee Das Bose Mishra Pandey Tiwari Yadav Thakur Chopra Malhotra Saxena Bhat Hegde Shetty Gowda".split()
CITIES = [("Bengaluru", "Karnataka", "56"), ("Pune", "Maharashtra", "41"), ("Hyderabad", "Telangana", "50"), ("Chennai", "Tamil Nadu", "60"), ("Mumbai", "Maharashtra", "40"), ("Delhi", "Delhi", "11"),
          ("Ahmedabad", "Gujarat", "38"), ("Kolkata", "West Bengal", "70"), ("Jaipur", "Rajasthan", "30"), ("Lucknow", "Uttar Pradesh", "22"), ("Kochi", "Kerala", "68"), ("Indore", "Madhya Pradesh", "45"),
          ("Nagpur", "Maharashtra", "44"), ("Chandigarh", "Chandigarh", "16"), ("Coimbatore", "Tamil Nadu", "64"), ("Bhopal", "Madhya Pradesh", "46"), ("Visakhapatnam", "Andhra Pradesh", "53"), ("Surat", "Gujarat", "39")]
STATE_CODE = {"Karnataka": "29", "Maharashtra": "27", "Telangana": "36", "Tamil Nadu": "33", "Delhi": "07", "Gujarat": "24", "West Bengal": "19", "Rajasthan": "08", "Uttar Pradesh": "09", "Kerala": "32",
              "Madhya Pradesh": "23", "Chandigarh": "04", "Andhra Pradesh": "37"}
AREAS = "Lakeview Gandhi Nagar|Green Park Extension|Sector 21|MG Road|Park Street Annexe|Whitefield Main Road|Banjara Heights|Model Town|Rose Garden Colony|Civil Lines|Anna Salai Cross|Kalyani Nagar|Salt Lake Block C|Vastrapur|Koregaon Lane|Indiranagar 2nd Stage|Hill View Layout|Residency Road".split("|")
BUILD = "Sai Residency|Shanti Apartments|Orchid Heights|Maple Court|Lotus Towers|Evergreen Villa|Sunrise Enclave|Palm Grove|Silver Oak Residency|Blue Ridge Homes|Cedar Heights|Anand Niwas".split("|")
ADJ = "Greenfield Lakeview Silverstone Maple Orchid Riverdale Horizon Lotus Evergreen Cedar Sapphire Meadow Brookside Crestwood Hillcrest Sunvalley Northstar Elmwood Willow Amber Coral Falcon Granite Juniper Kestrel Linden Mosaic Nimbus Opal Pioneer Quartz".split()
NOUN = "Vista Crest Ridge Haven Grove Gate Point Bridge Field Peak Cove Terrace Meadows Springs Harbour Summit".split()
DENY = ["hdfc", "icici", "sbi", "state bank", "axis", "kotak", "lic", "life insurance corporation", "star health", "bajaj", "tata", "sodexo", "pluxee", "zaggle", "ola", "ather", "mahindra", "maruti", "hero",
        "reliance", "airtel", "jio", "paytm", "phonepe", "amazon", "flipkart", "swiggy", "zomato", "dps", "delhi public", "kendriya", "dav", "ryan", "narayana", "aditya birla", "max bupa", "niva", "care health",
        "sbi life", "pnb", "canara", "union bank", "bank of baroda", "indian bank", "idfc", "yes bank", "ge", "infosys", "wipro", "tcs", "accenture", "cognizant", "hcl", "tesla", "byd", "mg motor"]

def clean(name):
    low = " " + name.lower() + " "
    bad = [d for d in DENY if f" {d} " in low or low.strip().startswith(d + " ")]
    assert not bad, f"deny-listed real brand in generated name {name!r}: {bad}"
    return name
def person(r, g=None):
    g = g or r.choice("MF"); return f"{r.choice(FIRST_M if g == 'M' else FIRST_F)} {r.choice(LAST)}", g
def pan(r, holder="P"): return "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3)) + holder + r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") + f"{r.randrange(10000):04d}" + r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
def tan(r): return "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(4)) + f"{r.randrange(100000):05d}"[:5] + r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
def gstin(r, state):
    body = STATE_CODE.get(state, "27") + pan(r, r.choice("CF")) + str(r.randrange(1, 9)) + "Z"
    alpha = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; tot = 0
    for i, c in enumerate(body):
        v = alpha.index(c) * (1 if i % 2 == 0 else 2); tot += v // 36 + v % 36
    return body + alpha[(36 - tot % 36) % 36]
def address(r, city=None):
    c = city or r.choice(CITIES); return f"{r.randrange(1, 480)}, {r.choice(BUILD)}, {r.choice(AREAS)}, {c[0]} - {c[2]}{r.randrange(10000):04d}", c
def org(r, kind):
    suf = {"school": ["Public School", "International School", "High School", "Vidyalaya", "Academy", "Model School"], "insurer": ["General Insurance Co. Ltd.", "Health Assurance Ltd.", "Life Insurance Ltd.", "Insurance Company Ltd."],
           "bank": ["Co-operative Bank Ltd.", "Finance Bank Ltd.", "Housing Finance Ltd.", "Small Finance Bank"], "meal": ["MealPass India Pvt. Ltd.", "FoodCard Services Pvt. Ltd.", "Bhojan Benefits Pvt. Ltd.", "TiffinPay Solutions Pvt. Ltd."],
           "ev_dealer": ["Electric Motors", "EV Mobility Pvt. Ltd.", "Auto Dealers LLP", "Green Wheels Pvt. Ltd."], "ev_brand": ["Motors", "Mobility", "Electric", "EV Works"], "charger": ["ChargeGrid Pvt. Ltd.", "PlugPoint Energy Ltd.", "VoltStop Networks Pvt. Ltd.", "EcoCharge Services Pvt. Ltd."],
           "employer": ["Technologies Pvt. Ltd.", "Solutions Pvt. Ltd.", "Systems Ltd.", "Consulting LLP", "Industries Ltd.", "Analytics Pvt. Ltd."], "amc": ["Asset Management Co. Ltd.", "Mutual Fund", "Capital Funds Ltd."]}[kind]
    return clean(f"{r.choice(ADJ)} {r.choice(NOUN)} {r.choice(suf)}" if kind not in ("meal", "charger") else f"{r.choice(ADJ)}{r.choice(NOUN)} {r.choice(suf)}")
def indian_fy(r):
    y = r.choice([2023, 2024, 2025, 2026]); return y, f"{y}-{str(y + 1)[2:]}"
def rand_date(r, start, end):
    return start + dt.timedelta(days=r.randrange((end - start).days + 1))
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
def fmt_date(d, r):
    return r.choice([d.strftime("%d/%m/%Y"), d.strftime("%d-%m-%Y"), d.strftime("%d %b %Y"), d.strftime("%d-%b-%y"), d.strftime("%d.%m.%Y")])

def inr(x, dec=2):
    s = f"{x:.{dec}f}"; ip, fp = (s.split(".") + [""])[:2]
    if len(ip) > 3:
        head, tail = ip[:-3], ip[-3:]; g = []
        while len(head) > 2: g.insert(0, head[-2:]); head = head[:-2]
        if head: g.insert(0, head)
        ip = ",".join(g + [tail])
    return ip + ("." + fp if dec else "")
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(); TENS = "x x twenty thirty forty fifty sixty seventy eighty ninety".split()
def _w(n):
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000: return ONES[n // 100] + " hundred" + (" and " + _w(n % 100) if n % 100 else "")
    for div, nm in ((10 ** 7, "crore"), (10 ** 5, "lakh"), (1000, "thousand")):
        if n >= div: return _w(n // div) + f" {nm}" + (" " + _w(n % div) if n % div else "")
def words_inr(n): return ("Rupees " + _w(int(n)).title() + " Only")

# ------------------------------------------------------------------ page + renderer
class Page:
    def __init__(self, w=W, h=H, bg=(255, 255, 255)):
        self.img = Image.new("RGB", (w, h), bg); self.d = ImageDraw.Draw(self.img); self.w, self.h = w, h; self.lines = []
    def text(self, xy, s, f, fill=(25, 25, 25), anchor="la"):
        self.d.text(xy, s, font=f, fill=fill, anchor=anchor); b = self.d.textbbox(xy, s, font=f, anchor=anchor); self.lines.append(dict(text=s, bbox=[int(b[0]), int(b[1]), int(b[2]), int(b[3])]))
        return b
class Style:
    def __init__(self, r, layout):
        self.layout = layout; self.r = r
        fam = {"A": r.choice(["serif", "sans"]), "B": r.choice(["sans", "dejavu"]), "C": "mono", "D": r.choice(["sans", "serif"])}[layout]
        self.fam = fam; base = {"A": 27, "B": 26, "C": 24, "D": 25}[layout] + r.choice([-2, -1, 0, 1, 2])
        self.fs = base; self.accent = r.choice([(24, 62, 120), (120, 30, 30), (20, 90, 60), (70, 40, 110), (30, 30, 30), (150, 80, 10)]); self.tint = tuple(min(255, int(c * 0.15 + 218)) for c in self.accent)
        self.margin = 90 if layout != "C" else 250; self.pw = W - 2 * self.margin
    def f(self, k="n", bold=False):
        sz = {"n": self.fs, "s": self.fs - 4, "h": self.fs + 14, "t": self.fs + 8, "xs": self.fs - 7}[k]; return font(self.fam, sz, bold)
class Doc:
    """collects pages; elements are drawn through the layout-specific methods; paginates automatically"""
    def __init__(self, st, issuer, header_lines, title, hi_title=None, doc_no=None):
        self.st, self.issuer, self.hl, self.title, self.hi = st, issuer, header_lines, title, hi_title; self.pages = []; self.doc_no = doc_no; self.new_page(first=True)
    def new_page(self, first=False):
        st = self.st; p = Page(h=(H if st.layout != "C" else H)); self.pages.append(p); self.p = p; self.y = 70
        if first: self.header()
        else:
            hdr = f"{self.issuer}  |  {self.title}  (contd.)"; f_ = st.f("xs")
            while f_.getlength(hdr) > st.pw and len(hdr) > 12: hdr = hdr[:-5] + "..."      # never run off the page
            p.text((st.margin, self.y), hdr, f_, (90, 90, 90)); self.y += 60; p.d.line([(st.margin, self.y), (W - st.margin, self.y)], fill=(120, 120, 120), width=2); self.y += 25
    def header(self):
        st, p = self.st, self.p; m = st.margin; L = st.layout; r = st.r
        if L == "A":
            p.text((W // 2, self.y), self.issuer, st.f("h", True), st.accent, "ma"); self.y += st.fs + 26
            for ln in self.hl: p.text((W // 2, self.y), ln, st.f("s"), (70, 70, 70), "ma"); self.y += st.fs + 2
            self.y += 12; p.d.line([(m, self.y), (W - m, self.y)], fill=st.accent, width=4); self.y += 30
            b = p.text((W // 2, self.y), self.title, st.f("t", True), (20, 20, 20), "ma"); p.d.line([(b[0], b[3] + 3), (b[2], b[3] + 3)], fill=(20, 20, 20), width=2); self.y = b[3] + 30
        elif L == "B":
            p.d.ellipse([m, self.y, m + 90, self.y + 90], fill=st.accent); p.text((m + 45, self.y + 45), "".join(w[0] for w in self.issuer.split()[:2]).upper(), st.f("t", True), (255, 255, 255), "mm")
            p.text((m + 115, self.y + 5), self.issuer, st.f("h", True), st.accent)
            yy = self.y + 5 + st.fs + 22
            for ln in self.hl[:2]: p.text((m + 115, yy), ln, st.f("s"), (80, 80, 80)); yy += st.fs
            self.y = max(self.y + 100, yy) + 18; p.d.rectangle([m, self.y, W - m, self.y + 62], fill=st.accent); p.text((m + 22, self.y + 31), self.title.upper(), st.f("t", True), (255, 255, 255), "lm"); self.y += 88
        elif L == "C":
            for ln in [self.issuer] + self.hl[:2]: p.text((W // 2, self.y), ln, st.f("n", ln == self.issuer), (20, 20, 20), "ma"); self.y += st.fs + 6
            self.y += 6; p.text((W // 2, self.y), "-" * 38, st.f("n"), (60, 60, 60), "ma"); self.y += st.fs + 8; p.text((W // 2, self.y), self.title.upper(), st.f("n", True), (20, 20, 20), "ma"); self.y += st.fs + 8
            p.text((W // 2, self.y), "-" * 38, st.f("n"), (60, 60, 60), "ma"); self.y += st.fs + 18
        else:  # D boxed form
            p.d.rectangle([m, self.y, W - m, self.y + 150 + (50 if self.hi and HINDI_OK else 0)], outline=(30, 30, 30), width=3)
            p.text((W // 2, self.y + 14), self.issuer, st.f("t", True), st.accent, "ma"); yy = self.y + 14 + st.fs + 16
            for ln in self.hl[:2]: p.text((W // 2, yy), ln, st.f("xs"), (70, 70, 70), "ma"); yy += st.fs - 5
            if self.hi and HINDI_OK: p.text((W // 2, yy + 6), self.hi, font("hindi", st.fs + 6, True), (20, 20, 20), "ma"); yy += st.fs + 18
            p.text((W // 2, yy + 8), self.title.upper(), st.f("n", True), (20, 20, 20), "ma"); self.y += 190 + (50 if self.hi and HINDI_OK else 0)
    def need(self, h):
        if self.y + h > H - 150: self.new_page()
    def kv(self, rows, cols=1):
        st, m, L = self.st, self.st.margin, self.st.layout
        for i, (k, v) in enumerate(rows):
            self.need(st.fs + 22); rh = st.fs + 20
            if L == "A": self.p.text((m, self.y), f"{k}", st.f("n", True)); self.p.text((m + 330, self.y), f": {v}", st.f("n")); self.y += rh
            elif L == "B":
                if i % 2 == 0: self.p.d.rectangle([m, self.y - 6, W - m, self.y + rh - 6], fill=st.tint)
                self.p.text((m + 14, self.y), k, st.f("n", True), st.accent); self.p.text((m + 420, self.y), str(v), st.f("n")); self.y += rh
            elif L == "C":
                lab = f"{k}"; val = str(v); wl = st.f("n").getlength(lab); wv = st.f("n").getlength(val); dots = max(2, int((st.pw - wl - wv) / st.f("n").getlength(".")) - 1)
                self.p.text((m, self.y), lab + "." * dots + val, st.f("n")); self.y += st.fs + 10
            else:
                self.p.d.rectangle([m, self.y, m + 380, self.y + rh + 4], fill=(232, 232, 232), outline=(40, 40, 40), width=2); self.p.d.rectangle([m + 380, self.y, W - m, self.y + rh + 4], outline=(40, 40, 40), width=2)
                self.p.text((m + 12, self.y + 8), k, st.f("s", True)); self.p.text((m + 396, self.y + 8), str(v), st.f("n")); self.y += rh + 4
        self.y += 18
    def para(self, s, bold=False, indent=0, gap=14):
        st = self.st; f = st.f("n", bold) if st.layout != "C" else st.f("s"); width = st.pw - indent
        words = s.split(); line = ""
        for w_ in words:
            t = (line + " " + w_).strip()
            if f.getlength(t) > width: self.need(st.fs + 12); self.p.text((st.margin + indent, self.y), line, f); self.y += st.fs + 10; line = w_
            else: line = t
        if line: self.need(st.fs + 12); self.p.text((st.margin + indent, self.y), line, f); self.y += st.fs + 10
        self.y += gap
    def table(self, headers, rows, widths, align=None, total_row=None):
        st, m, L = self.st, self.st.margin, self.st.layout; tw = st.pw; xs = [m]; align = align or ["l"] * len(headers)
        for w_ in widths: xs.append(xs[-1] + int(tw * w_))
        f = st.f("s") if L != "C" else st.f("xs"); fb = st.f("s", True) if L != "C" else st.f("xs", True); rh = st.fs + 14
        def row(cells, y, bold=False, fill=None):
            if L == "C":
                x = m
                for j, c in enumerate(cells): self.p.text((xs[j], y + 4), str(c), fb if bold else f, anchor="la") if align[j] == "l" else self.p.text((xs[j + 1] - 8, y + 4), str(c), fb if bold else f, anchor="ra")
                return
            if fill: self.p.d.rectangle([xs[0], y, xs[-1], y + rh], fill=fill)
            for j, c in enumerate(cells):
                if L in ("A", "D"): self.p.d.rectangle([xs[j], y, xs[j + 1], y + rh], outline=(60, 60, 60), width=2 if L == "D" else 1)
                tx = xs[j] + 8 if align[j] == "l" else xs[j + 1] - 8; self.p.text((tx, y + 6), str(c), fb if bold else f, anchor="la" if align[j] == "l" else "ra")
            if L == "B": self.p.d.line([(xs[0], y + rh), (xs[-1], y + rh)], fill=(190, 190, 190), width=1)
        self.need(rh * 3); row(headers, self.y, True, st.tint if L != "C" else None); self.y += rh + (4 if L == "C" else 0)
        if L == "C": self.p.text((m, self.y - 4), "-" * 46, st.f("xs"), (60, 60, 60)); self.y += 6
        for rw in rows:
            if self.y + rh > H - 150: self.new_page(); row(headers, self.y, True, st.tint if L != "C" else None); self.y += rh
            row(rw, self.y); self.y += rh
        if total_row: self.need(rh + 8); row(total_row, self.y, True, st.tint if L != "C" else None); self.y += rh
        self.y += 20
    def sign(self, labels):
        st = self.st; r = st.r; n = len(labels); lf = st.f("s")
        stack = n > 1 and (st.layout == "C" or max(lf.getlength(lb) for lb in labels) > st.pw / n - 40)     # narrow layouts / long labels: stack vertically so labels never collide
        if stack:
            for lb in labels:
                self.need(150); y = self.y + 70; x = st.margin; pts = [(x + k * 14, y - 30 + r.randrange(-18, 18)) for k in range(10)]; self.p.d.line(pts, fill=(25, 40, 110), width=3, joint="curve")
                self.p.d.line([(x, y + 10), (x + min(330, st.pw - 20), y + 10)], fill=(60, 60, 60), width=2); self.p.text((x, y + 18), lb, lf, (60, 60, 60)); self.y += 130
            return
        self.need(150); y = self.y + 70
        for i, lb in enumerate(labels):
            x = st.margin + int(i * st.pw / n) + (0 if n > 1 else st.pw - 380)
            pts = [(x + k * 14, y - 30 + r.randrange(-18, 18)) for k in range(10)]; self.p.d.line(pts, fill=(25, 40, 110), width=3, joint="curve")
            self.p.d.line([(x, y + 10), (x + 330, y + 10)], fill=(60, 60, 60), width=2); self.p.text((x, y + 18), lb, lf, (60, 60, 60))
        self.y += 140
    def stamp(self, s, x_frac=0.7):
        st = self.st; self.need(110); f = st.f("s", True); wb = f.getlength(s) + 36
        x, y = st.margin + int(st.pw * x_frac) - wb // 2, self.y; self.p.d.rounded_rectangle([x, y, x + wb, y + 70], 10, outline=(170, 30, 30), width=4); self.p.text((x + wb // 2, y + 35), s, f, (170, 30, 30), "mm"); self.y += 95
    def note(self, s): self.para(s, gap=6)
    def finish(self):
        for i, p in enumerate(self.pages):
            p.text((W // 2, H - 60), FOOTER, font("sans", 17), (150, 150, 150), "ma")
            if len(self.pages) > 1: p.text((W - self.st.margin, H - 100), f"Page {i + 1} of {len(self.pages)}", self.st.f("xs"), (90, 90, 90), "ra")
        return self.pages

# ------------------------------------------------------------------ document builders (structured record -> sections)
def money(r, lo, hi, step=1): return round(r.randrange(int(lo / step), int(hi / step) + 1) * step, 2)
def rs(a, r, fam=None): return f"₹ {inr(a)}" if fam == "dejavu" and r.random() < 0.6 else f"Rs. {inr(a)}"      # ₹ only with the font that has the glyph

def b_rent_receipt(r, st, ctx):
    ll, g = person(r); tn, _ = person(r); addr, c = address(r); y, fy = indian_fy(r); mi = r.randrange(12); sub = r.choice(["single_month"] * 3 + ["multi_month"]); rent = money(r, 6000, 85000, 500); n = 1 if sub == "single_month" else 3
    mode = r.choice(["Cash", "UPI", "Bank Transfer (NEFT)", "Cheque", "Cash", "UPI"]); amt = rent * n; ann = rent * 12; has_pan = ann > 100000 or r.random() < 0.3
    d1 = dt.date(y, mi + 1, 1); end_m = (mi + n - 1) % 12; period = f"{MONTHS[mi]} {y}" if n == 1 else f"{MONTHS[mi]} {y} to {MONTHS[end_m]} {y + (mi + n - 1) // 12}"; rd = d1 + dt.timedelta(days=r.randrange(0, 12 if n == 1 else 5))
    no = f"RR/{y}/{r.randrange(1, 9999):04d}"; issuer = ll
    fields = dict(landlord_name=ll, landlord_pan=pan(r) if has_pan else None, tenant_name=tn, property_address=addr, rent_amount=amt, monthly_rent=rent, months=n, period=period, payment_mode=mode, receipt_no=no, receipt_date=rd.isoformat(), financial_year=f"{y}-{str(y + 1)[2:]}" if mi >= 3 else f"{y - 1}-{str(y)[2:]}")
    def body(D):
        D.para(f"Received with thanks from {tn} a sum of {rs(amt, r, st.fam)} ({words_inr(amt)}) towards the rent of the residential premises situated at {addr} for the period {period}.")
        D.kv([("Receipt No.", no), ("Date", fmt_date(rd, r)), ("Tenant", tn), ("Landlord", ll)] + ([("Landlord PAN", fields["landlord_pan"])] if has_pan else []) + [("Monthly Rent", f"{inr(rent)}"), ("No. of months", str(n)), ("Amount Received", inr(amt)), ("Payment mode", mode), ("Period", period)])
        if mode == "Cash" and amt > 5000: D.stamp("Revenue Stamp  Rs. 1")
        D.sign(["Landlord's signature"])
    return dict(title="RENT RECEIPT", hi="किराया रसीद", issuer_lines=[addr], header_issuer=f"{ll} - Landlord", body=body, subtype=sub, fields=fields, key_phrases=["rent receipt", "rent", "landlord"], issuer=ll)

def b_rental_agreement(r, st, ctx):
    ll, _ = person(r); tn, _ = person(r); addr, c = address(r); y, fy = indian_fy(r); rent = money(r, 8000, 90000, 500); dep = rent * r.choice([2, 3, 6, 10]); d0 = dt.date(y, r.randrange(1, 13), 1); d1 = d0 + dt.timedelta(days=334)
    title = r.choice(["RENT AGREEMENT", "LEAVE AND LICENSE AGREEMENT", "RESIDENTIAL LEASE AGREEMENT"]); w1, w2 = person(r)[0], person(r)[0]; notice = r.choice([1, 2]); lock = r.choice([3, 6])
    fields = dict(landlord_name=ll, tenant_name=tn, property_address=addr, monthly_rent=rent, security_deposit=dep, start_date=d0.isoformat(), end_date=d1.isoformat(), term_months=11, notice_months=notice, city=c[0])
    def body(D):
        D.para(f"This {title.title()} is made at {c[0]} on {fmt_date(d0 - dt.timedelta(days=r.randrange(1, 9)), r)} BETWEEN {ll}, residing at {address(r, c)[0]} (hereinafter called the LESSOR/LICENSOR) AND {tn}, presently residing at {address(r, c)[0]} (hereinafter called the LESSEE/LICENSEE).")
        D.para(f"WHEREAS the Lessor is the lawful owner of the residential premises at {addr} and has agreed to let the same to the Lessee on the following terms and conditions:", gap=10)
        cl = [f"The tenancy shall be for a period of 11 (eleven) months commencing from {fmt_date(d0, r)} and ending on {fmt_date(d1, r)}, renewable by mutual consent.",
              f"The Lessee shall pay a monthly rent of Rs. {inr(rent)} ({words_inr(rent)}), payable in advance on or before the 5th day of each calendar month.",
              f"The Lessee has paid an interest-free refundable security deposit of Rs. {inr(dep)} ({words_inr(dep)}) to the Lessor, to be refunded at the time of vacating, subject to deductions for damages and dues.",
              f"Electricity, water and maintenance charges shall be paid by the Lessee as per actual bills. The premises shall be used for residential purposes only.",
              f"The Lessee shall not sublet or part with possession of the premises. Either party may terminate this agreement by giving {notice} month's written notice; a lock-in period of {lock} months applies.",
              "The Lessee shall maintain the premises in good condition and permit the Lessor to inspect it with prior notice. Any dispute shall be subject to the jurisdiction of courts at " + c[0] + "."]
        for i, t in enumerate(cl, 1): D.para(f"{i}.  {t}", indent=10)
        D.para("IN WITNESS WHEREOF the parties have signed this agreement on the date first written above in the presence of the following witnesses.")
        D.sign(["LESSOR", "LESSEE"]); D.sign([f"Witness 1: {w1}", f"Witness 2: {w2}"])
    return dict(title=title, hi="किराया अनुबंध", issuer_lines=[f"{c[0]}, {c[1]}"], header_issuer=f"{title.title()}", body=body, subtype="residential", fields=fields, key_phrases=["agreement", "lessor", "lessee"], issuer=ll, force_pages=2)

def b_insurance(r, st, ctx):
    sub = r.choice(["health", "life"]); ins = org(r, "insurer"); ph, _ = person(r); y, fy = indian_fy(r); prem = money(r, 3500, 68000, 50) if sub == "health" else money(r, 8000, 150000, 100); gst = r.choice([0, 0.18]) if sub == "health" else 0
    gst_amt = round(prem * gst, 2); tot = prem + gst_amt; pol = f"{r.choice('PHL')}{r.randrange(10 ** 9, 10 ** 10)}"; d = rand_date(r, dt.date(y, 4, 1), dt.date(y + 1, 3, 31)); p1 = d; p2 = d.replace(year=d.year + 1) - dt.timedelta(days=1)
    mem = [ph] + [person(r)[0] for _ in range(r.randrange(0, 3))] if sub == "health" else [ph]; mode = r.choice(["Net Banking", "UPI", "Debit Card", "Credit Card", "Cheque", "NACH/ECS"]); no = f"PR{y}{r.randrange(10 ** 7, 10 ** 8)}"; plan = r.choice(["Family Floater", "Individual", "Senior Citizen", "Top-up"]) if sub == "health" else r.choice(["Term Plan", "Endowment Plan", "Money-back Plan", "ULIP"])
    fields = dict(subtype=sub, insurer=ins, policy_no=pol, policyholder=ph, insured_members=mem, plan=plan, premium=prem, gst=gst_amt, total_paid=tot, period_from=p1.isoformat(), period_to=p2.isoformat(), payment_mode=mode, receipt_no=no, receipt_date=d.isoformat())
    def body(D):
        rows = [("Receipt No.", no), ("Receipt Date", fmt_date(d, r)), ("Policy No.", pol), ("Policyholder", ph), ("Plan", plan)] + ([("Insured members", ", ".join(mem))] if sub == "health" else [("Sum Assured", inr(prem * r.randrange(60, 200)))]) + [("Policy period", f"{fmt_date(p1, r)} to {fmt_date(p2, r)}"), ("Premium", inr(prem))] + ([("GST @ 18%", inr(gst_amt))] if gst else []) + [("Total premium paid", inr(tot)), ("Payment mode", mode)]
        D.kv(rows); D.para(f"Received with thanks {rs(tot, r, st.fam)} ({words_inr(tot)}) as premium for the above {'health insurance' if sub == 'health' else 'life insurance'} policy. This is a computer generated receipt and does not require a signature.")
        D.note("This certificate is issued on request of the policyholder for the purpose of income tax records.");
    ttl = "HEALTH INSURANCE PREMIUM RECEIPT" if sub == "health" else "LIFE INSURANCE PREMIUM RECEIPT"
    return dict(title=ttl, hi="प्रीमियम रसीद", issuer_lines=["Regd. Office: " + address(r)[0]], header_issuer=ins, body=body, subtype=sub, fields=fields, key_phrases=["premium", "policy"], issuer=ins)

def b_school(r, st, ctx):
    sch = org(r, "school"); stu, g = person(r); par, _ = person(r, "M"); y, fy = indian_fy(r); cls = r.choice(["Nursery", "LKG", "UKG"] + [f"Class {i}" for i in range(1, 13)]); sec = r.choice("ABCD"); adm = f"{r.randrange(1000, 99999)}"
    heads = [("Tuition Fee", money(r, 8000, 70000, 250))]; opt = [("Development Fee", money(r, 1500, 15000, 250)), ("Transport Fee", money(r, 3000, 24000, 250)), ("Activity Fee", money(r, 500, 6000, 100)), ("Computer Lab Fee", money(r, 500, 4000, 100)), ("Exam Fee", money(r, 500, 3000, 100)), ("Annual Charges", money(r, 1500, 9000, 250)), ("Library Fee", money(r, 300, 2500, 50))]
    sub = r.choice(["tuition_only", "with_other_fees", "with_other_fees"]); heads += [opt[5]] if sub == "tuition_only" and r.random() < 0.4 else ([] if sub == "tuition_only" else r.sample(opt, r.randrange(2, 6)))
    tot = sum(a for _, a in heads); term = r.choice(["Term 1", "Term 2", "Term 3", "Quarter 1", "Quarter 2", "Annual", "April-June", "July-September"]); d = rand_date(r, dt.date(y, 4, 1), dt.date(y + 1, 3, 31)); mode = r.choice(["Cash", "Cheque", "Online", "UPI", "Demand Draft", "Card"]); no = f"{r.choice(['FR', 'REC', 'F'])}-{y % 100}-{r.randrange(100, 99999)}"
    tuition = sum(a for n_, a in heads if n_ == "Tuition Fee")
    fields = dict(school=sch, student_name=stu, parent_name=par, class_=cls, section=sec, admission_no=adm, academic_year=fy, term=term, fee_heads=[dict(head=n_, amount=a) for n_, a in heads], tuition_component=tuition, total=tot, payment_mode=mode, receipt_no=no, receipt_date=d.isoformat())
    def body(D):
        D.kv([("Receipt No.", no), ("Date", fmt_date(d, r)), ("Academic Year", fy), ("Student Name", stu), ("Class / Section", f"{cls} / {sec}"), ("Admission No.", adm), ("Parent Name", par), ("Fee Period", term)])
        D.table(["S.No", "Fee Head", "Amount (Rs.)"], [[i + 1, n_, inr(a)] for i, (n_, a) in enumerate(heads)], [0.12, 0.58, 0.30], ["l", "l", "r"], ["", "TOTAL", inr(tot)])
        D.para(f"Received {rs(tot, r, st.fam)} ({words_inr(tot)}) by {mode}. Fees once paid are non-refundable."); D.sign(["Accountant / Cashier"])
    return dict(title=r.choice(["FEE RECEIPT", "SCHOOL FEE RECEIPT", "FEE PAYMENT RECEIPT"]), hi="शुल्क रसीद", issuer_lines=["Affiliated to a State Board - " + address(r)[0]], header_issuer=sch, body=body, subtype=sub, fields=fields, key_phrases=["fee", "receipt", "tuition"], issuer=sch)

def b_investment(r, st, ctx):
    sub = r.choice(["ppf", "elss", "tax_saver_fd", "savings_certificate", "ssy"]); y, fy = indian_fy(r); hold, _ = person(r); bank = org(r, "bank"); acc = f"{r.randrange(10 ** 10, 10 ** 12)}"
    if sub == "ppf":
        dep = [(rand_date(r, dt.date(y, 4, 1), dt.date(y + 1, 3, 31)), money(r, 5000, 50000, 500)) for _ in range(r.randrange(2, 8))]; dep.sort(); tot = sum(a for _, a in dep)
        fields = dict(subtype=sub, holder=hold, institution=bank, account_no=acc, financial_year=fy, deposits=[dict(date=d.isoformat(), amount=a) for d, a in dep], total_deposit=tot, interest_rate=7.1); ttl = "PUBLIC PROVIDENT FUND (PPF) ACCOUNT STATEMENT"
        def body(D):
            D.kv([("Account Holder", hold), ("PPF Account No.", acc), ("Financial Year", fy), ("Opening date", fmt_date(dt.date(y - r.randrange(1, 9), 5, 1), r))]); D.table(["Date", "Particulars", "Deposit (Rs.)"], [[fmt_date(d, r), "Cash/Online deposit", inr(a)] for d, a in dep], [0.25, 0.45, 0.30], ["l", "l", "r"], ["", "Total deposits in FY " + fy, inr(tot)])
    elif sub == "elss":
        amc = org(r, "amc"); sch = f"{r.choice(ADJ)} Tax Saver Fund (ELSS)"; inv = money(r, 5000, 150000, 500); nav = money(r, 12, 90, 0.01); units = round(inv / nav, 3); d = rand_date(r, dt.date(y, 4, 1), dt.date(y + 1, 3, 31)); folio = f"{r.randrange(10 ** 7, 10 ** 8)}/{r.randrange(10, 99)}"
        fields = dict(subtype=sub, holder=hold, institution=amc, scheme=sch, folio=folio, financial_year=fy, amount_invested=inv, nav=nav, units=units, date=d.isoformat(), lock_in_years=3); ttl = "ELSS MUTUAL FUND - STATEMENT OF ACCOUNT"; bank = amc
        def body(D):
            D.kv([("Investor", hold), ("Folio No.", folio), ("Scheme", sch), ("Financial Year", fy)]); D.table(["Date", "Transaction", "Amount (Rs.)", "NAV", "Units"], [[fmt_date(d, r), "Purchase - SIP/Lump sum", inr(inv), f"{nav:.2f}", f"{units:.3f}"]], [0.2, 0.32, 0.2, 0.12, 0.16], ["l", "l", "r", "r", "r"]); D.note("Units under this scheme carry a statutory lock-in of 3 years from the date of allotment.")
    elif sub == "tax_saver_fd":
        amt = money(r, 10000, 150000, 1000); rate = r.choice([6.5, 6.75, 7.0, 7.25, 7.5]); d = rand_date(r, dt.date(y, 4, 1), dt.date(y + 1, 3, 31)); mat = d.replace(year=d.year + 5); fdr = f"FDR{r.randrange(10 ** 8, 10 ** 9)}"
        fields = dict(subtype=sub, holder=hold, institution=bank, fd_no=fdr, financial_year=fy, deposit_amount=amt, interest_rate=rate, deposit_date=d.isoformat(), maturity_date=mat.isoformat(), tenure_years=5); ttl = "TAX SAVER FIXED DEPOSIT - RECEIPT"
        def body(D): D.kv([("Depositor", hold), ("FD Receipt No.", fdr), ("Deposit Date", fmt_date(d, r)), ("Deposit Amount", inr(amt)), ("Tenure", "5 years (lock-in)"), ("Rate of interest", f"{rate}% p.a."), ("Maturity Date", fmt_date(mat, r))])
    elif sub == "savings_certificate":
        amt = money(r, 1000, 100000, 1000); d = rand_date(r, dt.date(y, 4, 1), dt.date(y + 1, 3, 31)); mat = d.replace(year=d.year + 5); cno = f"{r.choice('ABCD')}{r.randrange(10 ** 5, 10 ** 6)}"; bank = f"{r.choice(ADJ)} Post Office Savings Branch"
        fields = dict(subtype=sub, holder=hold, institution=bank, certificate_no=cno, financial_year=fy, face_value=amt, issue_date=d.isoformat(), maturity_date=mat.isoformat(), tenure_years=5); ttl = "SAVINGS CERTIFICATE (5-YEAR) - HOLDER'S COPY"
        def body(D): D.kv([("Holder", hold), ("Certificate No.", cno), ("Issue date", fmt_date(d, r)), ("Amount invested", inr(amt)), ("Maturity date", fmt_date(mat, r)), ("Branch", bank)])
    else:
        dau, _ = person(r, "F"); amt = money(r, 1000, 150000, 500); d = rand_date(r, dt.date(y, 4, 1), dt.date(y + 1, 3, 31)); sa = f"{r.randrange(10 ** 9, 10 ** 10)}"
        fields = dict(subtype=sub, holder=hold, beneficiary=dau, institution=bank, account_no=sa, financial_year=fy, deposit_amount=amt, deposit_date=d.isoformat()); ttl = "SUKANYA SAMRIDDHI ACCOUNT - DEPOSIT RECEIPT"
        def body(D): D.kv([("Guardian", hold), ("Girl child", dau), ("Account No.", sa), ("Deposit date", fmt_date(d, r)), ("Deposit amount", inr(amt)), ("Financial year", fy)])
    def full(D): body(D); D.note("This statement/receipt is issued for the information of the account holder."); D.sign(["Authorised signatory"])
    return dict(title=ttl, hi="निवेश प्रमाणपत्र", issuer_lines=[address(r)[0]], header_issuer=bank, body=full, subtype=sub, fields=fields, key_phrases={"ppf": ["provident fund", "ppf"], "elss": ["elss", "mutual fund"], "tax_saver_fd": ["fixed deposit"], "savings_certificate": ["savings certificate"], "ssy": ["sukanya"]}[sub], issuer=bank)

def b_meal(r, st, ctx):
    sub = r.choice(["statement", "statement", "invoice"]); prov = org(r, "meal"); emp = org(r, "employer"); ename, _ = person(r); y, fy = indian_fy(r); mi = r.randrange(12); eid = f"E{r.randrange(1000, 99999)}"; card = f"XXXX XXXX XXXX {r.randrange(1000, 9999)}"; per = money(r, 50, 200, 10)
    if sub == "statement":
        days = sorted(r.sample(range(1, 29), r.randrange(8, 20))); rows = [[fmt_date(dt.date(y, mi + 1, dd), r), f"Meal card credit - {r.randrange(1, 3)} coupons"] for dd in days]; cnt = []; tot = 0
        for rw in rows: n_ = int(rw[1].split("- ")[1].split()[0]); cnt.append(n_); rw += [n_, inr(n_ * per)]; tot += n_ * per
        fields = dict(subtype=sub, provider=prov, employer=emp, employee=ename, employee_id=eid, card=card, month=f"{MONTHS[mi]} {y}", coupon_value=per, coupons=sum(cnt), total=tot)
        def body(D): D.kv([("Employee", ename), ("Employee ID", eid), ("Employer", emp), ("Meal card", card), ("Statement period", f"{MONTHS[mi]} {y}"), ("Value per coupon", f"Rs. {per:.2f}")]); D.table(["Date", "Description", "Coupons", "Amount (Rs.)"], rows, [0.2, 0.44, 0.14, 0.22], ["l", "l", "r", "r"], ["", "Total", sum(cnt), inr(tot)])
        ttl = "MEAL CARD STATEMENT"
    else:
        n_emp = r.randrange(20, 600); load = per * 22 * n_emp; fee = round(load * r.choice([0.01, 0.015, 0.02]), 2); gstv = round(fee * 0.18, 2); tot = load + fee + gstv; city = r.choice(CITIES); gs = gstin(r, city[1]); ino = f"MP/{y % 100}-{(y + 1) % 100}/{r.randrange(1, 99999):05d}"; d = dt.date(y, mi + 1, r.randrange(1, 28))
        fields = dict(subtype=sub, provider=prov, provider_gstin=gs, employer=emp, invoice_no=ino, invoice_date=d.isoformat(), meal_card_load=load, service_fee=fee, gst=gstv, total=tot, employees=n_emp)
        def body(D): D.kv([("Invoice No.", ino), ("Invoice Date", fmt_date(d, r)), ("Bill To", emp), ("GSTIN (supplier)", gs), ("Place of supply", city[1])]); D.table(["Description", "SAC", "Amount (Rs.)"], [[f"Meal card load - {n_emp} employees ({MONTHS[mi]} {y})", "9965", inr(load)], ["Service fee", "9985", inr(fee)], ["IGST/CGST+SGST @ 18% on service fee", "", inr(gstv)]], [0.62, 0.12, 0.26], ["l", "l", "r"], ["Invoice total", "", inr(tot)]); D.note("Meal vouchers are usable at food outlets only. E&OE.")
        ttl = "TAX INVOICE - MEAL CARD SERVICES"
    return dict(title=ttl, hi="भोजन वाउचर विवरण", issuer_lines=["Customer care: 1800-" + str(r.randrange(100, 999)) + "-" + str(r.randrange(1000, 9999))], header_issuer=prov, body=body, subtype=sub, fields=fields, key_phrases=["meal", "coupon" if sub == "statement" else "invoice"], issuer=prov)

def b_ev(r, st, ctx):
    sub = r.choice(["purchase_invoice", "charging_receipt", "charging_receipt", "lease_letter"]); y, fy = indian_fy(r); buyer, _ = person(r); city = r.choice(CITIES); model = f"{r.choice(ADJ)} {r.choice(['E1', 'EV', 'Volt', 'Spark', 'Glide'])} {r.choice(['', 'Pro', 'Max', 'Lite'])}".strip()
    if sub == "purchase_invoice":
        dealer = org(r, "ev_dealer"); brand = org(r, "ev_brand"); price = money(r, 70000, 1800000, 500); gstv = round(price * 0.05, 2); rto = money(r, 0, 12000, 100); ins = money(r, 3000, 40000, 100); tot = price + gstv + rto + ins; chas = "".join(r.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") for _ in range(17)); d = rand_date(r, dt.date(y, 4, 1), dt.date(y + 1, 3, 31)); gs = gstin(r, city[1]); ino = f"EV/{y % 100}/{r.randrange(1, 9999):04d}"; kwh = r.choice([2.9, 3.3, 3.7, 26, 30.2, 40.5]); subsidy = r.choice([0, 0, money(r, 5000, 60000, 500)])
        fields = dict(subtype=sub, dealer=dealer, brand=brand, model=model, buyer=buyer, invoice_no=ino, invoice_date=d.isoformat(), gstin=gs, chassis_no=chas, battery_kwh=kwh, ex_showroom=price, gst=gstv, registration=rto, insurance=ins, subsidy=subsidy, total=tot - subsidy)
        def body(D):
            D.kv([("Invoice No.", ino), ("Date", fmt_date(d, r)), ("Customer", buyer), ("Vehicle", f"{brand.split()[0]} {model} (Electric)"), ("Battery capacity", f"{kwh} kWh"), ("Chassis No.", chas), ("Dealer GSTIN", gs)])
            rows = [["Ex-showroom price (HSN 8703/8711)", inr(price)], ["GST @ 5%", inr(gstv)], ["Registration & road tax", inr(rto)], ["Insurance", inr(ins)]] + ([["Subsidy adjusted (as per scheme)", "-" + inr(subsidy)]] if subsidy else [])
            D.table(["Description", "Amount (Rs.)"], rows, [0.7, 0.3], ["l", "r"], ["Total payable", inr(tot - subsidy)]); D.sign(["Authorised signatory"])
        ttl = "TAX INVOICE - ELECTRIC VEHICLE"; issuer = dealer; hdr = [address(r, city)[0]]
    elif sub == "charging_receipt":
        net = org(r, "charger"); stn = f"{r.choice(ADJ)} {r.choice(['Mall', 'Highway', 'Metro', 'Tech Park', 'Plaza'])} Charging Hub"; kwh = round(r.uniform(4, 62), 2); rate = r.choice([12, 14, 16, 18, 20, 22.5]); en = round(kwh * rate, 2); gstv = round(en * 0.18, 2); tot = round(en + gstv, 2)
        d = rand_date(r, dt.date(y, 4, 1), dt.date(y + 1, 3, 31)); sess = f"CS{r.randrange(10 ** 8, 10 ** 9)}"; veh = f"{r.choice(['KA', 'MH', 'TN', 'DL', 'TS', 'GJ'])}{r.randrange(1, 99):02d}{r.choice(['AB', 'EV', 'CD'])}{r.randrange(1000, 9999)}"; pm = r.choice(["UPI", "Card", "Wallet"])
        fields = dict(subtype=sub, operator=net, station=stn, session_id=sess, vehicle_no=veh, energy_kwh=kwh, rate_per_kwh=rate, energy_charge=en, gst=gstv, total=tot, date=d.isoformat(), payment_mode=pm)
        def body(D): D.kv([("Session ID", sess), ("Date & time", fmt_date(d, r) + f" {r.randrange(0, 24):02d}:{r.randrange(0, 60):02d}"), ("Station", stn), ("Vehicle No.", veh), ("Energy delivered", f"{kwh} kWh"), ("Rate", f"Rs. {rate}/kWh"), ("Energy charge", inr(en)), ("GST @ 18%", inr(gstv)), ("Total", inr(tot)), ("Paid via", pm)])
        ttl = "EV CHARGING RECEIPT"; issuer = net; hdr = [stn, address(r, city)[0]]
    else:
        emp = org(r, "employer"); rental = money(r, 8000, 40000, 500); ten = r.choice([36, 48, 60]); d = rand_date(r, dt.date(y, 4, 1), dt.date(y + 1, 3, 31)); ref = f"HR/EVL/{y}/{r.randrange(1, 999):03d}"
        fields = dict(subtype=sub, employer=emp, employee=buyer, vehicle=model, monthly_lease_rental=rental, tenure_months=ten, letter_date=d.isoformat(), reference=ref)
        def body(D):
            D.kv([("Ref. No.", ref), ("Date", fmt_date(d, r)), ("Employee", buyer)]); D.para(f"Dear {buyer.split()[0]}, as per the Company's Electric Vehicle Lease Policy, you have opted for a leased electric vehicle ({model}). The monthly lease rental of Rs. {inr(rental)} for a tenure of {ten} months will be recovered through salary as per the policy. This letter is for your records; the tax treatment of the arrangement is to be determined by the payroll team in accordance with applicable rules.")
            D.sign(["For the Company - HR"])
        ttl = "EV LEASE POLICY - EMPLOYEE LETTER"; issuer = emp; hdr = [address(r, city)[0]]
    return dict(title=ttl, hi="इलेक्ट्रिक वाहन", issuer_lines=hdr, header_issuer=issuer, body=body, subtype=sub, fields=fields, key_phrases=["electric", "ev", "charging"] if sub != "lease_letter" else ["lease", "electric"], issuer=issuer)

def amort(P, rate, n):
    i = rate / 1200; emi = P * i * (1 + i) ** n / ((1 + i) ** n - 1); bal = P; out = []
    for k in range(n): intr = bal * i; pr = emi - intr; bal -= pr; out.append((emi, pr, intr, bal))
    return emi, out
def b_loan(r, st, ctx, statement=False):
    bank = org(r, "bank"); bw, _ = person(r); cb = person(r)[0] if r.random() < 0.5 else None; addr, c = address(r); y, fy = indian_fy(r); P = money(r, 800000, 12000000, 50000); rate = r.choice([8.35, 8.5, 8.75, 9.0, 9.25, 9.5]); n = r.choice([120, 180, 240, 300]); emi, sched = amort(P, rate, n)
    sanc = dt.date(y - r.randrange(1, 6), r.randrange(1, 13), r.randrange(1, 28)); acc = f"HL{r.randrange(10 ** 9, 10 ** 10)}"; k0 = max(0, min(n - 12, ((y - sanc.year) * 12 + (4 - sanc.month)) % (n - 12) if n > 24 else 0)); yr = sched[k0:k0 + 12]
    pr_y = round(sum(x[1] for x in yr), 2); in_y = round(sum(x[2] for x in yr), 2); bal = round(yr[-1][3], 2)
    if not statement:
        sub = r.choice(["summary", "with_schedule"]); fields = dict(subtype=sub, lender=bank, borrower=bw, co_borrower=cb, loan_account_no=acc, property_address=addr, sanction_date=sanc.isoformat(), sanctioned_amount=P, rate=rate, emi=round(emi, 2), financial_year=fy, principal_repaid=pr_y, interest_paid=in_y, outstanding_principal=bal)
        def body(D):
            D.kv([("Borrower", bw)] + ([("Co-borrower", cb)] if cb else []) + [("Loan Account No.", acc), ("Loan type", "Housing Loan"), ("Property", addr), ("Date of sanction", fmt_date(sanc, r)), ("Sanctioned amount", inr(P)), ("Rate of interest", f"{rate}% p.a."), ("EMI", inr(emi))])
            D.para(f"This is to certify that the following amounts were paid / are payable by the borrower(s) in respect of the above housing loan during the financial year {fy}:", gap=8)
            D.kv([("Principal repaid", inr(pr_y)), ("Interest paid / payable", inr(in_y)), ("Total EMI paid", inr(pr_y + in_y)), ("Outstanding principal at year end", inr(bal))])
            if sub == "with_schedule": D.table(["Month", "EMI", "Principal", "Interest"], [[MONTHS[(3 + j) % 12][:3] + " " + str(y + (3 + j) // 12), inr(x[0]), inr(x[1]), inr(x[2])] for j, x in enumerate(yr)], [0.22, 0.26, 0.26, 0.26], ["l", "r", "r", "r"], ["Total", inr(pr_y + in_y), inr(pr_y), inr(in_y)])
            D.note("This certificate is issued at the request of the borrower for the purpose of income tax records. It does not constitute a statement of tax eligibility."); D.sign(["Authorised signatory"])
        return dict(title=r.choice(["HOUSING LOAN INTEREST CERTIFICATE", "PROVISIONAL INTEREST CERTIFICATE", "CERTIFICATE OF INTEREST AND PRINCIPAL REPAID"]), hi="ऋण ब्याज प्रमाणपत्र", issuer_lines=["Regd. Office: " + address(r)[0]], header_issuer=bank, body=body, subtype=sub, fields=fields, key_phrases=["interest certificate", "loan", "principal"], issuer=bank)
    nrows = r.randrange(12, 26); k1 = max(0, min(n - nrows, r.randrange(0, 40))); rows_ = sched[k1:k1 + nrows]; d0 = dt.date(y, r.randrange(1, 13), 5)
    fields = dict(subtype="account_statement", lender=bank, borrower=bw, loan_account_no=acc, sanctioned_amount=P, rate=rate, emi=round(emi, 2), opening_balance=round(rows_[0][3] + rows_[0][1], 2), closing_balance=round(rows_[-1][3], 2), rows=len(rows_), total_principal=round(sum(x[1] for x in rows_), 2), total_interest=round(sum(x[2] for x in rows_), 2))
    def body(D):
        D.kv([("Borrower", bw), ("Loan Account No.", acc), ("Loan type", "Housing Loan"), ("Sanctioned amount", inr(P)), ("Rate of interest", f"{rate}% p.a."), ("Opening principal outstanding", inr(rows_[0][3] + rows_[0][1]))])
        tr = []
        for j, x in enumerate(rows_): m_ = d0.month - 1 + j; tr.append([fmt_date(dt.date(d0.year + m_ // 12, m_ % 12 + 1, 5), r), "EMI debited", inr(x[0]), inr(x[1]), inr(x[2]), inr(x[3])])
        D.table(["Date", "Narration", "EMI", "Principal", "Interest", "Balance"], tr, [0.16, 0.2, 0.15, 0.16, 0.15, 0.18], ["l", "l", "r", "r", "r", "r"], ["Total", "", inr(sum(x[0] for x in rows_)), inr(fields["total_principal"]), inr(fields["total_interest"]), inr(fields["closing_balance"])])
    return dict(title="LOAN ACCOUNT STATEMENT", hi="ऋण खाता विवरण", issuer_lines=["Regd. Office: " + address(r)[0]], header_issuer=bank, body=body, subtype="account_statement", fields=fields, key_phrases=["loan account statement", "principal", "interest"], issuer=bank)
def b_loan_stmt(r, st, ctx): return b_loan(r, st, ctx, statement=True)

def b_salary(r, st, ctx):
    emp = org(r, "employer"); en, g = person(r); y, fy = indian_fy(r); mi = r.randrange(12); basic = money(r, 18000, 190000, 500); hra = round(basic * r.choice([0.4, 0.5]), 0); spl = money(r, 5000, 90000, 500); conv = r.choice([0, 1600, 2000]); bonus = r.choice([0, 0, money(r, 2000, 30000, 500)])
    gross = basic + hra + spl + conv + bonus; pf = round(basic * 0.12); pt = r.choice([0, 200, 200, 208]); tds = r.choice([0, 0, money(r, 1000, 25000, 100)]); ded = pf + pt + tds; net = gross - ded; eid = f"EMP{r.randrange(1000, 99999)}"; pn = pan(r); days = r.choice([28, 30, 31]); lop = r.choice([0, 0, 0, 1, 2])
    fields = dict(employer=emp, employee=en, employee_id=eid, pan=pn, month=f"{MONTHS[mi]} {y}", designation=r.choice(["Software Engineer", "Analyst", "Manager", "Associate", "Senior Executive", "Accountant", "Consultant"]), earnings=dict(basic=basic, hra=hra, special_allowance=spl, conveyance=conv, bonus=bonus), gross=gross, deductions=dict(provident_fund=pf, professional_tax=pt, tds=tds), total_deductions=ded, net_pay=net, days=days, lop_days=lop)
    def body(D):
        D.kv([("Employee Name", en), ("Employee ID", eid), ("Designation", fields["designation"]), ("PAN", pn), ("Pay period", f"{MONTHS[mi]} {y}"), ("Days paid", f"{days - lop} / {days}")])
        er = [["Basic Salary", inr(basic)], ["House Rent Allowance", inr(hra)], ["Special Allowance", inr(spl)]] + ([["Conveyance Allowance", inr(conv)]] if conv else []) + ([["Performance Bonus", inr(bonus)]] if bonus else [])
        D.table(["Earnings", "Amount (Rs.)"], er, [0.65, 0.35], ["l", "r"], ["Gross Earnings", inr(gross)])
        dd = [["Provident Fund", inr(pf)]] + ([["Professional Tax", inr(pt)]] if pt else []) + ([["Income Tax (TDS)", inr(tds)]] if tds else [])
        D.table(["Deductions", "Amount (Rs.)"], dd, [0.65, 0.35], ["l", "r"], ["Total Deductions", inr(ded)]); D.para(f"Net Pay: Rs. {inr(net)} ({words_inr(net)})", bold=True); D.note("This is a system-generated payslip and does not require a signature.")
    return dict(title=r.choice(["PAYSLIP", "SALARY SLIP", "PAY SLIP FOR THE MONTH"]), hi="वेतन पर्ची", issuer_lines=[address(r)[0]], header_issuer=emp, body=body, subtype="monthly", fields=fields, key_phrases=["salary", "payslip", "pay slip", "net pay"], issuer=emp)

def b_form16(r, st, ctx):
    emp = org(r, "employer"); en, _ = person(r); y, fy = indian_fy(r); ay = f"{y + 1}-{str(y + 2)[2:]}"; basic = money(r, 300000, 2400000, 1000); hra_ex = round(basic * r.choice([0, 0.1, 0.2])); std = 50000 if y < 2024 else 75000; c80 = r.choice([0, 50000, 100000, 150000]); d80 = r.choice([0, 15000, 25000, 50000]); gross = basic; taxable = max(0, gross - hra_ex - std - c80 - d80)
    def slab(t): return max(0, min(t, 500000) - 250000) * 0.05 + max(0, min(t, 1000000) - 500000) * 0.2 + max(0, t - 1000000) * 0.3
    tax = round(slab(taxable)); cess = round(tax * 0.04); tot_tax = tax + cess; q = [round(tot_tax / 4)] * 3 + [tot_tax - 3 * round(tot_tax / 4)]; tn_, pn_, pe = tan(r), pan(r, "C"), pan(r); cert = "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(7))
    fields = dict(employer=emp, employee=en, employer_tan=tn_, employer_pan=pn_, employee_pan=pe, assessment_year=ay, financial_year=fy, certificate_no=cert, gross_salary=gross, hra_exempt=hra_ex, standard_deduction=std, deduction_80c=c80, deduction_80d=d80, taxable_income=taxable, tax_payable=tot_tax, tds_quarters=q)
    def body(D):
        D.para("Certificate under section 203 of the Income-tax Act, 1961 for tax deducted at source on salary", bold=True)
        D.kv([("Certificate No.", cert), ("Name of Employer", emp), ("TAN of Deductor", tn_), ("PAN of Deductor", pn_), ("Name of Employee", en), ("PAN of Employee", pe), ("Assessment Year", ay), ("Period", f"01-Apr-{y} to 31-Mar-{y + 1}")])
        D.table(["Quarter", "Receipt No.", "Amount of tax deducted (Rs.)"], [[f"Q{i + 1}", f"{r.randrange(10 ** 7, 10 ** 8)}", inr(q[i])] for i in range(4)], [0.2, 0.4, 0.4], ["l", "l", "r"], ["Total", "", inr(sum(q))])
        D.para("PART B - Details of salary paid and tax deducted", bold=True)
        D.table(["Particulars", "Amount (Rs.)"], [["Gross salary", inr(gross)], ["Less: House rent allowance exemption u/s 10(13A)", inr(hra_ex)], ["Less: Standard deduction", inr(std)], ["Deduction u/s 80C", inr(c80)], ["Deduction u/s 80D", inr(d80)], ["Total taxable income", inr(taxable)], ["Tax on total income", inr(tax)], ["Health and education cess", inr(cess)]], [0.7, 0.3], ["l", "r"], ["Net tax payable", inr(tot_tax)])
        D.sign(["Signature of person responsible for deduction of tax"])
    return dict(title="FORM NO. 16", hi="फॉर्म 16", issuer_lines=["Part A and Part B", address(r)[0]], header_issuer=emp, body=body, subtype="part_a_b", fields=fields, key_phrases=["form no. 16", "tax deducted at source", "tds"], issuer=emp)

BUILDERS = {"rent_receipt": b_rent_receipt, "rental_agreement": b_rental_agreement, "insurance_premium_receipt": b_insurance, "school_fee_receipt": b_school, "investment_certificate": b_investment,
            "meal_voucher_statement": b_meal, "ev_document": b_ev, "loan_interest_certificate": b_loan, "loan_statement": b_loan_stmt, "salary_slip": b_salary, "form_16": b_form16}

# ------------------------------------------------------------------ degradation
def degrade(img, r, mode):
    meta = dict(mode=mode)
    if mode == "clean": return img, meta
    a = np.asarray(img).astype(np.float32)
    if mode == "scan":
        rot = r.uniform(-1.8, 1.8); img = img.rotate(rot, resample=Image.BICUBIC, expand=False, fillcolor=(245, 243, 238)); meta["rotation_deg"] = round(rot, 2)
        img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.4, 1.0))); a = np.asarray(img).astype(np.float32); a = a * r.uniform(0.9, 1.0) + np.random.RandomState(r.randrange(2 ** 31)).normal(0, r.uniform(3, 9), a.shape); a[..., 2] *= r.uniform(0.93, 1.0)
        meta.update(blur="gaussian", noise=True, jpeg_q=r.randrange(55, 85))
    else:  # photo / low-res
        sc = r.uniform(0.38, 0.62); w, h = img.size; img = img.resize((int(w * sc), int(h * sc)), Image.BILINEAR).resize((w, h), Image.BILINEAR); rot = r.uniform(-3.5, 3.5); img = img.rotate(rot, resample=Image.BICUBIC, fillcolor=(120, 120, 120)); meta.update(downscale=round(sc, 2), rotation_deg=round(rot, 2))
        a = np.asarray(img).astype(np.float32); yy, xx = np.mgrid[0:h, 0:w]; vig = 1 - 0.28 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2; a = a * vig[..., None] * r.uniform(0.82, 1.0) + np.random.RandomState(r.randrange(2 ** 31)).normal(0, 6, a.shape); meta.update(vignette=True, noise=True, jpeg_q=r.randrange(40, 75))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), meta

# ------------------------------------------------------------------ driver
def doc_seed(base, typ, i): return int(hashlib.sha256(f"{base}|{typ}|{i}".encode()).hexdigest()[:12], 16)
LAYOUT_W = [("A", 30), ("B", 30), ("C", 25), ("D", 15)]
def generate_one(base_seed, typ, i, out):
    sd = doc_seed(base_seed, typ, i); r = random.Random(sd); np.random.seed(sd % (2 ** 32))
    layout = r.choices([l for l, _ in LAYOUT_W], [w for _, w in LAYOUT_W])[0]; st = Style(r, layout); ctx = {}
    spec = BUILDERS[typ](r, st, ctx); hindi = HINDI_OK and layout == "D" and r.random() < 0.7
    D = Doc(st, spec["header_issuer"], spec["issuer_lines"], spec["title"], spec["hi"] if hindi else None); spec["body"](D); pages = D.finish()
    mode = r.choices(["clean", "scan", "photo"], [40, 40, 20])[0]; doc_id = f"synth_{typ}_{i:04d}"; pdir = out / "images" / typ; pdir.mkdir(parents=True, exist_ok=True); files = []; lines = []; dmeta = None
    for pi, p in enumerate(pages):
        img, dmeta = degrade(p.img, random.Random(sd + pi), mode); fn = pdir / f"{doc_id}_p{pi + 1}.{'png' if mode == 'clean' else 'jpg'}"
        img.save(fn, quality=dmeta.get("jpeg_q", 90)) if fn.suffix == ".jpg" else img.save(fn, optimize=True); files.append(str(fn.relative_to(out)))
        for ln in p.lines: lines.append(dict(page=pi + 1, **ln))
    full_text = "\n".join(l["text"] for l in lines); issuer = spec["issuer"]
    rec = dict(doc_id=doc_id, doc_type=typ, subtype=spec["subtype"], layout=layout, issuer=issuer, issuer_group=f"{typ}__{layout}__{hashlib.sha1(issuer.encode()).hexdigest()[:8]}", language="en+hi" if hindi else "en", country="IN", synthetic=True, generator_version=VERSION,
               base_seed=base_seed, doc_seed=sd, n_pages=len(pages), pages=files, degrade=dmeta, fields=spec["fields"], key_phrases=spec["key_phrases"], title=spec["title"], lines=lines, layout_note={"A": "formal letterhead", "B": "banded modern", "C": "narrow thermal slip", "D": "boxed form (test-only unseen layout)"}[layout])
    (out / "ann").mkdir(exist_ok=True); (out / "ann" / f"{doc_id}.json").write_text(json.dumps(rec, ensure_ascii=False)); return rec

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="datasets/synthetic_doctype/v1"); ap.add_argument("--n", type=int, default=110); ap.add_argument("--seed", type=int, default=20261019); ap.add_argument("--types", nargs="*"); a = ap.parse_args()
    out = (ROOT / a.out) if not os.path.isabs(a.out) else Path(a.out); out.mkdir(parents=True, exist_ok=True); types = a.types or list(BUILDERS); idx = []
    for t in types:
        for i in range(a.n):
            rec = generate_one(a.seed, t, i, out); idx.append(dict(doc_id=rec["doc_id"], doc_type=t, subtype=rec["subtype"], layout=rec["layout"], issuer_group=rec["issuer_group"], n_pages=rec["n_pages"], degrade=rec["degrade"]["mode"], language=rec["language"]))
        print(f"{t:28s} {a.n} docs", flush=True)
    with open(out / "index.jsonl", "w") as f:
        for x in idx: f.write(json.dumps(x) + "\n")
    meta = dict(generator_version=VERSION, base_seed=a.seed, n_per_type=a.n, types=types, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), hindi_enabled=HINDI_OK, fonts={k: [Path(x).name if x else None for x in v] for k, v in FONT_FILES.items()}, footer=FOOTER, created=dt.datetime.now().isoformat(timespec="seconds"))
    (out / "GENERATOR.json").write_text(json.dumps(meta, indent=1)); print("total", len(idx), "->", out)
if __name__ == "__main__": main()
