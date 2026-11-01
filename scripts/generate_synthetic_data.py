"""Synthetic Indian bills with exact OCR boxes + field labels. Everything is synthetic=True.
Fills the gap categories (no public Indian data). Deterministic given --seed.
Usage: python generate_synthetic_data.py [--per-class 150] [--seed 42]"""
import argparse, json, random, string, datetime as dt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from common import SYNTH, CATEGORIES, DEFAULT_DOC_TYPE, SEED

FD = "/usr/share/fonts/truetype/"
FONTS = {"sans": FD + "dejavu/DejaVuSans.ttf", "sansb": FD + "dejavu/DejaVuSans-Bold.ttf",
         "mono": FD + "liberation/LiberationMono-Regular.ttf", "monob": FD + "liberation/LiberationMono-Bold.ttf",
         "serif": FD + "dejavu/DejaVuSerif.ttf", "serifb": FD + "dejavu/DejaVuSerif-Bold.ttf"}  # all have U+20B9 (rupee)

STATES = {"29": ("Bengaluru", "Karnataka", "560"), "27": ("Mumbai", "Maharashtra", "400"), "07": ("New Delhi", "Delhi", "110"),
          "33": ("Chennai", "Tamil Nadu", "600"), "19": ("Kolkata", "West Bengal", "700"), "36": ("Hyderabad", "Telangana", "500"),
          "24": ("Ahmedabad", "Gujarat", "380"), "09": ("Lucknow", "Uttar Pradesh", "226"), "08": ("Jaipur", "Rajasthan", "302"),
          "32": ("Kochi", "Kerala", "682"), "03": ("Chandigarh", "Punjab", "160"), "23": ("Indore", "Madhya Pradesh", "452")}
STREETS = ["MG Road", "Brigade Road", "Station Road", "Gandhi Nagar", "Nehru Place", "Sector 18", "Park Street", "Residency Road",
           "Linking Road", "Anna Salai", "Civil Lines", "Church Street", "Ring Road", "Jubilee Hills", "Koramangala 4th Block"]
FIRST = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohit", "Kavita", "Arjun", "Neha", "Suresh", "Divya", "Manoj", "Pooja"]
LAST = ["Sharma", "Verma", "Iyer", "Reddy", "Patel", "Singh", "Nair", "Gupta", "Das", "Mehta", "Kulkarni", "Joshi", "Khan", "Rao"]
PAY = ["Cash", "UPI", "Card", "Credit Card", "Debit Card", "Net Banking", "Paytm"]
PAN_L = string.ascii_uppercase

def gstin(r, st):
    pan = "".join(r.choices(PAN_L, k=5)) + "".join(r.choices(string.digits, k=4)) + r.choice(PAN_L)
    return f"{st}{pan}{r.randint(1,9)}Z{r.choice(string.digits + PAN_L)}"

def money(x): return f"{x:,.2f}"
def addr(r, st):
    c, s, pin = STATES[st]
    return f"{r.randint(1,250)}, {r.choice(STREETS)}, {c}, {s} - {pin}{r.randint(0,999):03d}"
def phone(r): return f"+91 {r.choice('6789')}{r.randint(0,9999):04d}{r.randint(0,99999):05d}"
def person(r): return f"{r.choice(FIRST)} {r.choice(LAST)}"
def date(r): return dt.date(2023, 1, 1) + dt.timedelta(days=r.randint(0, 900))
def datefmt(r, d): return d.strftime(r.choice(["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d.%m.%Y"]))

# (merchant names, item pool [(name, lo, hi, gst%)], default gst) per category. Names are fictional.
FOOD_ITEMS = [("Paneer Butter Masala", 220, 320, 5), ("Veg Biryani", 180, 280, 5), ("Butter Naan", 40, 70, 5), ("Masala Dosa", 90, 160, 5),
              ("Chicken Biryani", 240, 380, 5), ("Dal Tadka", 150, 220, 5), ("Filter Coffee", 40, 90, 5), ("Cold Coffee", 120, 190, 5),
              ("Gulab Jamun", 70, 120, 5), ("Veg Thali", 160, 260, 5), ("Fresh Lime Soda", 60, 110, 5), ("Chicken Tikka", 260, 380, 5), ("Mineral Water", 20, 40, 18)]
MED_ITEMS = [("Paracetamol 650mg Strip", 25, 45, 12), ("Azithromycin 500mg", 90, 140, 12), ("Cetirizine 10mg", 18, 40, 12), ("Vitamin D3 60K", 110, 190, 12),
             ("ORS Powder", 20, 35, 12), ("Pantoprazole 40mg", 80, 150, 12), ("Cough Syrup 100ml", 70, 130, 12), ("Digital Thermometer", 150, 300, 18), ("Surgical Mask Pack", 50, 120, 12)]
OFFICE_ITEMS = [("A4 Paper Ream 500s", 280, 420, 12), ("Ball Pens Box of 50", 150, 300, 12), ("Stapler Heavy Duty", 120, 350, 18), ("Wireless Mouse", 400, 1200, 18),
                ("USB Keyboard", 500, 1500, 18), ("HDMI Cable 2m", 200, 450, 18), ("Printer Toner Cartridge", 2200, 4800, 18), ("Whiteboard Markers Set", 120, 260, 12), ("Laptop Bag", 900, 2400, 18)]
HOTEL_ROOMS = [("Deluxe Room Tariff", 2200, 6500, 12), ("Executive Room Tariff", 3500, 8500, 18), ("Breakfast Buffet", 400, 900, 5), ("Laundry", 150, 600, 18), ("Room Service", 300, 1400, 5)]

SALARY_ITEMS = [("Basic Salary", 25000, 60000, 0), ("House Rent Allowance", 10000, 24000, 0), ("Special Allowance", 5000, 20000, 0), ("Conveyance Allowance", 1600, 1600, 0), ("Provident Fund (deduction)", 1800, 7200, 0), ("Professional Tax (deduction)", 200, 200, 0)]
TXN_ITEMS = [("UPI/ZOMATO/Payment", 150, 900, 0), ("NEFT-Salary Credit", 30000, 90000, 0), ("ATM WDL Cash", 500, 10000, 0), ("IMPS/Rent Transfer", 8000, 35000, 0), ("ECS EMI Home Loan", 12000, 40000, 0), ("POS Purchase Amazon", 300, 5000, 0), ("Interest Credit", 20, 400, 0)]
CAT = {  # category -> (merchants, items, title, kind)
    "food": (["Spice Garden Restaurant", "Udupi Sagar", "The Biryani House", "Cafe Coffee Corner", "Punjabi Tadka Dhaba", "Saravana Meals", "Hotel Annapurna"], FOOD_ITEMS, "TAX INVOICE", "thermal"),
    "medical": (["Apollo Pharmacy", "MedPlus Chemist", "Wellness Forever Pharmacy", "City Care Hospital", "Sri Sai Medicals", "Lifeline Diagnostics"], MED_ITEMS, "PHARMACY BILL", "thermal"),
    "miscellaneous": (["Sharma Stationers", "Bharat Office Mart", "Digital Planet Electronics", "Metro Cash & Carry Traders", "Krishna Enterprises"], OFFICE_ITEMS, "TAX INVOICE", "a4"),
    "hotel": (["Hotel Grand Residency", "The Lotus Inn", "Taj-style Palace Hotels", "Comfort Stay Suites", "Hotel Sunrise Regency"], HOTEL_ROOMS, "HOTEL TAX INVOICE", "a4"),
    "fuel": (["Indian Oil Retail Outlet", "Bharat Petroleum Dealer", "HP Fuel Station", "Reliance Petro Pump", "Nayara Energy Pump"], [], "FUEL RECEIPT", "thermal"),
    "parking": (["City Mall Parking", "Airport Parking Services", "Municipal Parking Lot", "Phoenix Mall Parking"], [], "PARKING RECEIPT", "thermal"),
    "transport": (["Ola Cabs", "Uber India", "Meru Cabs", "Rapido Auto", "City Radio Taxi"], [], "TRIP RECEIPT", "ride"),
    "train": (["IRCTC e-Ticketing", "Indian Railways PRS"], [], "ELECTRONIC RESERVATION SLIP", "ticket"),
    "bus": (["RedBus Ticket", "KSRTC", "MSRTC", "VRL Travels", "SRS Travels"], [], "BUS TICKET", "ticket"),
    "flight": (["IndiGo", "Air India", "SpiceJet", "Vistara-style Airways", "Akasa Air-style"], [], "FLIGHT E-TICKET / TAX INVOICE", "ticket"),
    "electricity": (["BESCOM", "MSEDCL", "Tata Power-DDL", "TNEB (TANGEDCO)", "KSEB", "UPPCL"], [], "ELECTRICITY BILL", "utility"),
    "water": (["BWSSB", "Delhi Jal Board", "Municipal Water Supply Dept", "Hyderabad Metro Water"], [], "WATER CHARGES BILL", "utility"),
    "gas": (["Indane Gas", "HP Gas", "Bharat Gas", "Mahanagar Gas", "Indraprastha Gas"], [], "GAS BILL / CASH MEMO", "utility"),
    "internet": (["Airtel Broadband", "Jio Fiber", "ACT Fibernet", "BSNL Broadband", "Vi Postpaid"], [], "TELECOM / BROADBAND BILL", "utility"),
    "rent": (["Rent Receipt"], [], "RENT RECEIPT", "a4"),
    # hard negatives: carry INR/GST/invoice-like numbers but are NOT expense receipts (category non_expense)
    "neg_purchase_order": (CAT_VENDORS := ["Sharma Stationers", "Bharat Office Mart", "Digital Planet Electronics", "Krishna Enterprises"], OFFICE_ITEMS, "PURCHASE ORDER", "a4"),
    "neg_quotation": (CAT_VENDORS, OFFICE_ITEMS, "QUOTATION / PROFORMA INVOICE", "a4"),
    "neg_delivery_challan": (CAT_VENDORS, OFFICE_ITEMS, "DELIVERY CHALLAN (NOT A TAX INVOICE)", "a4"),
    "neg_salary_slip": (["Infosys-style Technologies Pvt Ltd", "Sunrise Software Solutions", "Bharat Manufacturing Co", "Apex Consulting LLP"], SALARY_ITEMS, "SALARY SLIP", "a4"),
    "neg_bank_statement": (["State Bank of India", "HDFC Bank", "ICICI Bank", "Punjab National Bank", "Axis Bank"], TXN_ITEMS, "ACCOUNT STATEMENT", "a4"),
    "neg_payment_confirmation": (["PhonePe", "Google Pay", "Paytm", "BHIM UPI"], [], "PAYMENT SUCCESSFUL - TRANSACTION CONFIRMATION", "ride"),
}
NEG_SUB = {k: k[4:] for k in CAT if k.startswith("neg_")}

def build_content(cat, r):
    """Return dict(fields, lines[(text,style)], items) for the document body."""
    st = r.choice(list(STATES))
    merch = r.choice(CAT[cat][0]); title = CAT[cat][2]
    d = date(r); f = {"merchant": merch, "date": datefmt(r, d), "currency": "INR", "address": addr(r, st), "phone": phone(r),
                      "GSTIN": gstin(r, st), "invoice_number": f"{r.choice(['INV','BL','RC','T','B'])}{r.randint(2023,2025)}/{r.randint(1000,999999)}",
                      "customer_name": person(r), "payment_method": r.choice(PAY), "title": title}
    items, sub = [], 0.0
    pool = CAT[cat][1]
    if pool:
        for nm, lo, hi, g in r.sample(pool, k=min(len(pool), r.randint(2, 6))):
            q = r.randint(1, 3 if cat != "miscellaneous" else 5); p = round(r.uniform(lo, hi), 2); items.append((nm, q, p, g, round(q * p, 2)))
    elif cat == "fuel":
        fuel = r.choice(["Petrol", "Diesel", "Speed Petrol", "XP95 Petrol"]); rate = round(r.uniform(92, 108), 2); litres = round(r.uniform(5, 45), 2)
        items = [(f"{fuel} @ Rs {rate}/L  Vol {litres} L", 1, round(rate * litres, 2), 0, round(rate * litres, 2))]
        f["vehicle_no"] = f"{r.choice(['KA','MH','DL','TN','GJ','UP'])}{r.randint(1,60):02d}{r.choice(PAN_L)}{r.choice(PAN_L)}{r.randint(1000,9999)}"
    elif cat == "parking":
        h = r.randint(1, 10); items = [(f"Parking charges {h} hr", 1, 20.0 * h, 18, 20.0 * h)]; f["vehicle_no"] = f"{r.choice(['KA','MH','DL','TN'])}{r.randint(1,60):02d}{r.choice(PAN_L)}{r.choice(PAN_L)}{r.randint(1000,9999)}"
    elif cat == "transport":
        km = round(r.uniform(2, 28), 1); fare = round(60 + km * r.uniform(11, 19), 2); items = [(f"Ride fare ({km} km)", 1, fare, 5, fare), ("Booking fee", 1, round(r.uniform(5, 25), 2), 18, 0)]; items[1] = (items[1][0], 1, items[1][2], 18, items[1][2])
    elif cat in ("train", "bus", "flight"):
        a, b = r.sample([v[0] for v in STATES.values()], 2); base = {"train": (250, 2800), "bus": (300, 1800), "flight": (2500, 14000)}[cat]
        fare = round(r.uniform(*base), 2); items = [(f"{a} to {b} - base fare", 1, fare, 5 if cat != "flight" else 5, fare)]; f.update(origin=a, destination=b, pnr=str(r.randint(10**9, 10**10 - 1)))
        if cat == "train": f["train_no"] = f"{r.randint(10000,22999)}"
        if cat == "flight": f["flight_no"] = f"{r.choice(['6E','AI','SG','QP'])}-{r.randint(100,999)}"
    elif cat == "electricity":
        u = r.randint(80, 640); amt = round(u * r.uniform(5.5, 8.5) + 120, 2); items = [(f"Energy charges ({u} kWh)", 1, amt, 0, amt), ("Fixed charges", 1, 120.0, 0, 120.0)]; f["consumer_no"] = str(r.randint(10**9, 10**10 - 1)); f["units"] = u
    elif cat == "water":
        kl = r.randint(8, 40); amt = round(kl * r.uniform(6, 22), 2); items = [(f"Water charges ({kl} KL)", 1, amt, 0, amt), ("Sewerage cess", 1, round(amt * .3, 2), 0, round(amt * .3, 2))]; f["consumer_no"] = str(r.randint(10**8, 10**9))
    elif cat == "gas":
        items = [("Domestic LPG 14.2 kg cylinder" if r.random() < .6 else "Piped natural gas usage", 1, round(r.uniform(850, 1100), 2), 5, round(r.uniform(850, 1100), 2))]; items[0] = (items[0][0], 1, items[0][2], 5, items[0][2]); f["consumer_no"] = str(r.randint(10**9, 10**10 - 1))
    elif cat == "internet":
        plan = r.choice(["Fibre 100 Mbps", "Fibre 300 Mbps", "Postpaid 599 plan", "Broadband unlimited 50 Mbps"]); amt = round(r.uniform(399, 1499), 2); items = [(f"{plan} - monthly rental", 1, amt, 18, amt)]; f["account_no"] = str(r.randint(10**9, 10**10 - 1))
    elif cat == "neg_payment_confirmation":
        amt = float(r.randint(50, 20000)); items = [(f"Paid to {person(r)} (UPI ID {r.choice(string.ascii_lowercase)}{r.randint(100,999)}@upi)", 1, amt, 0, amt)]; f["upi_ref"] = str(r.randint(10**11, 10**12 - 1))
    elif cat == "rent":
        amt = float(r.choice(range(8000, 60001, 1000))); items = [("House rent for the month of " + d.strftime("%B %Y"), 1, amt, 0, amt)]; f["landlord"] = person(r); f["landlord_pan"] = "".join(r.choices(PAN_L, k=5)) + f"{r.randint(1000,9999)}" + r.choice(PAN_L)
    sub = round(sum(i[4] for i in items), 2)
    tax = round(sum(i[4] * i[3] / 100 for i in items), 2)
    intra = r.random() < .75
    f.update(subtotal=money(sub), tax_amount=money(tax), total_amount=money(round(sub + tax)), round_off=money(round(sub + tax) - (sub + tax)))
    f["tax_split"] = ({"CGST": money(tax / 2), "SGST": money(tax / 2)} if intra else {"IGST": money(tax)}) if tax else {}
    return f, items

def render(cat, f, items, r, out_png, extra=None):  # extra: optional [(text,bold,align,accent)] lines (Task B synthetic_b); default output unchanged
    kind = CAT[cat][3]
    W = {"thermal": r.randint(380, 520), "ride": 560, "ticket": 900, "utility": 900, "a4": 900}[kind]
    fam = r.choice(["mono", "sans", "serif"]) if kind == "thermal" else r.choice(["sans", "serif"])
    sz = r.randint(15, 19) if kind == "thermal" else r.randint(17, 21)
    fn, fb = ImageFont.truetype(FONTS[fam], sz), ImageFont.truetype(FONTS[fam + "b" if fam + "b" in FONTS else fam], sz + 3)
    bg = (255, 255, 255) if kind != "thermal" else r.choice([(255, 255, 255), (250, 248, 240), (245, 245, 245)])
    accent = r.choice([(20, 60, 140), (150, 30, 30), (20, 110, 60), (60, 60, 60)])
    ink = (20, 20, 20)
    L = []  # (text, bold, align, accent)
    sep = "-" * (W // (sz * 6 // 10)) if kind == "thermal" else None
    L += [(f["merchant"].upper(), True, "c", True), (f["address"], False, "c", False), (f"Ph: {f['phone']}", False, "c", False)]
    if cat not in ("rent",) and f.get("GSTIN"): L.append((f"GSTIN: {f['GSTIN']}", False, "c", False))
    L += [(f["title"], True, "c", True), ("", False, "l", False),
          (f"Bill No: {f['invoice_number']}", False, "l", False), (f"Date: {f['date']}", False, "l", False), (f"Customer: {f['customer_name']}", False, "l", False)]
    for k, lab in [("vehicle_no", "Vehicle No"), ("pnr", "PNR"), ("train_no", "Train No"), ("flight_no", "Flight"), ("consumer_no", "Consumer No"),
                   ("account_no", "Account No"), ("landlord", "Landlord"), ("landlord_pan", "Landlord PAN")]:
        if k in f: L.append((f"{lab}: {f[k]}", False, "l", False))
    if "origin" in f: L.append((f"From: {f['origin']}   To: {f['destination']}", False, "l", False))
    if "units" in f: L.append((f"Units consumed: {f['units']} kWh", False, "l", False))
    L += extra or []
    if sep: L.append((sep, False, "l", False))
    L.append(("Description" + " " * 4 + "Qty   Rate   Amount", True, "l", False))
    for nm, q, p, g, a in items: L.append((f"{nm[:34]}  {q} x {money(p)} = {money(a)}", False, "l", False))
    if sep: L.append((sep, False, "l", False))
    L.append((f"Sub Total: Rs. {f['subtotal']}", False, "r", False))
    for k, v in f["tax_split"].items(): L.append((f"{k}: Rs. {v}", False, "r", False))
    if f["round_off"] not in ("0.00", "-0.00"): L.append((f"Round off: {f['round_off']}", False, "r", False))
    L.append((f"TOTAL: ₹ {f['total_amount']}", True, "r", True))
    if not cat.startswith("neg_"): L.append((f"Paid via {f['payment_method']}", False, "l", False))
    if cat in ("neg_purchase_order", "neg_quotation"): L.append(("Terms: delivery within 15 days. Valid for 30 days. Authorised signatory.", False, "l", False))
    L.append(("Thank you! Visit again." if kind == "thermal" else "This is a computer generated document.", False, "c", False))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1))); W2 = []
    for text, bold, al, acc in L:  # wrap lines that would overflow the page
        ft = fb if bold else fn; cur = ""
        for tok in text.split(" "):
            if cur and probe.textlength(cur + " " + tok, font=ft) > W - 48: W2.append((cur, bold, al, acc)); cur = tok
            else: cur = (cur + " " + tok).strip() if cur else tok
        W2.append((cur, bold, al, acc))
    L = W2
    lh = sz + 10; H = lh * len(L) + 120
    img = Image.new("RGB", (W, H), bg); dr = ImageDraw.Draw(img); words = []
    if kind in ("a4", "utility", "ticket"): dr.rectangle([0, 0, W, 70], fill=accent)
    y = 24 if kind == "thermal" else 90
    for text, bold, al, acc in L:
        ft = fb if bold else fn; w = dr.textlength(text, font=ft)
        x = {"c": (W - w) / 2, "l": 24, "r": W - w - 24}[al]
        dr.text((x, y), text, font=ft, fill=accent if (acc and kind != "thermal") else ink)
        cx = x
        for tok in text.split(" "):
            if tok:
                tw = dr.textlength(tok, font=ft); words.append({"text": tok, "bbox": [int(cx), y, int(cx + tw), y + sz + 4]})
            cx += dr.textlength(tok + " ", font=ft)
        y += lh
    if kind == "utility": dr.rectangle([8, 82, W - 8, H - 8], outline=accent, width=2)
    return img, words

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-class", type=int, default=150); ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--neg-per-class", type=int, default=30)
    a = ap.parse_args(); r = random.Random(a.seed)
    (SYNTH / "images").mkdir(parents=True, exist_ok=True); n = 0
    with open(SYNTH / "annotations.jsonl", "w") as out:
        for cat in CAT:
            for i in range(a.neg_per_class if cat.startswith("neg_") else a.per_class):
                f, items = build_content(cat, r); sid = f"synth_{cat}_{i:04d}"
                img, words = render(cat, f, items, r, None); img.save(SYNTH / "images" / f"{sid}.png")
                out.write(json.dumps({"source_id": sid, "image": f"images/{sid}.png", "expense_category": "non_expense" if cat.startswith("neg_") else cat, "subcategory": NEG_SUB.get(cat),
                                      "hard_negative": cat.startswith("neg_"), "document_type": "non_expense_document" if cat.startswith("neg_") else DEFAULT_DOC_TYPE[cat],
                                      "fields": {k: v for k, v in f.items() if k != "title"}, "words": words}, ensure_ascii=False) + "\n"); n += 1
    print("generated", n)

if __name__ == "__main__":
    main()
