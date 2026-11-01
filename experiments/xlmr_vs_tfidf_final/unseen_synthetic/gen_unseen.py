"""Independent synthetic 'unseen' challenge set for the XLM-R vs TF-IDF study.
NOT the project's scripts/generate_synthetic_data.py: new merchants, item vocabulary, title wording, field-label synonyms, four new page layouts, three conditions
(clean / noisy-photo / long-preamble). Labels are the project's 16 Task A labels, assigned by construction (no annotation step). It reads nothing from the training data.
Output: unseen_synthetic/images/*.png + manifest.jsonl (id, label, condition, group, layout, merchant). Deterministic: python gen_unseen.py --seed 20261101
This set measures robustness to wording/layout/noise shift on documents written by the study author; it is NOT evidence about real-world accuracy."""
import argparse, json, random, string, datetime as dt
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
HERE = Path(__file__).resolve().parent; FD = "/usr/share/fonts/truetype/"
FONTS = {"sans": ("dejavu/DejaVuSans.ttf", "dejavu/DejaVuSans-Bold.ttf"), "serif": ("dejavu/DejaVuSerif.ttf", "dejavu/DejaVuSerif-Bold.ttf"), "mono": ("liberation/LiberationMono-Regular.ttf", "liberation/LiberationMono-Bold.ttf"),
         "lsans": ("liberation/LiberationSans-Regular.ttf", "liberation/LiberationSans-Bold.ttf"), "lserif": ("liberation/LiberationSerif-Regular.ttf", "liberation/LiberationSerif-Bold.ttf")}
CITIES = ["Pune", "Nagpur", "Coimbatore", "Surat", "Bhopal", "Visakhapatnam", "Guwahati", "Dehradun", "Mysuru", "Thiruvananthapuram", "Ranchi", "Patna"]
NAMES = ["Kiran Bhatt", "Meera Pillai", "Farhan Qureshi", "Deepa Menon", "Harsh Vora", "Tanvi Kapoor", "Nikhil Rane", "Sana Sheikh", "Gaurav Tiwari", "Ishita Bose"]
# class -> (merchants, titles, body-line generators producing [(label, value)] and item pool [(name, lo, hi)])
C = {
 "food": (["Chaat Junction", "Mangalore Pearl Kitchen", "Tandoor Tales", "Dosa Plaza Express", "Biryani By Kilo Outlet", "Chai Sutta Corner"], ["Dine-in Bill", "Restaurant Invoice", "Food Order Receipt", "Cafe Bill"],
         [("Veg Hakka Noodles", 150, 260), ("Chilli Paneer Dry", 190, 310), ("Mutton Seekh Kebab", 280, 420), ("Rava Idli Plate", 70, 130), ("Masala Chai x2", 30, 70), ("Gobi Manchurian", 140, 230), ("Rumali Roti Basket", 60, 110), ("Mango Lassi", 70, 130)], ["Table No", "Covers", "Waiter"]),
 "medical": (["Sanjeevani Chemists", "CarePlus Dispensary", "Dr. Rao Diagnostics Centre", "Nova Health Pharmacy", "Arogya Medico"], ["Chemist Cash Memo", "Medicine Bill", "Clinic Consultation Receipt", "Diagnostic Lab Invoice"],
         [("Amoxicillin 500 caps", 70, 140), ("Metformin 500mg tabs", 25, 60), ("Consultation fee - physician", 300, 800), ("CBC blood test", 250, 500), ("Betadine ointment 20g", 60, 110), ("Glucometer strips x25", 450, 900), ("Ibuprofen 400 strip", 20, 45)], ["Doctor", "Rx No", "Patient"]),
 "miscellaneous": (["Orbit Computers & Peripherals", "Navneet Book Depot", "SunTech Solutions", "Ace Hardware Mart", "Pixel Print Studio"], ["Sales Invoice", "Cash Memo", "Retail Invoice", "Equipment Purchase Bill"],
         [("Wireless headset", 900, 2600), ("Cartridge refill", 350, 900), ("Spiral notebooks x10", 300, 600), ("Extension board 6-socket", 280, 700), ("Monitor stand", 700, 1600), ("Software licence 1yr", 1500, 6000), ("Lamination pouches", 120, 300)], ["Salesman", "Counter", "Warranty"]),
 "hotel": (["The Riverside Residency", "Lakeview Grand Hotel", "Palm Court Suites", "Jade Orchid Inn", "Heritage Haveli Stay"], ["Guest Folio", "Room Invoice", "Hotel Final Bill", "Accommodation Tax Invoice"],
         [("Superior twin room night", 2500, 7000), ("Club lounge access", 600, 1600), ("Airport pickup", 700, 1500), ("Spa treatment", 1200, 3500), ("Mini bar", 200, 900), ("Late checkout fee", 500, 1500)], ["Room No", "Check-in", "Check-out", "Nights"]),
 "fuel": (["Shell Retail Pump", "Essar Petrol Bunk", "Jio-bp Mobility Station", "Sri Venkateswara Fuels", "Gupta Filling Station"], ["Fuel Sale Slip", "Pump Receipt", "Petrol Pump Invoice", "Fuel Bill"],
         [("MS Petrol", 94, 107), ("HSD Diesel", 86, 96), ("Power Petrol 99", 104, 114)], ["Nozzle", "Vehicle No", "Density"]),
 "parking": (["Skyline Mall Parking", "Central Station Car Park", "Metro Plaza Parking", "Orbit Valet Services", "Civic Centre Pay-and-Park"], ["Parking Fee Slip", "Car Park Ticket", "Valet Parking Receipt", "Pay & Park Receipt"],
         [("Two wheeler parking", 10, 60), ("Four wheeler parking", 30, 200), ("Overnight parking", 100, 400)], ["Slot", "Entry Time", "Exit Time", "Vehicle No"]),
 "transport": (["Blu Ride Cabs", "QuickCab Mobility", "AutoWala Rides", "Namma Taxi", "City Wheels Cab"], ["Ride Invoice", "Cab Trip Summary", "Auto Rickshaw Fare Slip", "Trip Receipt"],
         [("Base fare", 40, 80), ("Distance charge", 90, 520), ("Waiting time", 10, 60), ("Toll", 30, 150)], ["Driver", "Pickup", "Drop", "Distance"]),
 "train": (["IRCTC", "Indian Railways Reservation", "Konkan Railway Booking"], ["E-Ticket Booking Receipt", "Reservation Slip", "Railway Passenger Ticket"],
         [("Sleeper class fare", 300, 700), ("3AC fare", 800, 2200), ("Reservation fee", 40, 60), ("Superfast charge", 30, 75), ("Catering charge", 0, 120)], ["Train No", "Class", "Quota", "PNR", "Boarding"]),
 "bus": (["Orange Travels", "Neeta Tours", "Paulo Coach Lines", "State Road Transport Corp", "Kallada Volvo"], ["Bus Boarding Pass", "Coach Ticket", "Volvo Sleeper Ticket", "Intercity Bus Booking"],
         [("Seat fare (AC sleeper)", 500, 1800), ("Convenience fee", 20, 60), ("Seat fare (non-AC seater)", 250, 700)], ["Seat", "Boarding Point", "Departure", "Ticket No"]),
 "flight": (["Skyward Airlines", "Akasa-style Air", "Bharat Wings", "AirConnect India", "Cirrus Air"], ["Itinerary and Receipt", "Air Ticket Confirmation", "Boarding Confirmation and Invoice", "E-Ticket"],
         [("Base airfare", 2200, 9500), ("Airport development fee", 150, 600), ("Fuel surcharge", 300, 1800), ("Web check-in seat", 0, 450)], ["Flight", "Gate", "PNR", "Cabin", "Baggage"]),
 "electricity": (["Western Discom", "Eastern Power Distribution", "Capital Electric Supply", "Southern Grid Corporation", "Hill State Power Board"], ["Electricity Consumption Bill", "Energy Bill", "Power Supply Demand Notice", "Monthly Power Bill"],
         [("Energy charge", 400, 2600), ("Fixed demand charge", 90, 260), ("Electricity duty", 30, 220), ("Fuel adjustment charge", 20, 190)], ["Consumer No", "Meter No", "Units (kWh)", "Billing Cycle", "Sanctioned Load"]),
 "water": (["Metropolitan Water Board", "Jal Nigam Urban", "City Waterworks Department", "Municipal Aqua Supply"], ["Water Tax Bill", "Water Supply Charges Notice", "Domestic Water Bill", "Sewerage & Water Statement"],
         [("Water usage charge", 120, 900), ("Sewerage charge", 40, 300), ("Meter rent", 10, 60)], ["Connection No", "Meter Reading", "Kilolitres", "Ward"]),
 "gas": (["Gujarat City Gas", "Adani Total Gas Style", "Vasundhara LPG Agency", "Sundaram Gas Distributors"], ["LPG Refill Cash Memo", "PNG Consumption Bill", "Cylinder Delivery Receipt", "Gas Supply Invoice"],
         [("LPG cylinder 14.2kg refill", 820, 1100), ("PNG usage (SCM)", 150, 900), ("Delivery charge", 0, 50)], ["Consumer No", "Cylinder Type", "Delivery Man", "SCM"]),
 "internet": (["SpeedNet Broadband", "Hathway-style Cable & Net", "NetOne Fibre", "Starlink-Lite Wireless", "UltraFast Fibre Services"], ["Broadband Statement", "Fibre Plan Invoice", "Internet Service Bill", "Data Plan Bill"],
         [("Unlimited fibre plan 200 Mbps", 499, 1299), ("Static IP charge", 100, 300), ("Router rental", 50, 150), ("Installation charge", 0, 500)], ["Account ID", "Plan", "Data Used", "Billing Period"]),
 "rent": (["Landlord Receipt"], ["Monthly Rent Receipt", "House Rent Acknowledgement", "Rental Payment Receipt", "Rent Paid Receipt"],
         [("Monthly rent for flat", 9000, 55000)], ["Landlord", "Landlord PAN", "Tenant", "Property"]),
 "non_expense": (["Human Resources Dept", "Office of the Registrar", "Admin Section", "Project Management Office", "Legal Department"], ["Internal Memorandum", "Appointment Letter", "Minutes of Meeting", "Non-Disclosure Agreement", "Leave Application", "Policy Circular", "Employee Handbook Extract"],
         [], []),
}
LETTER = ["Please note the following instructions regarding the upcoming quarterly review and the revised reporting structure which will be effective from the first working day of next month.",
          "All employees are requested to complete the mandatory compliance training before the stated deadline and to acknowledge receipt of this notice by return email.",
          "The undersigned agrees to keep confidential all proprietary information disclosed in the course of the engagement and to return all materials on request.",
          "The committee discussed the agenda items, assigned action points to the named owners and resolved to reconvene after two weeks to review progress.",
          "This is to inform you that your application has been considered and that the competent authority has been pleased to approve the request subject to the conditions below."]
BOILER = ["Terms and conditions apply. Goods once sold will not be taken back. Subject to local jurisdiction. Please retain this document for your records and quote the reference number in all correspondence.",
          "Customer notice: this document is issued electronically and does not require a physical signature. For grievances contact the helpdesk during business hours. Prices are inclusive of applicable taxes unless stated otherwise.",
          "Privacy statement: we collect only the information necessary to provide this service. Your data is processed in accordance with applicable law and retained for the statutory period. You may request correction of inaccurate details.",
          "Disclaimer: the issuer shall not be liable for delays caused by events beyond reasonable control. Any dispute shall be settled by arbitration. Refunds, where eligible, are processed within seven to ten working days.",
          "Safety information: please verify the details on this document immediately. Report any discrepancy within forty-eight hours. Keep the document away from heat and moisture to avoid fading of printed characters."]
def money(x): return f"{x:,.2f}"
def ref(r, p): return f"{p}{r.randint(10000, 99999999)}"
def content(cls, r):
    mer, titles, pool, keys = C[cls]; m = r.choice(mer); title = r.choice(titles); city = r.choice(CITIES); d = dt.date(2024, 1, 1) + dt.timedelta(days=r.randint(0, 700))
    fmt = r.choice(["%d/%m/%Y", "%d-%b-%Y", "%Y-%m-%d", "%d %B %Y"]); L = []
    if cls == "non_expense":
        para = r.sample(LETTER, 3); body = [("p", t) for t in para]; return dict(merchant=m, title=title, city=city, date=d.strftime(fmt), head=[("Ref", ref(r, "HR/")), ("Date", d.strftime(fmt)), ("To", r.choice(NAMES))], items=[], body=body, total=None, extra=[])
    its = []
    if cls == "fuel":
        nm, lo, hi = r.choice(pool); rate = round(r.uniform(lo, hi), 2); qty = round(r.uniform(4, 40), 2); its = [(f"{nm} @ {rate}/L x {qty} L", round(rate * qty, 2))]
    elif cls == "rent":
        amt = float(r.randrange(*[9000, 55001], 500)); its = [(f"Monthly rent for {d.strftime('%B %Y')}", amt)]
    else:
        for nm, lo, hi in r.sample(pool, k=min(len(pool), r.randint(2, 5))): its.append((nm, round(r.uniform(lo, hi), 2)))
    sub = round(sum(a for _, a in its), 2); tax = round(sub * r.choice([0, 0.05, 0.12, 0.18]), 2); total = round(sub + tax)
    head = [("Bill No", ref(r, r.choice(["A/", "BL-", "No."]))), ("Date", d.strftime(fmt)), (r.choice(["Name", "Customer", "Billed to"]), r.choice(NAMES))]
    for k in r.sample(keys, k=min(len(keys), r.randint(2, 4))): head.append((k, ref(r, "") if "No" in k or k in ("PNR", "Slot", "Seat") else str(r.randint(1, 60))))
    return dict(merchant=m, title=title, city=city, date=d.strftime(fmt), head=head, items=its, body=[], total=total, sub=sub, tax=tax, extra=[])
def wrap(dr, text, font, width):
    out, cur = [], ""
    for w in text.split(" "):
        if cur and dr.textlength(cur + " " + w, font=font) > width: out.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    if cur: out.append(cur)
    return out
def render(c, style, r, long=False):
    fam = r.choice(list(FONTS)); sz = r.randint(15, 20); f1 = ImageFont.truetype(FD + FONTS[fam][0], sz); f2 = ImageFont.truetype(FD + FONTS[fam][1], sz + 3)
    W = {"s1": 820, "s2": 900, "s3": r.randint(360, 460), "s4": 880}[style]; probe = ImageDraw.Draw(Image.new("RGB", (1, 1))); lines = []   # (text, bold, align)
    if long: lines += [(t, False, "l") for t in r.sample(BOILER, 4) + r.sample(BOILER, 3)]
    if style == "s3": lines += [(c["merchant"].upper(), True, "c"), (c["city"], False, "c"), (c["title"], True, "c"), ("-" * 28, False, "c")]
    elif style == "s4": lines += [(c["title"].upper(), True, "l"), (f"Issued by {c['merchant']}, {c['city']}", False, "l"), ("", False, "l")]
    else: lines += [(c["merchant"], True, "c" if style == "s1" else "l"), (c["title"], True, "c" if style == "s1" else "l"), (c["city"], False, "c" if style == "s1" else "l"), ("", False, "l")]
    for k, v in c["head"]: lines.append((f"{k}: {v}" if style != "s2" else f"{k:<16}{v}", False, "l"))
    lines.append(("", False, "l"))
    for t in [x[1] for x in c["body"]]: lines.append((t, False, "l")); lines.append(("", False, "l"))
    if c["items"]:
        lines.append(("Particulars" + " " * 6 + "Amount (Rs)", True, "l"))
        for nm, a in c["items"]: lines.append((f"{nm}   {money(a)}", False, "l"))
        if c["tax"]: lines += [(f"Subtotal {money(c['sub'])}", False, "r"), (f"Tax {money(c['tax'])}", False, "r")]
        lines.append((f"{r.choice(['Amount payable', 'Net total', 'Grand Total', 'Total due'])}: Rs. {money(c['total'])}", True, "r"))
    if long: lines += [(t, False, "l") for t in r.sample(BOILER, 2)]
    if c["body"]: lines += [("", False, "l"), ("Authorised signatory", False, "l")]
    wl = []
    for t, b, a in lines:
        for s in (wrap(probe, t, f2 if b else f1, W - 48) or [""]): wl.append((s, b, a))
    lh = sz + 9; H = lh * len(wl) + 100; img = Image.new("RGB", (W, H), r.choice([(255, 255, 255), (250, 248, 240), (244, 246, 250)])); dr = ImageDraw.Draw(img); y = 40
    accent = r.choice([(30, 60, 120), (120, 30, 40), (30, 100, 70), (50, 50, 50)])
    if style in ("s2", "s4"): dr.rectangle([0, 0, W, 22], fill=accent)
    for s, b, a in wl:
        ft = f2 if b else f1; w = dr.textlength(s, font=ft); x = {"c": (W - w) / 2, "l": 24, "r": W - w - 24}[a]; dr.text((x, y), s, font=ft, fill=accent if (b and style != "s3") else (20, 20, 20)); y += lh
    if style == "s2": dr.rectangle([6, 28, W - 6, H - 6], outline=accent, width=2)
    return img
def degrade(img, r):
    img = img.rotate(r.uniform(-3.5, 3.5), expand=True, fillcolor=(235, 235, 235), resample=Image.BICUBIC)
    s = r.uniform(0.5, 0.75); img = img.resize((max(200, int(img.width * s)), max(200, int(img.height * s))), Image.BILINEAR).filter(ImageFilter.GaussianBlur(r.uniform(0.6, 1.3)))
    a = np.asarray(img).astype(np.float32) * r.uniform(0.75, 1.05) + r.uniform(-18, 12); a += np.random.RandomState(r.randint(0, 10**6)).normal(0, r.uniform(6, 16), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261101); ap.add_argument("--per-class-condition", type=int, default=12); a = ap.parse_args(); r = random.Random(a.seed)
    (HERE / "images").mkdir(exist_ok=True); rows = []
    for cls in C:
        for cond in ("clean", "noisy", "long"):
            for i in range(a.per_class_condition):
                c = content(cls, r); style = r.choice(["s1", "s2", "s3", "s4"]) if cls != "non_expense" else r.choice(["s1", "s2", "s4"]); img = render(c, style, r, long=(cond == "long"))
                if cond == "noisy": img = degrade(img, r)
                did = f"us_{cls}_{cond}_{i:02d}"; img.convert("RGB").save(HERE / "images" / f"{did}.jpg" if cond == "noisy" else HERE / "images" / f"{did}.png", **({"quality": r.randint(35, 60)} if cond == "noisy" else {}))
                rows.append(dict(id=did, file=f"images/{did}.{'jpg' if cond == 'noisy' else 'png'}", label=cls, condition=cond, layout=style, merchant=c["merchant"], group=f"{cls}|{c['merchant']}|{style}"))
    (HERE / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n"); print(len(rows), "documents;", len({x["group"] for x in rows}), "groups")
