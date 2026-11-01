"""Task B synthetic_b: Indian synthetic bills carrying the fields NO real dataset annotates (HSN/SAC, hotel stay dates/room/nights, patient/doctor/Rx/pharmacy,
employee name/id, passenger/ticket/bus numbers, travel date). Reuses the Task A renderer (generate_synthetic_data) with extra lines; Task A images/annotations are untouched.
Output: datasets/synthetic_b/images/*.png + annotations.jsonl (canonical Task B field values + exact word boxes). Always synthetic=True. Deterministic (seed 4242)."""
import argparse, json, random, string, datetime as dt
from PIL import Image
from schema import *   # (also puts scripts/ on sys.path)
import generate_synthetic_data as g
OUT = DS / "synthetic_b"
SAC = {"food": "996331", "rent": "997212", "internet": "998422", "hotel": "996311", "miscellaneous": None, "medical": "3004", "transport": "996411", "fuel": None}
HSN_MISC = ["4820", "8471", "8517", "9403", "3926", "4802", "8443", "8528"]
DOCTORS = ["Dr. " + n for n in ["Anil Kapoor", "S. Raghavan", "Meera Nair", "R. K. Gupta", "Priya Menon", "Imran Qureshi", "Sunita Rao", "A. Banerjee"]]
def vehicle(r): return f"{r.choice(['KA','MH','DL','TN','GJ','UP'])}-{r.randint(1,60):02d}-{r.choice(string.ascii_uppercase)}-{r.randint(1000,9999)}"
def dfmt(r, d, style): return d.strftime(style)

def make(cat, kind, r):
    f, items = g.build_content(cat, r); extra = []; gt = {}
    base = dt.date(2023, 1, 1) + dt.timedelta(days=r.randint(0, 900)); sty = r.choice(["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d.%m.%Y"])
    def line(label, val, field, normval=None):
        extra.append((f"{label}: {val}", False, "l", False)); gt[field] = {"value": str(val), "norm": normval}
    if kind == "hotel":
        n = r.randint(1, 6); ci = base; co = base + dt.timedelta(days=n); nm = g.person(r)
        line("Guest Name", nm, "employee_name"); line("Room No", r.randint(101, 1215), "room_number")
        line("Check-in", ci.strftime(sty), "check_in", ci.isoformat()); line("Check-out", co.strftime(sty), "check_out", co.isoformat()); line("Nights", n, "number_of_nights", n)
        if r.random() < .6: line("Employee ID", f"EMP{r.randint(1000, 99999)}", "employee_id")
        line("HSN/SAC", "996311", "hsn_sac")
    elif kind == "medical":
        line("Patient Name", g.person(r), "patient_name"); line("Doctor", r.choice(DOCTORS), "doctor_name"); line("Rx No", f"RX{r.randint(100000, 9999999)}", "prescription_number")
        if any(k in f["merchant"] for k in ("Pharmacy", "Chemist", "Medicals", "Medplus", "MedPlus")): gt["pharmacy_name"] = {"value": f["merchant"], "norm": None}
        line("HSN", r.choice(["3004", "3005", "9018"]), "hsn_sac")
    elif kind == "hsn":
        code = SAC.get(cat) or r.choice(HSN_MISC); line("HSN/SAC", code, "hsn_sac")
    elif kind == "employee":
        line("Employee Name", g.person(r), "employee_name"); line("Employee ID", f"EMP{r.randint(1000, 99999)}", "employee_id")
        if r.random() < .5: line("HSN/SAC", SAC.get(cat) or r.choice(HSN_MISC), "hsn_sac")
    elif kind == "travel":
        line("Passenger", g.person(r), "passenger_name"); line("Ticket No", str(r.randint(10**9, 10**10 - 1)), "ticket_number")
        line("Journey Date", (base + dt.timedelta(days=r.randint(0, 30))).strftime(sty), "travel_date")
        if cat == "bus": line("Bus No", vehicle(r), "bus_number")
    img, words = g.render(cat, f, items, r, None, extra=extra)
    # canonical values of the base content (only those actually rendered are kept by the Task B builder via word matching)
    base_fields = {"merchant_name": f["merchant"], "merchant_address": f["address"], "invoice_number": f["invoice_number"], "invoice_date": f["date"], "subtotal": f["subtotal"], "tax_total": f["tax_amount"],
                   "total_amount": f["total_amount"], "currency": "INR", "payment_method": f["payment_method"], "gstin": f.get("GSTIN") if cat != "rent" else None,
                   "cgst": f["tax_split"].get("CGST"), "sgst": f["tax_split"].get("SGST"), "igst": f["tax_split"].get("IGST"),
                   "origin": f.get("origin"), "destination": f.get("destination"), "pnr": f.get("pnr"), "train_number": f.get("train_no"), "flight_number": f.get("flight_no"), "vehicle_number": f.get("vehicle_no")}
    for k, v in base_fields.items():
        if v not in (None, "") and k not in gt: gt[k] = {"value": str(v), "norm": None}
    if "travel_date" in gt: gt["travel_date"]["norm"] = parse_date(gt["travel_date"]["value"])
    return img, words, gt

PLAN = [("hotel", "hotel", 220), ("medical", "medical", 220), ("food", "hsn", 80), ("miscellaneous", "hsn", 100), ("rent", "hsn", 60), ("internet", "hsn", 60),
        ("food", "employee", 50), ("transport", "employee", 50), ("miscellaneous", "employee", 50), ("bus", "travel", 70), ("train", "travel", 60), ("flight", "travel", 60)]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=4242); a = ap.parse_args(); r = random.Random(a.seed)
    (OUT / "images").mkdir(parents=True, exist_ok=True); n = 0
    with open(OUT / "annotations.jsonl", "w") as fo:
        for cat, kind, cnt in PLAN:
            for i in range(cnt):
                img, words, gt = make(cat, kind, r); sid = f"synthB_{cat}_{kind}_{i:04d}"; img.save(OUT / "images" / f"{sid}.png")
                fo.write(json.dumps({"source_id": sid, "image": f"images/{sid}.png", "expense_category": cat, "kind": kind, "fields": gt, "words": words}, ensure_ascii=False) + "\n"); n += 1
    print("synthetic_b docs", n)
