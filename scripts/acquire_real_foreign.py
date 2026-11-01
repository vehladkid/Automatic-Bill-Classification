"""Pull a diverse, category-balanced SUBSET of real foreign receipts (no full-dataset downloads):
 * ReceiptBench (HF T0rl/ReceiptBench, MIT): 10.6k real receipts typed by the authors; images are read from the remote 5.2 GB zip
   with HTTP range requests, only the selected ones are written to datasets/public/receiptbench/.
 * InvoicesReceiptsPT (HF Francisco-Cruz/InvoicesReceiptsPT, Apache-2.0 / Zenodo 6371710+7213544): Portuguese invoices; seller-name
   rules pick fuel/pharmacy/clinic/parking/hotel/restaurant documents, at most 6 per seller.
 * my_receipts (Kaggle jenswalter/receipts, CC0) is downloaded whole (176 MB) by download_datasets.py.
Selection is deterministic (sorted ids, seed 42) and diversity-first: round-robin over sellers, then countries."""
import json, re, zipfile, random, collections, os
from common import *

CAPS = {"hotel": 80, "flight": 80, "transport": 80, "train": 50, "bus": 80, "fuel": 60, "parking": 40, "internet": 60, "utility": 25, "medical": 8, "food": 50}
TYPE2CAT = {"HotelBill": "hotel", "MLHotelTicket": "hotel", "FlightItinerary": "flight", "FlightTicket": "flight", "AirTicket": "flight", "MLFlightTicket": "flight",
            "TaxiTicket": "transport", "Taxi": "transport", "MLTaxiTicket": "transport", "SubwayTicket": "transport", "Subway": "transport",
            "TrainTicket": "train", "MLTrainTicket": "train", "BusTicket": "bus", "CarBusTicket": "bus", "MLBusTicket": "bus", "FuelFee": "fuel",
            "ParkingFee": "parking", "Parking": "parking", "InternetFee": "internet", "PhoneFee": "internet", "UtilityFee": "utility", "Medical": "medical",
            "DiningTicket": "food", "DiningPurchaseTicket": "food"}
RB = PUBLIC / "receiptbench"

def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())[:30]

def pick(items, cap, rng):  # items: (id, seller, country); round-robin over sellers, countries interleaved
    by = collections.defaultdict(list)
    for i in sorted(items): by[norm(i[1]) or i[0]].append(i)
    ks = sorted(by); rng.shuffle(ks); ks.sort(key=lambda k: by[k][0][2])  # stable: spreads countries
    out = []
    for r in range(max(map(len, by.values()))):
        for k in ks:
            if r < len(by[k]) and len(out) < cap: out.append(by[k][r])
    return out

def receiptbench():
    from huggingface_hub import hf_hub_download, HfFileSystem
    rng = random.Random(SEED); RB.mkdir(parents=True, exist_ok=True)
    jz = zipfile.ZipFile(hf_hub_download("T0rl/ReceiptBench", "ReceiptBench_json_v1.zip", repo_type="dataset", local_dir=RB))
    meta = {}
    for n in jz.namelist():
        if n.endswith(".json") and not n.startswith("__MACOSX"):
            b = n.split("/")[-1][:-5]; t = b.split("_")[1]
            if t in TYPE2CAT:
                j = json.loads(jz.read(n)); s = j.get("seller_name"); s = (s[0] if isinstance(s, list) and s else s) or ""
                meta[b] = dict(type=t, cat=TYPE2CAT[t], seller=s.strip(), place=j.get("place") or "", cur=j.get("std_curr"), json=j)
    cand = collections.defaultdict(list)
    for b, m in meta.items(): cand[m["cat"]].append((b, m["seller"], m["place"].split("-")[0].strip()))
    chosen = {b: c for c, v in cand.items() for b, *_ in pick(v, CAPS[c], rng)}
    fs = HfFileSystem(); z = zipfile.ZipFile(fs.open("datasets/T0rl/ReceiptBench/ReceiptBench_image_v1.zip", "rb", block_size=1 << 20))
    files = {i.filename.split("/")[-1].rsplit(".", 1)[0]: i for i in z.infolist() if not i.filename.startswith("__MACOSX") and not i.is_dir()}
    (RB / "images").mkdir(exist_ok=True); (RB / "json").mkdir(exist_ok=True); sel = []
    for b in sorted(chosen):
        if b not in files: continue
        info = files[b]; out = RB / "images" / f"{b}.{info.filename.rsplit('.', 1)[-1].lower()}"
        if not out.exists(): out.write_bytes(z.read(info))
        (RB / "json" / f"{b}.json").write_text(json.dumps(meta[b]["json"], ensure_ascii=False))
        sel.append({"id": b, "image": str(out.relative_to(DS)), "type": meta[b]["type"], "category_hint": meta[b]["cat"], "seller": meta[b]["seller"], "place": meta[b]["place"], "currency": meta[b]["cur"]})
    dump(RB / "selection.json", sel); print("receiptbench selected", len(sel), collections.Counter(s["category_hint"] for s in sel))

INDIA_CAPS = {"hotel": 60, "flight": 48, "bus": 52, "transport": 60, "miscellaneous": 40}  # all India-tagged ReceiptBench docs not already selected
INDIA_TYPES = dict(TYPE2CAT, PurchaseTicket="miscellaneous", MLPurchaseTicket="miscellaneous")
def receiptbench_india():
    """Extra pass: Indian ReceiptBench receipts (place/country India) for the categories that had no real Indian data. Appends to selection.json."""
    from huggingface_hub import hf_hub_download, HfFileSystem
    rng = random.Random(SEED); selp = RB / "selection.json"; sel = json.loads(selp.read_text()); have = {x["id"] for x in sel}
    jz = zipfile.ZipFile(hf_hub_download("T0rl/ReceiptBench", "ReceiptBench_json_v1.zip", repo_type="dataset", local_dir=RB)); meta = {}
    for n in jz.namelist():
        if n.endswith(".json") and not n.startswith("__MACOSX"):
            b = n.split("/")[-1][:-5]; t = b.split("_")[1]
            if t in INDIA_TYPES and INDIA_TYPES[t] in INDIA_CAPS and b not in have:
                j = json.loads(jz.read(n)); place = j.get("place") or ""
                if place.split("-")[0].strip().lower() in ("india", "ind", "in"):
                    sl = j.get("seller_name"); sl = (sl[0] if isinstance(sl, list) and sl else sl) or ""
                    meta[b] = dict(type=t, cat=INDIA_TYPES[t], seller=sl.strip(), place=place, cur=j.get("std_curr"), json=j)
    cand = collections.defaultdict(list)
    for b, m in meta.items(): cand[m["cat"]].append((b, m["seller"], m["place"]))
    chosen = [b for c, v in cand.items() for b, *_ in pick(v, INDIA_CAPS[c], rng)]
    z = zipfile.ZipFile(HfFileSystem().open("datasets/T0rl/ReceiptBench/ReceiptBench_image_v1.zip", "rb", block_size=1 << 20))
    files = {i.filename.split("/")[-1].rsplit(".", 1)[0]: i for i in z.infolist() if not i.filename.startswith("__MACOSX") and not i.is_dir()}
    for b in sorted(chosen):
        if b not in files: continue
        info = files[b]; out = RB / "images" / f"{b}.{info.filename.rsplit('.', 1)[-1].lower()}"
        if not out.exists(): out.write_bytes(z.read(info))
        (RB / "json" / f"{b}.json").write_text(json.dumps(meta[b]["json"], ensure_ascii=False))
        sel.append({"id": b, "image": str(out.relative_to(DS)), "type": meta[b]["type"], "category_hint": meta[b]["cat"], "seller": meta[b]["seller"], "place": meta[b]["place"], "currency": meta[b]["cur"]})
    dump(selp, sel); print("india pass added", len(chosen), collections.Counter(meta[b]["cat"] for b in chosen))

PT_RULES = {"fuel": r"galp|repsol|\bbp\b|cepsa|prio\b|petrogal|posto |combust|abastec", "medical": r"farm[aá]c|pharm|cl[ií]nic|hospital|sa[uú]de|dent|laborat|m[eé]dic",
            "food": r"restaur|caf[eé]|pastelar|snack|churrasq|tasca|cervej|padaria|pizza|burger|mcdonald|sushi|taberna|cantina|bar\b|gelat",
            "parking": r"parque|empark|emel|parking|estacionamento", "hotel": r"\bhotel\b|hostel|residencial|alojamento|pousada"}  # 'hotelaria' (catering) excluded below
def pt():
    from huggingface_hub import snapshot_download
    p = snapshot_download("Francisco-Cruz/InvoicesReceiptsPT", repo_type="dataset", allow_patterns=["2_Annotations_Json/*", "README.md"], local_dir=INTERIM / "pt_ann")
    d = PUBLIC / "invoices_receipts_pt"; (d / "images").mkdir(parents=True, exist_ok=True); per = collections.Counter(); sel = []
    for f in sorted((INTERIM / "pt_ann/2_Annotations_Json").glob("*.txt")):
        j = json.loads(f.read_text()); c = (j.get("company") or "").lower()
        if "hotelaria" in c: continue
        for cat, rx in PT_RULES.items():
            if re.search(rx, c):
                if per[norm(c)] >= 6: break
                per[norm(c)] += 1; sel.append((f.stem, cat, c, j)); break
    from huggingface_hub import hf_hub_download
    out = []
    for stem, cat, c, j in sel:
        for ext in ("jpg", "jpeg", "png"):
            try: src = hf_hub_download("Francisco-Cruz/InvoicesReceiptsPT", f"1_Images/{stem}.{ext}", repo_type="dataset", local_dir=d / "_dl"); break
            except Exception: src = None
        if not src: continue
        dst = d / "images" / os.path.basename(src); os.replace(src, dst)
        (d / "annotations").mkdir(exist_ok=True); (d / "annotations" / f"{stem}.json").write_text(json.dumps(j, ensure_ascii=False))
        out.append({"id": stem, "image": str(dst.relative_to(DS)), "category_hint": cat, "company": c, "fields": j})
    dump(d / "selection.json", out); print("pt selected", len(out), collections.Counter(o["category_hint"] for o in out))

if __name__ == "__main__":
    receiptbench(); receiptbench_india(); pt()
