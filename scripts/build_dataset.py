"""Ingest every source -> canonical manifest (datasets/metadata/manifest_raw.jsonl) + per-doc annotations
(datasets/interim/annotations/<document_id>.json: {"words":[{"text","bbox":[x0,y0,x1,y1]}], "fields":{...}, "ocr_source":...}).
Originals are never modified. Parquet sources are decoded to datasets/interim/<src>/images/ (lossless pixels, JPEG q95)."""
import io, json, re, glob, hashlib, collections
import pandas as pd
from PIL import Image
from common import *

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RVL = ["letter", "form", "email", "handwritten", "advertisement", "scientific_report", "scientific_publication", "specification",
       "file_folder", "news_article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]
(ANN := INTERIM / "annotations").mkdir(parents=True, exist_ok=True)
rows = []

def add(source, source_id, path, *, category, doc_type=None, sub=None, quality, qsrc, lang=None, country="unknown", currency=None,
        synthetic=False, collection, license, ann=None, group_hint=None, origin_split=None, **extra):
    doc_id = f"{source}__{re.sub(r'[^A-Za-z0-9_.-]', '_', str(source_id))}"
    ocr = bool(ann and ann.get("words")); fld = bool(ann and ann.get("fields"))
    if ann: dump(ANN / f"{doc_id}.json", ann)
    lang = extra.get("language") or lang
    rows.append(dict(document_id=doc_id, source_dataset=source, source_id=str(source_id), file_path=str(Path(path).relative_to(DS)),
                     document_type=doc_type or DEFAULT_DOC_TYPE.get(category, "other"), expense_category=category,
                     parent_group=CATEGORIES.get(category, "unlabeled"), subcategory=sub, label_quality=quality, label_source=qsrc,
                     language=lang, country=country, currency=currency, synthetic=synthetic, collection=collection,
                     ocr_available=ocr, ocr_source=(ann or {}).get("ocr_source"), field_annotations_available=fld,
                     ann_path=f"interim/annotations/{doc_id}.json" if ann else None, license=license,
                     group_hint=group_hint, origin_split=origin_split,
                     **{k: extra.get(k) for k in ("source_url", "merchant", "layout_id", "country_source", "language_source", "batch", "source_type", "n_pages", "hard_negative")}))

def save_img(b, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists(): Image.open(io.BytesIO(b)).convert("RGB").save(out, "JPEG", quality=95)

def bbox(q): xs = [q[k] for k in q if k.startswith("x")]; ys = [q[k] for k in q if k.startswith("y")]; return [min(xs), min(ys), max(xs), max(ys)]

# ---- 1. user-collected Indian images (folder name == label) -------------------------------------------------------
ISO = {"us": "US", "usa": "US", "united states": "US", "ca": "CA", "canada": "CA", "de": "DE", "germany": "DE", "india": "IN", "ind": "IN", "in": "IN", "uk": "GB", "united kingdom": "GB",
       "gb": "GB", "my": "MY", "malaysia": "MY", "jp": "JP", "japan": "JP", "cn": "CN", "china": "CN", "hk": "HK", "ee": "EE", "es": "ES", "nl": "NL", "se": "SE", "fr": "FR", "lt": "LT",
       "pl": "PL", "at": "AT", "hr": "HR", "au": "AU", "australia": "AU", "indonesia": "ID", "id": "ID", "egypt": "EG", "singapore": "SG", "sg": "SG", "uae": "AE", "ae": "AE", "bangladesh": "BD",
       "saudi arabia": "SA", "can": "CA", "aus": "AU", "australian": "AU", "mys": "MY", "malaysian": "MY", "hungary": "HU", "kenya": "KE", "united arab emirates": "AE", "nz": "NZ", "cz": "CZ", "gr": "GR", "be": "BE", "ph": "PH", "ir": "IR", "philippines": "PH", "pt": "PT", "portugal": "PT", "thailand": "TH", "mexico": "MX", "brazil": "BR", "france": "FR", "italy": "IT", "spain": "ES", "turkey": "TR"}
def iso(x): x = (x or "").strip().lower(); return ISO.get(x, x.upper() if 2 <= len(x) <= 3 and x else (x.title() if x else None)) or None

def batches():
    f = META / "user_batch_2026-10-04.json"; m = {}
    if f.exists():
        for cat, b in json.loads(f.read_text()).items():
            for n in b["files"]: m[(b["unpacked_to"], n)] = "user_batch_2026-10-04"
    return m

def manual():
    bt = batches()
    for d in sorted(INDIAN.iterdir()):
        if not d.is_dir(): continue
        if d.name not in CATEGORIES: print("skip unknown folder", d); continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT and not p.name.startswith("_"):
                add("manual_in", f"{d.name}_{p.stem}_{p.suffix[1:]}".replace(" ", "_"), p, category=d.name, quality="manual_folder", qsrc="folder name chosen by project owner",
                    country="IN", country_source="project owner: Indian bills", currency="INR", collection="manual_indian", language="en", language_source="assumed (not detected)", source_type="photo_or_screenshot",
                    batch=bt.get((f"datasets/indian/{d.name}/", p.name), "user_batch_2026-09"), license="project-owner collected; redistribution rights unverified")

def real_user():  # datasets/real/<category>/ : user-supplied real FOREIGN (or non-Indian) bills; _provenance.json beside them
    if not (DS / "real").exists(): return
    for d in sorted(p for p in (DS / "real").iterdir() if p.is_dir()):
        pv = json.loads((d / "_provenance.json").read_text()) if (d / "_provenance.json").exists() else {}
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT and d.name in CATEGORIES:
                add("user_real", f"{d.name}_{p.stem}_{p.suffix[1:]}".replace(" ", "_"), p, category=d.name, quality="manual_folder", qsrc="folder name chosen by project owner",
                    country=pv.get("country"), country_source=pv.get("country_source"), currency="USD" if pv.get("country") == "US" else None, collection="user_real_foreign",
                    language=pv.get("language"), language_source="from _provenance.json", batch="user_batch_2026-10-04", source_type="photo", license=pv.get("license", "user-supplied"))

# ---- 2. Kaggle: Indian grocery GST invoices ----------------------------------------------------------------------
def grocery():
    for p in sorted((PUBLIC / "indian_grocery_gst").glob("**/*.jpg")):
        add("in_grocery_gst", p.stem, p, category="miscellaneous", sub="grocery", doc_type="invoice", quality="source_label",
            qsrc="Kaggle metadata: grocery_tax_invoice", country="IN", currency="INR", collection="public_real", license="CC-BY-4.0")

# ---- 3. SROIE (ICDAR 2019, Malaysia). Category via company-name rules; unmatched -> unlabeled ----------------------
SROIE_RULES = [("fuel", r"PETRON|PETROL|SHELL|CALTEX|BHPETROL"), ("hotel", r"\bHOTEL\b|RESORT"),
               ("medical", r"PHARMACY|FARMASI|KLINIK|CLINIC|HOSPITAL"),
               ("food", r"RESTORAN|RESTAURANT|CAFE|COFFEE|PIZZA|SEAFOOD|KOPITIAM|KITCHEN|\bFOOD\b|DOMINO|\bCAKE|CONFECTIONERY|MCDONALD|\bKFC\b|BURGER|SUSHI|BBQ|DIMSUM|NASI"),
               ("miscellaneous", r"STATIONERY|HARDWARE|ELECTRICAL|\bPRINT|COMPUTER|OFFICE")]
def sroie():
    base = PUBLIC / "sroie/extracted/SROIE2019"
    for split in ("train", "test"):
        for img in sorted((base / split / "img").glob("*.jpg")):
            ent = base / split / "entities" / f"{img.stem}.txt"; box = base / split / "box" / f"{img.stem}.txt"
            f = json.loads(ent.read_text()) if ent.exists() else {}
            words = []
            if box.exists():
                for ln in box.read_text(errors="ignore").splitlines():
                    t = ln.split(",", 8)
                    if len(t) == 9:
                        try: xs, ys = [int(v) for v in t[0:8:2]], [int(v) for v in t[1:8:2]]
                        except ValueError: continue
                        words.append({"text": t[8], "bbox": [min(xs), min(ys), max(xs), max(ys)]})
            comp = (f.get("company") or "").upper(); cat, q = "unlabeled", "unlabeled"
            for c, rx in SROIE_RULES:
                if re.search(rx, comp): cat, q = c, "rule_high"; break
            add("sroie", img.stem, img, category=cat, doc_type="receipt" if cat != "fuel" else "fuel_receipt", quality=q,
                qsrc="regex on SROIE company field" if q == "rule_high" else "no rule matched", country="MY", currency="MYR",
                collection="public_real", license="ICDAR2019 SROIE: research use; see docs", origin_split=split,
                group_hint="co:" + re.sub(r"[^A-Z]", "", comp)[:25],
                ann={"words": words, "fields": {k: v for k, v in f.items()}, "ocr_source": "ground_truth"})

# ---- 4. CORD v2 (Indonesia, CC-BY-4.0). No category labels in source -> unlabeled (IE-only) ----------------------
def cord():
    seen = collections.Counter()  # running index per split: train is spread over 4 shards
    for pq in sorted(glob.glob(str(PUBLIC / "cord/data/*.parquet"))):
        split = Path(pq).name.split("-")[0].replace("validation", "val")
        for r in pd.read_parquet(pq).itertuples():
            g = json.loads(r.ground_truth); sid = f"{split}_{seen[split]:04d}"; seen[split] += 1; out = INTERIM / "cord/images" / f"{sid}.jpg"; save_img(r.image["bytes"], out)
            words = [{"text": w["text"], "bbox": bbox(w["quad"])} for ln in g["valid_line"] for w in ln["words"]]
            p = g["gt_parse"]; tot = p.get("total", {}) if isinstance(p.get("total"), dict) else {}; sub = p.get("sub_total", {}) if isinstance(p.get("sub_total"), dict) else {}
            fields = {"total_amount": tot.get("total_price"), "subtotal": sub.get("subtotal_price"), "tax_amount": sub.get("tax_price"),
                      "payment_cash": tot.get("cashprice"), "menu": p.get("menu")}
            add("cord", sid, out, category="unlabeled", doc_type="receipt", quality="unlabeled", qsrc="CORD has no expense category", country="ID",
                currency="IDR", collection="public_real", license="CC-BY-4.0", origin_split=split,
                ann={"words": words, "fields": fields, "ocr_source": "ground_truth"})

# ---- 5. RVL-CDIP sample + FUNSD: non-expense negatives -----------------------------------------------------------
def rvl():
    for pq in sorted(glob.glob(str(PUBLIC / "rvl_cdip_160/data/*.parquet"))):
        split = Path(pq).name.split("-")[0]
        for i, r in enumerate(pd.read_parquet(pq).itertuples()):
            name = RVL[r.label]; sid = f"{split}_{i:03d}_{name}"; out = INTERIM / "rvl_cdip/images" / f"{sid}.jpg"; save_img(r.image["bytes"], out)
            inv = name == "invoice"
            add("rvl_cdip", sid, out, category="unlabeled" if inv else "non_expense", doc_type="invoice" if inv else "non_expense_document", sub=name,
                quality="unlabeled" if inv else "source_label", qsrc="RVL-CDIP class " + name, country="US", currency="USD" if inv else None,
                collection="public_real", license="IIT-CDIP/RVL-CDIP: research use only", origin_split=split)

def funsd():
    for split in ("training_data", "testing_data"):
        for js in sorted((PUBLIC / f"funsd/dataset/{split}/annotations").glob("*.json")):
            form = json.loads(js.read_text())["form"]; img = js.parent.parent / "images" / f"{js.stem}.png"
            words = [{"text": w["text"], "bbox": w["box"]} for e in form for w in e["words"]]
            kv = {e["text"]: e["label"] for e in form}
            add("funsd", js.stem, img, category="non_expense", doc_type="non_expense_document", sub="form", quality="source_label", qsrc="FUNSD = scanned forms",
                country="US", collection="public_real", license="FUNSD: non-commercial research (Jaume et al. 2019)", origin_split=split,
                ann={"words": words, "fields": {"entities": kv}, "ocr_source": "ground_truth"})

# ---- 5b. Kaggle "Financial Document Image Dataset" (Google-Images scrape; DbCL-1.0 on the DB, image copyright unclear) -----
def in_financial():
    base = PUBLIC / "in_financial_docs"
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXT: continue
            util = d.name == "Utility"  # utility sub-type (electricity/water/gas) is decided later by normalize_labels.py from OCR text
            add("in_financial", f"{d.name}_{p.stem}".replace(" ", "_"), p, category="unlabeled" if util else "non_expense",
                doc_type="utility_bill" if util else "non_expense_document", sub=None if util else d.name, quality="unlabeled" if util else "source_label",
                qsrc="folder name in source dataset", country="mixed_unverified", hard_negative=not util, collection="public_real",
                license="DbCL-1.0 (database); underlying web images - academic use only")

# ---- 5c. real foreign: my_receipts (CC0), ReceiptBench subset (MIT), InvoicesReceiptsPT subset (Apache-2.0) -------
from pdf_utils import rasterize, words as pdf_words, pages as pdf_pages
FOLDER2CAT = {"hotel": ("hotel", None), "flights": ("flight", None), "taxi": ("transport", "taxi"), "restaurant": ("food", "restaurant"), "cafe": ("food", "cafe"),
              "retail": ("miscellaneous", "retail")}  # 'public transport', 'tourist attraction', 'miscellaneous' need text rules / stay unlabeled
TXT = [("train", r"\bICE\b|deutsche bahn|fahrkarte|online-ticket|zugbindung|\bIC\b|railway|rail ticket"), ("flight", r"boarding pass|lufthansa|airline|flight|flug\b|e-ticket"),
       ("hotel", r"\bhotel\b|folio|zimmer|room charge|check-?in|logis|hostel"), ("fuel", r"tankstelle|\bshell\b|\baral\b|\besso\b|benzin|diesel|super e10|fuel|gas station|petrol"),
       ("parking", r"parkhaus|parken|parking"), ("transport", r"\btaxi\b|\buber\b|\bbart\b|u-bahn|s-bahn|metro|subway|tram|fahrschein|bvg|hvv|bus ticket|\bMTR\b|octopus|transit"),
       ("medical", r"apotheke|pharmacy|pharmacie|farmacia|clinic|hospital|arzt"), ("food", r"restaurant|bewirtung|tisch|table|cafe|caf\u00e9|bar\b|pizza|burger|kellner|server|dine|guest check|rechnung.*mwst")]
def text_cat(t):
    for c, rx in TXT:
        if re.search(rx, t, re.I): return c
def my_receipts():
    base = PUBLIC / "my_receipts"; out = INTERIM / "my_receipts/images"; seen = collections.Counter()
    for pdf in sorted(p for p in base.rglob("*") if p.suffix.lower() == ".pdf"):
        rel = pdf.relative_to(base).parts; year, cc = rel[0], rel[1] if len(rel) > 2 else None
        sub = rel[2] if len(rel) > 3 else None; sid = "_".join(rel).rsplit(".", 1)[0].replace(" ", "_")
        try: img = rasterize(pdf, out / f"{sid}.jpg"); w = pdf_words(pdf); npg = pdf_pages(pdf)
        except Exception as e: print("pdf fail", pdf, str(e)[:60]); continue
        txt = " ".join(x["text"] for x in w); cat, sc, q = "unlabeled", None, "unlabeled"
        if sub in FOLDER2CAT: cat, sc = FOLDER2CAT[sub]; q = "source_label"
        else:
            tc = text_cat(txt)
            if tc: cat, q = tc, "rule_high"
        layout = "db_online_ticket" if re.search(r"online-ticket", txt, re.I) and re.search(r"fahrkarte|\bICE\b|bahn", txt, re.I) else None
        add("my_receipts", sid, img, category=cat, sub=sc, quality=q, qsrc="folder in source repo" if q == "source_label" else "regex on PDF text layer", country=iso(cc), country_source="folder code in source repo" if cc else None,
            collection="public_real", license="CC0-1.0", source_url="https://www.kaggle.com/datasets/jenswalter/receipts", language_source=None, source_type="scanned_pdf", n_pages=npg, layout_id=layout,
            doc_type=None if cat != "unlabeled" else "receipt", ann={"words": w, "fields": {}, "ocr_source": "pdf_text_layer"} if w else None,
            group_hint=("layout:" + layout) if layout else (("fn:" + fp) if (fp := re.match(r"^([A-Za-z][A-Za-z\-]{2,}?)(?=[-_]?\d{6,}|[-_][A-Za-z0-9]{6}$)", pdf.stem.replace(" ", "")) and re.match(r"^([A-Za-z][A-Za-z\-]{2,}?)(?=[-_]?\d{6,}|[-_][A-Za-z0-9]{6}$)", pdf.stem.replace(" ", "")).group(1).lower()) else None),
            origin_split=year)

UTIL = [("electricity", r"electric|energy|power|pg&e|duke|entergy|georgia power|txu|reliant|ambit|constellation|dominion"), ("gas", r"\bgas\b"), ("water", r"water"), ("internet", r"verizon|comcast|xfinity|at&t|spectrum|telecom|wireless|fiber")]
def receiptbench():
    sel = json.loads((PUBLIC / "receiptbench/selection.json").read_text())
    for s in sel:
        cat = s["category_hint"]
        if cat == "utility":
            cat = next((c for c, rx in UTIL if re.search(rx, s["seller"], re.I)), "unlabeled")
        j = json.loads((PUBLIC / f"receiptbench/json/{s['id']}.json").read_text())
        place = s["place"].split("-"); country = iso(place[0]) if place[0] else None
        dt = {"hotel": "hotel_bill", "flight": "travel_document", "train": "travel_document", "bus": "travel_document", "fuel": "fuel_receipt", "medical": "medical_bill", "internet": "utility_bill", "electricity": "utility_bill", "gas": "utility_bill", "water": "utility_bill"}.get(cat, "receipt")
        add("receiptbench", s["id"], DS / s["image"], category=cat, doc_type=dt, sub=s["type"], quality="source_label" if cat != "unlabeled" else "unlabeled", qsrc="ReceiptBench annotation 'type'" + (" + seller keyword" if s["category_hint"] == "utility" else ""),
            country=country, country_source="ReceiptBench 'place' field" if country else None, currency=s["currency"], collection="public_real", license="MIT (dataset repo); images are real personal receipts, per-image consent not documented",
            source_url="https://huggingface.co/datasets/T0rl/ReceiptBench", merchant=s["seller"] or None, source_type="photo_or_scan",
            ann={"words": [], "fields": {"std_total": j.get("std_total"), "std_invoice_time": j.get("std_invoice_time"), "seller_name": j.get("seller_name"), "tax_number": j.get("tax_number"), "invoice_number": j.get("invoice_number"), "detail": j.get("detail")}, "ocr_source": None},
            group_hint=("co:" + re.sub(r"[^a-z0-9]", "", s["seller"].lower())[:25]) if s["seller"] else None)

def pt_receipts():
    f = PUBLIC / "invoices_receipts_pt/selection.json"
    if not f.exists(): return
    for s in json.loads(f.read_text()):
        add("invoices_pt", s["id"], DS / s["image"], category=s["category_hint"], quality="rule_high", qsrc="regex on seller name field", country="PT", country_source="dataset description (Portuguese invoices)", language="pt", language_source="dataset description",
            currency="EUR", collection="public_real", license="Apache-2.0 (HF) / CC-BY-4.0 (Zenodo 6371710, 7213544)", source_url="https://zenodo.org/records/6371710", merchant=s["company"], source_type="photo",
            doc_type="receipt" if s["category_hint"] != "fuel" else "fuel_receipt", ann={"words": [], "fields": s["fields"], "ocr_source": None}, group_hint="co:" + re.sub(r"[^a-z0-9]", "", s["company"])[:25])

def medical_samples():  # 15 real Indian hospital/pharmacy bills (HF destryet/medical_bills = sansverse/medical-bill-samples); license NOT stated
    base = PUBLIC / "hf_medical_bills"; out = INTERIM / "hf_medical_bills/images"
    for pdf in sorted(base.glob("*.pdf"), key=lambda p: int(p.stem.split("_")[-1])):
        try: img = rasterize(pdf, out / f"{pdf.stem}.jpg"); npg = pdf_pages(pdf); w = pdf_words(pdf)
        except Exception as e: print("pdf fail", pdf, str(e)[:60]); continue
        add("hf_medical", pdf.stem, img, category="medical", doc_type="medical_bill", quality="source_label", qsrc="repo name 'medical_bills'; all 15 pages-1 inspected visually (hospital IPD bills, pharmacy cash bills)",
            country="IN", country_source="inferred from visible GSTIN / INR amounts (not stated by source)", currency="INR", language_source=None, collection="public_real",
            license="NONE STATED (unverified) - appears to be hackathon-style sample bills; academic use only, do not redistribute", source_url="https://huggingface.co/datasets/destryet/medical_bills",
            source_type="scanned_pdf", n_pages=npg, ann={"words": w, "fields": {}, "ocr_source": "pdf_text_layer"} if len(w) > 20 else None)

# ---- 6. synthetic Indian bills ------------------------------------------------------------------------------------
def synth():
    for ln in open(SYNTH / "annotations.jsonl"):
        a = json.loads(ln); path = SYNTH / a["image"]
        add("synthetic_in", a["source_id"], path, category=a["expense_category"], doc_type=a["document_type"], sub=a.get("subcategory"), quality="synthetic", source_type="rendered",
            layout_id=None, language="en", language_source="generator", country_source="generator", hard_negative=a.get("hard_negative", False), qsrc="generated with known category",
            country="IN", currency="INR", synthetic=True, collection="synthetic", license="generated by this project (CC0)",
            ann={"words": a["words"], "fields": a["fields"], "ocr_source": "render_ground_truth"})

if __name__ == "__main__":
    for fn in (manual, real_user, grocery, sroie, cord, rvl, funsd, in_financial, my_receipts, receiptbench, pt_receipts, medical_samples, synth):
        n = len(rows); fn(); print(f"{fn.__name__:8s} +{len(rows) - n}")
    assert len({r["document_id"] for r in rows}) == len(rows), "document_id collision"
    for r in rows:  # integrity + hashes needed for dedup
        try:
            with Image.open(DS / r["file_path"]) as im:
                r["width"], r["height"] = im.size; r["image_ok"] = True
            r["sha256"] = hashlib.sha256((DS / r["file_path"]).read_bytes()).hexdigest()
        except Exception as e:
            r.update(image_ok=False, sha256=None, width=None, height=None, error=str(e)[:100])
    META.mkdir(parents=True, exist_ok=True)
    with open(META / "manifest_raw.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("total", len(rows), "corrupt", sum(not r["image_ok"] for r in rows))
