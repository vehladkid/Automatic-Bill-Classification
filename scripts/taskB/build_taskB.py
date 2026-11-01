"""Builds the canonical Task B dataset from the existing manifest + annotation files (+ synthetic_b).
 source schema -> canonical schema mapping is explicit (MAPPING below); unmappable source fields are listed as UNSUPPORTED, never guessed.
 Outputs: datasets/processed/taskB/{train,validation,test}.jsonl (doc rows with words, boxes, BIO labels, canonical fields, support set),
          reports/taskB/annotation_audit.json + .md, reports/taskB/schema_mapping.md, models/taskB/field_schema.json.
 Splits: the manifest's group-based split is reused unchanged (synthetic_b documents are independent renders -> seeded document-level 80/10/10)."""
import json, re, random, collections
from PIL import Image
from schema import *
from words import *

SPLIT_NAME = {"train": "train", "val": "validation", "test": "test"}
MAPPING = {  # source -> {canonical: source field / rule}
 "sroie": {"merchant_name": "company", "merchant_address": "address", "invoice_date": "date", "total_amount": "total"},
 "cord": {"total_amount": "total_amount", "subtotal": "subtotal", "tax_total": "tax_amount"},
 "receiptbench": {"merchant_name": "seller_name", "invoice_number": "invoice_number", "invoice_date": "std_invoice_time", "total_amount": "std_total, or orig_total when std_total is the placeholder '0.00' (182 docs); neither usable -> total not scored", "currency": "std_curr", "gstin": "tax_number (only if GSTIN-format)",
                  "cgst/sgst/igst": "detail[] lines whose content contains CGST/SGST(UTGST)/IGST (India only)", "origin/destination": "departure/arrival (flight/train/bus types; city part)", "travel_date": "std_start_time (flight/train/bus)", "check_in/check_out": "std_start_time/std_end_time (hotel)"},
 "invoices_pt": {"merchant_name": "company", "merchant_address": "address", "invoice_date": "date", "total_amount": "total", "invoice_number": "invoice_number", "tax_total": "iva_amount"},
 "synthetic_in": {"merchant_name": "merchant", "merchant_address": "address", "invoice_number": "invoice_number", "invoice_date": "date", "subtotal": "subtotal", "tax_total": "tax_amount", "total_amount": "total_amount", "currency": "currency",
                  "payment_method": "payment_method", "gstin": "GSTIN", "cgst/sgst/igst": "tax_split", "origin/destination/pnr/train_number/flight_number/vehicle_number": "origin/destination/pnr/train_no/flight_no/vehicle_no"},
 "synthetic_b": {"*": "all canonical fields incl. hsn_sac, hotel stay, medical, employee, passenger/ticket/bus (generated with the rendered text)"},
}
UNSUPPORTED = {"sroie": ["(none: only 4 fields annotated)"], "cord": ["menu items (line items, not a Task B field)", "payment_cash (cash tendered, not payment method)"],
               "receiptbench": ["seller_address (city-level 'IN-Bengaluru', not a street address)", "tax_number when not GSTIN-format (foreign tax ids)", "std_end_time for non-hotel types", "detail line items", "tax_total/subtotal (not separable from detail)"],
               "invoices_pt": ["nif_buyer / nif_seller (tax ids; no canonical field)"], "synthetic_in": ["customer_name (role depends on category)", "phone", "round_off", "consumer_no/account_no/landlord(_pan)/units/upi_ref (no canonical field)"],
               "funsd": ["form entities (non-expense forms; not a receipt schema)"], "all real sources": ["hsn_sac, employee_*, pnr, ticket_number, passenger_name, flight/train/bus/vehicle number, room_number, number_of_nights, patient/doctor/prescription/pharmacy: NO real annotation exists"]}
SUPPORT = {  # fields whose absence is meaningful (evaluated) per source; others are not evaluated for that source
 "sroie": ["merchant_name", "merchant_address", "invoice_date", "total_amount"], "cord": ["total_amount", "subtotal", "tax_total"],
 "invoices_pt": ["merchant_name", "merchant_address", "invoice_date", "total_amount", "invoice_number", "tax_total"],
 "synthetic_in": ["merchant_name", "merchant_address", "invoice_number", "invoice_date", "subtotal", "tax_total", "total_amount", "currency", "payment_method", "gstin", "cgst", "sgst", "igst", "origin", "destination", "pnr", "train_number", "flight_number", "vehicle_number"],
 "synthetic_b": list(FIELDS)}
RB_BASE = ["merchant_name", "invoice_number", "invoice_date", "total_amount", "currency", "gstin"]
KW = {"total_amount": r"total|amount|due|payable|grand|net|balance|jumlah|bill", "subtotal": r"sub ?total|taxable|net|before|amount", "tax_total": r"tax|gst|vat|iva|service", "cgst": r"cgst|c\.g\.s\.t", "sgst": r"sgst|utgst|s\.g\.s\.t", "igst": r"igst|i\.g\.s\.t"}
DAYFIRST_OFF = {"US", "CA", "PH"}

def doc_ctx(row):
    return dict(locale={"cord": "idr", "invoices_pt": "eu"}.get(row["source_dataset"], "default"), dayfirst=row["country"] not in DAYFIRST_OFF)

# ---------- source readers -> {field: raw value}  (raw values exactly as annotated) ----------
def read_fields(row, ann):
    s, f = row["source_dataset"], ann.get("fields") or {}; out = {}
    if s == "sroie": out = {"merchant_name": f.get("company"), "merchant_address": f.get("address"), "invoice_date": f.get("date"), "total_amount": f.get("total")}
    elif s == "cord": out = {"total_amount": f.get("total_amount"), "subtotal": f.get("subtotal"), "tax_total": f.get("tax_amount")}
    elif s == "invoices_pt": out = {"merchant_name": f.get("company"), "merchant_address": f.get("address"), "invoice_date": f.get("date"), "total_amount": f.get("total"), "invoice_number": f.get("invoice_number"), "tax_total": f.get("iva_amount")}
    elif s == "receiptbench":
        full = json.loads((PUBLIC_RB / f"{row['source_id']}.json").read_text()) if (PUBLIC_RB / f"{row['source_id']}.json").exists() else {}
        sn = full.get("seller_name"); sn = (sn[0] if isinstance(sn, list) and sn else sn)
        st = parse_amount(full.get("std_total")); ot = parse_amount(full.get("orig_total"))
        tot = full.get("std_total") if st not in (None, 0.0) else (full.get("orig_total") if ot not in (None, 0.0) else None)   # ReceiptBench writes std_total='0.00' when unknown; orig_total holds the printed value (182 docs)
        out = {"merchant_name": sn, "invoice_number": full.get("invoice_number"), "invoice_date": full.get("std_invoice_time"), "total_amount": tot, "currency": full.get("std_curr")}
        out["_total_unknown"] = tot is None
        tn = norm_gstin(full.get("tax_number") or ""); out["gstin"] = tn if tn and gstin_valid_format(tn) else None
        det = full.get("detail") or []
        if row["country"] == "IN" and det:
            for d in det:
                c = str(d.get("content", "")).lower(); a = d.get("amount")
                for fld, rx in (("cgst", r"cgst"), ("sgst", r"sgst|utgst"), ("igst", r"igst")):
                    if re.search(rx, c) and parse_amount(a) not in (None, 0.0) and fld not in out: out[fld] = a
        typ = (full.get("type") or "") + row["source_id"]; cat = row["expense_category"]
        if cat in ("flight", "train", "bus"):
            for k, a in (("origin", "departure"), ("destination", "arrival")):
                v = full.get(a); v = " ".join(v) if isinstance(v, list) else v
                out[k] = v or None
            out["travel_date"] = full.get("std_start_time") or None
        if cat == "hotel": out["check_in"] = full.get("std_start_time") or None; out["check_out"] = full.get("std_end_time") or None
    elif s == "synthetic_in":
        ts = f.get("tax_split") or {}
        out = {"merchant_name": f.get("merchant"), "merchant_address": f.get("address"), "invoice_number": f.get("invoice_number"), "invoice_date": f.get("date"), "subtotal": f.get("subtotal"), "tax_total": f.get("tax_amount"), "total_amount": f.get("total_amount"),
               "currency": f.get("currency"), "payment_method": f.get("payment_method"), "gstin": f.get("GSTIN"), "cgst": ts.get("CGST"), "sgst": ts.get("SGST"), "igst": ts.get("IGST"), "origin": f.get("origin"), "destination": f.get("destination"),
               "pnr": f.get("pnr"), "train_number": f.get("train_no"), "flight_number": f.get("flight_no"), "vehicle_number": f.get("vehicle_no")}
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}
PUBLIC_RB = DS / "public/receiptbench/json"
def support_for(row):
    s = row["source_dataset"]
    if s == "receiptbench":
        sup = list(RB_BASE) + ((["cgst", "sgst", "igst"]) if row["country"] == "IN" else [])
        if row["expense_category"] in ("flight", "train", "bus"): sup += ["origin", "destination", "travel_date"]
        if row["expense_category"] == "hotel": sup += ["check_in", "check_out"]
        return sup
    return SUPPORT.get(s, [])

# ---------- alignment ----------
def line_of(words, i):
    c = (words[i]["bbox"][1] + words[i]["bbox"][3]) / 2; tol = 0.6 * max(words[i]["h"], 6)
    return [j for j, w in enumerate(words) if abs((w["bbox"][1] + w["bbox"][3]) / 2 - c) <= tol]
def align_text(words, gt_norm, field):
    nw = [norm_text(w["text"]) for w in words]; n = len(words); best = (0.0, None); L = len(gt_norm)
    if L == 0: return None
    for i in range(n):
        acc = ""
        for j in range(i, min(n, i + 40)):
            acc += nw[j]
            if acc == gt_norm: return list(range(i, j + 1))
            if len(acc) > L * 1.25 + 3: break
            if len(acc) >= L * 0.8 and L >= 6:
                r = similarity(acc, gt_norm)
                if r > best[0]: best = (r, list(range(i, j + 1)))
    return best[1] if best[0] >= 0.88 else None
def align_id(words, gt_norm):
    nw = [norm_id(w["text"]) for w in words]; n = len(words)
    for i in range(n):
        acc = ""
        for j in range(i, min(n, i + 6)):
            acc += nw[j]
            if acc == gt_norm: return list(range(i, j + 1))
            if len(acc) >= len(gt_norm): break
    return None
def align_amount(words, text, offs, field, gt, locale):
    cands = [c for c in amount_candidates(text, offs, locale) if amount_eq(c[0], gt)]
    if not cands: return None
    kw = re.compile(KW.get(field, "total"), re.I); keyed = [c for c in cands if c[1] and kw.search(" ".join(words[j]["text"] for j in line_of(words, c[1][0])))]
    if keyed: return sorted({i for c in keyed for i in c[1]})
    return sorted({i for c in cands for i in c[1]}) if len(cands) <= 2 else None
def align_date(words, text, offs, gt, dayfirst):
    cands = [c for c in date_candidates(text, offs, dayfirst) + date_candidates(text, offs, not dayfirst) if c[0] == gt]
    if not cands: return None
    key = [c for c in cands if re.search(r"date|dt|issued|check|arriv|depart", " ".join(words[j]["text"] for j in line_of(words, c[1][0])), re.I)]
    return (key or cands)[0][1]
CURSYM = {"\u20b9": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rm": "MYR", "myr": "MYR", "idr": "IDR", "rp": "IDR", "rp.": "IDR", "$": "USD", "usd": "USD", "\u20ac": "EUR", "eur": "EUR", "\u00a3": "GBP", "gbp": "GBP", "aed": "AED", "sgd": "SGD", "cad": "CAD", "aud": "AUD"}
DOLLAR = {"USD", "CAD", "AUD", "SGD", "NZD", "HKD", "MXN"}
def align_currency(words, norm):
    hit = []
    for i, w in enumerate(words):
        t = re.sub(r"[\d,.]+$", "", w["text"].lower()).strip(); t0 = re.sub(r"^[\d,.]+", "", t)
        code = CURSYM.get(t0) or CURSYM.get(t0.rstrip(".")) or (CURSYM.get(t0[:1]) if t0[:1] in ("\u20b9", "$", "\u20ac", "\u00a3") else None)
        if code and (code == norm or (code == "USD" and norm in DOLLAR)): hit.append(i)
    return hit[:3] or None
def align_field(words, text, offs, field, norm, raw, ctx):
    t = FIELDS[field]
    if field == "currency": return align_currency(words, norm)
    if t == "amount": return align_amount(words, text, offs, field, norm, ctx["locale"])
    if t == "date": return align_date(words, text, offs, norm, ctx["dayfirst"])
    if t == "gstin" or t == "id": return align_id(words, norm_id(norm if t == "gstin" else raw) if t == "gstin" else norm_id(raw))
    if t == "place":
        r = align_text(words, norm_place(raw), field); return r
    if t == "enum":
        for i in range(len(words)):
            for n in (2, 1):
                if i + n <= len(words) and norm and norm_pm(" ".join(w["text"] for w in words[i:i + n])) == norm: return list(range(i, i + n))
        return None
    if t == "int":
        return next(([i] for i, w in enumerate(words) if re.fullmatch(str(norm), w["text"])), None) if norm is not None else None
    return align_text(words, norm_text(raw), field)

PRIORITY = ["gstin", "hsn_sac", "pnr", "ticket_number", "cgst", "sgst", "igst", "tax_total", "subtotal", "total_amount", "invoice_date", "travel_date", "check_in", "check_out", "invoice_number", "room_number", "number_of_nights",
            "employee_id", "train_number", "flight_number", "bus_number", "vehicle_number", "prescription_number", "payment_method", "currency", "origin", "destination", "employee_name", "passenger_name", "patient_name", "doctor_name", "merchant_name", "pharmacy_name", "merchant_address"]
TAGS = ["O"] + [f"{p}-{f}" for f in FIELDS for p in ("B", "I")]

def build_doc(row, ann, raw_fields, image_wh):
    ctx = doc_ctx(row); words = reading_order(split_words(ann.get("words") or []))
    unknown_total = bool(raw_fields.pop("_total_unknown", False))
    text, offs = text_view(words); sup = [f for f in support_for(row) if not (unknown_total and f == "total_amount")]; fields = {}; labels = ["O"] * len(words); dropped = []
    for f in PRIORITY:
        if f not in raw_fields or f not in sup: continue
        raw = raw_fields[f]; raw_s = raw["value"] if isinstance(raw, dict) else raw
        norm = (raw.get("norm") if isinstance(raw, dict) else None)
        if norm is None: norm = normalize(f, raw_s, ctx["locale"], ctx["dayfirst"])
        if f == "currency": norm = str(raw_s).upper()
        if norm is None: continue
        idx = align_field(words, text, offs, f, norm, raw_s, ctx) if words else None
        shared = f == "currency"      # currency symbols sit inside amount spans: evidence only, never token-labelled
        if idx is not None and not shared:
            keep = [i for i in idx if labels[i] == "O"]
            if not keep and any(set(idx) <= set(v["word_idx"]) for v in fields.values()): shared = True      # same span already labelled by an equal field (pharmacy_name == merchant_name)
            else: idx = keep or None
        if idx and not shared:
            idx = sorted(idx); prev = None
            for i in idx: labels[i] = ("I-" if prev is not None and i == prev + 1 else "B-") + f; prev = i
        elif row["synthetic"] and not shared and f != "currency": dropped.append(f); continue       # synthetic OCR is exact: a value that is not rendered is absent from the document
        fields[f] = dict(value=raw_s, norm=norm, aligned=bool(idx), word_idx=idx or [], bbox=[min(words[i]["bbox"][0] for i in idx), min(words[i]["bbox"][1] for i in idx), max(words[i]["bbox"][2] for i in idx), max(words[i]["bbox"][3] for i in idx)] if idx else None)
    return dict(document_id=row["document_id"], split=row["split"], source_dataset=row["source_dataset"], synthetic=row["synthetic"], country=row["country"], expense_category=row["expense_category"], layout_id=row.get("layout_id"),
                image=row["file_path"], width=image_wh[0], height=image_wh[1], supported_fields=sup, fields=fields, words=[w["text"] for w in words], boxes=[w["bbox"] for w in words], labels=labels, dropped_unrendered=dropped)

def main():
    rows = [json.loads(l) for l in open(META / "manifest.jsonl")]; docs = []
    for r in rows:
        if r["source_dataset"] not in ("sroie", "cord", "receiptbench", "invoices_pt", "synthetic_in") or not r["ann_path"]: continue
        ann = load_ann(r["ann_path"]); raw = read_fields(r, ann)
        d = build_doc(r, ann, raw, (r["width"], r["height"]))
        if d["fields"] or d["supported_fields"]: docs.append(d)
    # synthetic_b: seeded document-level split (independent renders)
    rng = random.Random(SEED); sb = [json.loads(l) for l in open(DS / "synthetic_b/annotations.jsonl")]; rng.shuffle(sb)
    for i, a in enumerate(sb):
        sp = "train" if i % 10 < 8 else ("val" if i % 10 == 8 else "test")
        w, h = Image.open(DS / "synthetic_b" / a["image"]).size
        row = dict(document_id="synthetic_b__" + a["source_id"], split=sp, source_dataset="synthetic_b", synthetic=True, country="IN", expense_category=a["expense_category"], file_path="synthetic_b/" + a["image"], layout_id=None, source_id=a["source_id"])
        d = build_doc(row, {"words": a["words"], "fields": {}}, a["fields"], (w, h)); docs.append(d)
    out = {s: [] for s in SPLIT_NAME.values()}
    for d in docs: out[SPLIT_NAME[d["split"]]].append(d)
    (PROCESSED / "taskB").mkdir(parents=True, exist_ok=True)
    for s, v in out.items(): (PROCESSED / "taskB" / f"{s}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in v) + "\n")
    # ---- audit ----
    aud = {}; per_src = collections.defaultdict(lambda: collections.Counter())
    for f in FIELDS:
        a = dict(total=0, real=0, synthetic=0, indian_real=0, aligned=0, sources=collections.Counter())
        for d in docs:
            if f in d["fields"]:
                x = d["fields"][f]; a["total"] += 1; a["synthetic"] += d["synthetic"]; a["real"] += (not d["synthetic"]); a["indian_real"] += (not d["synthetic"] and d["country"] == "IN"); a["aligned"] += x["aligned"]; a["sources"][d["source_dataset"]] += 1
        a["sources"] = dict(a["sources"]); a["eval_docs_real"] = sum(1 for d in docs if f in d["supported_fields"] and not d["synthetic"]); a["eval_docs_synth"] = sum(1 for d in docs if f in d["supported_fields"] and d["synthetic"]); aud[f] = a
    summ = dict(total_docs=len(docs), real=sum(not d["synthetic"] for d in docs), synthetic=sum(d["synthetic"] for d in docs), indian_real=sum((not d["synthetic"]) and d["country"] == "IN" for d in docs),
                by_source=dict(collections.Counter(d["source_dataset"] for d in docs)), by_split={s: len(v) for s, v in out.items()}, dropped_unrendered={f: sum(f in d["dropped_unrendered"] for d in docs) for f in FIELDS if any(f in d["dropped_unrendered"] for d in docs)})
    (REP := ROOT / "reports/taskB").mkdir(parents=True, exist_ok=True)
    (REP / "annotation_audit.json").write_text(json.dumps({"summary": summ, "fields": aud, "mapping": MAPPING, "unsupported": UNSUPPORTED, "support_sets": SUPPORT}, indent=1))
    md = ["# Task B annotation audit (generated by scripts/taskB/build_taskB.py)", "", f"Documents: {summ['total_docs']} (real {summ['real']}, synthetic {summ['synthetic']}, Indian real {summ['indian_real']}). By split: {summ['by_split']}. By source: {summ['by_source']}.", "",
          "`annotated` = documents with a GT value for the field; `aligned` = of those, the value could be located in the OCR words (the rest can be scored at value level but give no token labels: OCR/alignment gap). Synthetic values that are not rendered are dropped as absent.", "",
          "| Field | Total annotated | Real | Synthetic | Indian real | Aligned to OCR | Source datasets |", "|---|---:|---:|---:|---:|---:|---|"]
    for f, a in aud.items(): md.append(f"| {f} | {a['total']} | {a['real']} | {a['synthetic']} | {a['indian_real']} | {a['aligned']} | {', '.join(f'{k}:{v}' for k, v in a['sources'].items()) or '-'} |")
    md += ["", "Fields with ZERO real annotation: " + ", ".join(f for f, a in aud.items() if a["real"] == 0), "", "Fields with real Indian annotation: " + ", ".join(f"{f} ({a['indian_real']})" for f, a in aud.items() if a["indian_real"])]
    (REP / "annotation_audit.md").write_text("\n".join(md))
    sm = ["# Source schema -> canonical Task B schema", ""]
    for s, m in MAPPING.items(): sm += [f"## {s}"] + [f"* `{k}` <- {v}" for k, v in m.items()] + ([f"* UNSUPPORTED: " + "; ".join(UNSUPPORTED[s])] if s in UNSUPPORTED else []) + [""]
    sm += ["## Not mapped anywhere"] + [f"* {k}: {'; '.join(v)}" for k, v in UNSUPPORTED.items() if k in ("funsd", "all real sources")]
    (REP / "schema_mapping.md").write_text("\n".join(sm))
    (ROOT / "models/taskB").mkdir(parents=True, exist_ok=True)
    (ROOT / "models/taskB/field_schema.json").write_text(json.dumps({"fields": FIELDS, "groups": GROUPS, "tags": TAGS, "payment_methods": PAYMENT, "mapping": MAPPING, "support_sets": SUPPORT}, indent=1))
    print(json.dumps(summ)); print("\n".join(md[6:6 + len(FIELDS) + 4]))
if __name__ == "__main__": main()
