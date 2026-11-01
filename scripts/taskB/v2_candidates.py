"""V2 step 3: supplier grouping, document typing, value tiers (A/B/C) and SUGGESTION_ONLY candidate generation for the oriented in_grocery_gst pool.
Deterministic OCR + regex only. No Task B model is used anywhere here, so no candidate is a model prediction and none is ground truth.
Outputs: datasets/interim/v2_pool/{pool_table.json, review_candidates.json}"""
import sys, json, re, collections as C
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import parse_amount, parse_date
from v2_pool_analysis import lines_of
ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "datasets" / "interim" / "v2_pool"

SUPPLIERS = [  # name regexes on the whole OCR text (OCR-tolerant); order = priority. Buyer (Lakshmi Narayan Bhandar) is the shop that photographed the invoices.
    ("SHIV SHANKAR EDIBLE", r"SHIV\s*SHANKAR|SHANKAR\s*EDIBLE|EDIBLE\s*PVT"), ("SCOOTSY LOGISTICS", r"SCOOTSY"), ("KIRTIRAAJ TRADERS", r"KIRTI\s*RAA?J"),
    ("DURGA ENTERPRISE", r"DURGA\s*ENTERPRISE"), ("MALATI TRADING", r"MALATI"), ("EKADANT AGENCY", r"EKADANT"), ("B R ENTERPRISE", r"B\s*\.?\s*R\.?\s*ENTERPRISE"),
    ("B G DISTRIBUTORS", r"B\.?\s*G\.?\s*DISTRIBUTO"), ("SUCHITRA ENTERPRISE", r"SUCHITRA"), ("LOKENATH ENTERPRISE", r"LOKENATH"), ("K D ENTERPRISE", r"K\.\s*D\.\s*ENTERPRISE"),
    ("BINOY TRADING", r"BINOY"), ("S S AGENCIES", r"S\.?\s*S\.?\s*AGENCIES"), ("ESTEEM ENTERPRISE", r"ESTEEM"), ("SK MARKETING", r"S\.?\s*K\.?\s*MARKETING"),
    ("SUJATA AGENCY", r"SUJATA"), ("GANAPATI FOOD SUPPLIER", r"GANAPATI"), ("MAA KALI ENTERPRISES", r"MAA\s*KALI"), ("ARTYAR TRADERS", r"ARTYAR|ARTYAN"),
]
BUYER_PAN = "CPQPS9464F"  # buyer = Lakshmi Narayan Bhandar (most frequent PAN in the pool, 130/169 pages)
BUYER_RE = r"LAKSH?MI\s*NARA?Y?AN|LAXMI\s*NARAYAN|LAKSMI"
AMT = r"(?<![\d.])\d{1,3}(?:,\d{2,3})*\.\d{2}(?!\d)|(?<![\d.])\d+\.\d{2}(?!\d)"

def seller_of(T, f, canon, buyer):
    for name, rx in SUPPLIERS:
        # the buyer's own handwritten slips mention the buyer; supplier names appear in printed headers
        if re.search(rx, T): return name, "name"
    s = [canon.get(g[0][2:12], g[0][2:12]) for g in f["gstins"] if canon.get(g[0][2:12], g[0][2:12]) != buyer]
    if s: return "PAN:" + s[0], "pan"
    return "UNKNOWN", "none"

def doc_type(f, T):
    inv = f["has_gst_word"] and (f["has_cgst"] or f["has_igst"] or f["has_hsn"]) and f["has_total"]
    if re.search(r"APPROVAL\s*SLIP|BILL\s*/\s*CHALL?AN|CASH\s*MEMO", T) and not f["has_cgst"]: return "buyer_handwritten_slip"
    if inv and (f["invoice_no_candidate"] or f["dates"] or f["n_gstin"]): return "gst_tax_invoice"
    if inv: return "gst_invoice_fragment_or_continuation"
    return "other_or_non_invoice"

def quality(f): return f["n_words"] >= 100 and f["mean_conf"] >= 0.88

def diversity_score(f):
    return (2 * f["multi_rate"] + 2 * (f["has_igst"] and any(g[0][:2] != "19" for g in f["gstins"])) + 1 * f["has_hsn"] + 1 * f["has_subtotal"] + 1 * bool(f["invoice_no_candidate"]) + 1 * (f["gstin_checksum_ok"] > 0) + 0.5 * f["mean_conf"] + 0.5 * (not f["multi_page"]))

# ---------------- candidates ----------------
def span_words(line, text_offsets, a, b):
    ws = [w for w, (s, e) in zip(line, text_offsets) if e > a and s < b]
    return ws
def bb(ws): return [min(w["bbox"][0] for w in ws), min(w["bbox"][1] for w in ws), max(w["bbox"][2] for w in ws), max(w["bbox"][3] for w in ws)]
def line_text(line):
    t, offs, p = "", [], 0
    for w in line:
        offs.append((p, p + len(w["text"]))); t += w["text"] + " "; p += len(w["text"]) + 1
    return t, offs

def find_after(L, label, value, cap=3, below=True):
    out = []
    for li, line in enumerate(L):
        t, offs = line_text(line)
        for m in re.finditer(label, t, re.I):
            m2 = re.search(value, t[m.end():], re.I)
            if m2:
                a, b = m.end() + m2.start(), m.end() + m2.end(); ws = span_words(line, offs, a, b)
                if ws: out.append({"value": t[a:b].strip(), "bbox": bb(ws), "line_text": t.strip()[:120]})
            elif below and li + 1 < len(L):  # label in a header row, value in the row beneath it, same x-range
                lw = span_words(line, offs, m.start(), m.end())
                if not lw: continue
                x0, x1 = min(w["bbox"][0] for w in lw), max(w["bbox"][2] for w in lw); nt, noffs = line_text(L[li + 1])
                for w, (s, e) in zip(L[li + 1], noffs):
                    cx = (w["bbox"][0] + w["bbox"][2]) / 2
                    if x0 - 20 <= cx <= x1 + 80 and re.fullmatch(value, w["text"].strip(), re.I): out.append({"value": w["text"], "bbox": w["bbox"], "line_text": nt.strip()[:120], "from_row_below": True}); break
            if len(out) >= cap: return out
    return out

def last_amount_on_lines(L, label, cap=4):
    out = []
    for line in L:
        t, offs = line_text(line)
        if re.search(label, t, re.I):
            ms = list(re.finditer(AMT, t))
            if ms:
                m = ms[-1]; ws = span_words(line, offs, m.start(), m.end())
                if ws: out.append({"value": m.group(0), "bbox": bb(ws), "line_text": t.strip()[:120]})
        if len(out) >= cap: break
    return out

def candidates(d):
    o = json.loads((POOL / "ocr" / f"{d}.json").read_text()); L = lines_of(o["words"]); c = {}
    hdr = [l for l in L if sum((w["bbox"][1] + w["bbox"][3]) / 2 for w in l) / len(l) < 0.2 * o["height"]]
    comp = []
    for l in hdr:
        t, offs = line_text(l)
        if re.search(r"(PVT|LTD|LIMITED|TRADERS|ENTERPRISE|AGENC|DISTRIB|STORES?|TRADING|MARKETING|SUPPLIER|FOODS?)", t, re.I) and 6 < len(t) < 90 and not re.search(BUYER_RE, t, re.I): comp.append({"value": t.strip(), "bbox": bb(l)})
    c["merchant_name"] = comp[:3]
    gs = [{"value": g[0], "bbox": g[2], "checksum_ok": g[1]} for g in __import__("v2_pool_analysis").gstins(o["words"])]
    gs.sort(key=lambda x: (x["value"][2:12] == BUYER_PAN or sum(a != b for a, b in zip(x["value"][2:12], BUYER_PAN)) <= 2, not x["checksum_ok"]))  # seller-looking GSTINs first; the shop (buyer) GSTIN is on nearly every page
    c["gstin"] = gs[:4]
    c["invoice_number"] = find_after(L, r"(?:INVOICE|INV|BILL|TAX\s*INV)\s*\.?\s*(?:NO|NUMBER|#)\.?\s*[:.\-]?", r"[A-Z0-9][A-Z0-9/\-]{3,}")
    c["invoice_date"] = find_after(L, r"(?:INVOICE\s*DATE|BILL\s*DATE|DATED?|DT\.?)\s*[:.\-]?", r"\d{1,2}\s*[-/.]\s*(?:\d{1,2}|[A-Za-z]{3})\s*[-/.]\s*\d{2,4}")
    c["subtotal"] = last_amount_on_lines(L, r"AMOUNT\s*BEFORE\s*TAX|TAXABLE\s*(?:VALUE|AMOUNT|AMT)|SUB\s*-?\s*TOTAL|TOTAL\s*TAXABLE|TOTAL\s*BEFORE\s*TAX|GROSS\s*AMOUNT")
    c["cgst"] = last_amount_on_lines(L, r"C\s*-?\s*GST\s*@|OUTPUT\s*C\s*-?\s*GST|TOTAL\s*C\s*-?\s*GST|\bCGST\s*(?:AMT|AMOUNT)?\s*[:=]")
    c["sgst"] = last_amount_on_lines(L, r"(?:S|UT)\s*-?\s*GST\s*@|OUTPUT\s*(?:S|UT)\s*-?\s*GST|TOTAL\s*(?:S|UT)\s*-?\s*GST|\bSGST\s*(?:AMT|AMOUNT)?\s*[:=]")
    c["igst"] = last_amount_on_lines(L, r"I\s*-?\s*GST\s*@|OUTPUT\s*I\s*-?\s*GST|TOTAL\s*I\s*-?\s*GST|\bIGST\s*(?:AMT|AMOUNT)?\s*[:=]")
    hs = []
    for line in L:
        for w in line:
            if re.fullmatch(r"\d{4}(?:\d{2}(?:\d{2})?)?", w["text"].strip()) and 1 <= len(hs) + 1 <= 3 and not any(h["value"] == w["text"] for h in hs): hs.append({"value": w["text"], "bbox": w["bbox"]})
        if len(hs) >= 3: break
    c["hsn_sac"] = hs if any(re.search(r"HSN|SAC", " ".join(w["text"] for w in l), re.I) for l in L) else []
    c["total_amount"] = last_amount_on_lines(L, r"TOTAL\s*BILL\s*AMOUNT|GRAND\s*TOTAL|NET\s*AMOUNT\s*(?:AFTER|PAYABLE)|INVOICE\s*VALUE|AMOUNT\s*PAYABLE|NET\s*PAYABLE|NET\s*AMT|TOTAL\s*AMOUNT|\bTOTAL\b")
    c["payment_method"] = []
    for k, v in c.items():
        for x in v: x["status"] = "SUGGESTION_ONLY"; x["source"] = "ocr_regex_rule"
    return c, o

def main():
    F = json.load(open(POOL / "pool_features.json")); ori = json.load(open(POOL / "orientation.json"))
    cnt = C.Counter(p for f in F for p in set(g[0][2:12] for g in f["gstins"])); buyer = cnt.most_common(1)[0][0]; canon = {}
    for p, _ in cnt.most_common():
        for q in set(canon.values()):
            if sum(a != b for a, b in zip(p, q)) <= 2 and cnt[q] >= cnt[p]: canon[p] = q; break
        else: canon[p] = p
    rows = []; TT = {}
    for f in F:
        o = json.loads((POOL / "ocr" / f"{f['document_id']}.json").read_text()); TT[f["document_id"]] = "\n".join(" ".join(w["text"] for w in l) for l in lines_of(o["words"])).upper()
    p2n = C.defaultdict(C.Counter)  # seller PAN -> supplier name learned from docs whose printed name matched (fixes OCR-garbled headers)
    for f in F:
        n, how = seller_of(TT[f["document_id"]], f, canon, canon[buyer])
        if how == "name":
            for g in f["gstins"]:
                p = canon.get(g[0][2:12], g[0][2:12])
                if p != canon[buyer]: p2n[p][n] += 1
    for f in F:
        d = f["document_id"]; o = json.loads((POOL / "ocr" / f"{d}.json").read_text()); T = TT[d]
        sup, how = seller_of(T, f, canon, canon[buyer])
        if how != "name":
            for g in f["gstins"]:
                p = canon.get(g[0][2:12], g[0][2:12])
                if p in p2n and p != canon[buyer]: sup, how = p2n[p].most_common(1)[0][0], "pan->name"; break
        dt = doc_type(f, T)
        seller_state = next((g[0][:2] for g in f["gstins"] if canon.get(g[0][2:12], g[0][2:12]) != canon[buyer]), None)
        rows.append(dict(document_id=d, supplier=sup, supplier_evidence=how, doc_type=dt, readable=quality(f), n_words=f["n_words"], mean_conf=f["mean_conf"], rotation_ccw=ori[d]["rotation_ccw"],
                         layout_cluster=f["layout_cluster"], multi_page=f["multi_page"], seller_state=seller_state, interstate=bool(seller_state and seller_state != "19"), n_gstin=f["n_gstin"],
                         has={k: bool(f["has_" + k]) for k in ["invoice_no", "date", "subtotal", "cgst", "sgst", "igst", "hsn", "total"]}, has_invoice_no_value=bool(f["invoice_no_candidate"]), has_date_value=bool(f["dates"]),
                         gstin_checksum_ok=f["gstin_checksum_ok"], rates=f["rates"], multi_rate=f["multi_rate"], div=round(diversity_score(f), 2), text_dup_of=f["text_dup_of"]))
    # ---- tiers: per supplier, rank the readable GST invoices; A = top 4 (>=1 per supplier), B = next up to 6 for large suppliers (>=10 docs) else 2; rest C ----
    by = C.defaultdict(list)
    for r in rows: by[r["supplier"]].append(r)
    for r in rows: r["tier"], r["tier_reason"] = "C", ""
    for s, l in by.items():
        ok = [r for r in l if r["doc_type"] == "gst_tax_invoice" and r["readable"] and not r["text_dup_of"]]
        for r in l:
            if r not in ok: r["tier_reason"] = "not a readable GST tax invoice (%s%s)" % (r["doc_type"], "" if r["readable"] else ", low OCR quality") if not r["text_dup_of"] else "near-duplicate of " + r["text_dup_of"]
        if not ok and s != "UNKNOWN":  # only layout of this supplier: allow borderline-readable invoices (>=70 words, conf>=0.83), max 2, tier B
            bl = sorted([r for r in l if r["doc_type"] == "gst_tax_invoice" and r["n_words"] >= 70 and r["mean_conf"] >= 0.83 and not r["text_dup_of"]], key=lambda r: -r["mean_conf"])[:2]
            for r in bl: r["tier"], r["tier_reason"] = "B", "borderline OCR readability; sole example of this supplier/layout"
            continue
        ok.sort(key=lambda r: -r["div"]); nb = 6 if len(l) >= 10 else 2
        for i, r in enumerate(ok):
            if s == "UNKNOWN": r["tier"] = "B" if i < 10 else "C"
            else: r["tier"] = "A" if i < 4 else "B" if i < 4 + nb else "C"
            if r["tier"] == "C": r["tier_reason"] = "supplier/layout already represented (%d better candidates)" % i
    json.dump(rows, open(POOL / "pool_table.json", "w"), indent=1)
    rev = []
    for r in rows:
        if r["tier"] in ("A", "B"):
            c, o = candidates(r["document_id"])
            rev.append({"document_id": r["document_id"], "tier": r["tier"], "supplier_candidate": r["supplier"], "original_image": ori[r["document_id"]]["original"], "working_image": f"interim/v2_pool/images/{r['document_id']}.jpg",
                        "corrected_orientation": {"rotation_ccw_deg": r["rotation_ccw"], "exif_applied": True}, "image_size": o["width"] and [o["width"], o["height"]], "ocr_text": " \n".join(" ".join(w["text"] for w in l) for l in lines_of(o["words"])),
                        "ocr_boxes": [{"text": w["text"], "bbox": w["bbox"], "conf": w["conf"]} for w in o["words"]], "candidates": c, "candidate_status": "SUGGESTION_ONLY"})
    json.dump(rev, open(POOL / "review_candidates.json", "w"), ensure_ascii=False)
    return rows, rev

if __name__ == "__main__":
    rows, rev = main(); print(C.Counter(r["tier"] for r in rows), C.Counter(r["doc_type"] for r in rows), len(rev))
