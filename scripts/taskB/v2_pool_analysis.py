"""V2 step 2: diversity / field-availability analysis of the oriented-OCR in_grocery_gst pool + value tiers + SUGGESTION_ONLY review file.
Uses only OCR text and deterministic regex rules - no Task B model output. Writes reports/taskB/v2/{pool_analysis.json, annotation_pool_analysis.md}
and datasets/interim/v2_pool/{pool_features.json, review_candidates.json}."""
import sys, json, re, collections as C
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import repair_gstin, gstin_checksum_ok, parse_amount
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "datasets" / "interim" / "v2_pool"
REP = ROOT / "reports" / "taskB" / "v2"

def lines_of(words):
    ws = sorted(words, key=lambda w: ((w["bbox"][1] + w["bbox"][3]) / 2, w["bbox"][0]))
    out, cur, cy = [], [], None
    for w in ws:
        y = (w["bbox"][1] + w["bbox"][3]) / 2; h = max(w["bbox"][3] - w["bbox"][1], 1)
        if cur and abs(y - cy) > 0.6 * h: out.append(cur); cur = []
        cur.append(w); cy = y if len(cur) == 1 else (cy * (len(cur) - 1) + y) / len(cur)
    if cur: out.append(cur)
    return [sorted(l, key=lambda w: w["bbox"][0]) for l in out]

def gstins(words):
    """15-char windows over each OCR word / line (alnum-only); position-aware repair; returns [(gstin, checksum_ok, bbox, raw)] in reading order."""
    res = []
    for l in lines_of(words):
        for w in l:  # per word and per word-pair (OCR often splits 'GSTIN NO : xxxx')
            t = re.sub(r"[^A-Z0-9]", "", w["text"].upper())
            for i in range(0, max(len(t) - 14, 0)):
                g = repair_gstin(t[i:i + 15])
                if g: res.append((g, gstin_checksum_ok(g), w["bbox"], w["text"])); break
    seen, out = set(), []
    for g in res:
        if g[0] not in seen: seen.add(g[0]); out.append(g)
    return out

KW = {
    "invoice_no": r"(INVOICE|INV|BILL|TAX INVOICE)\s*(NO|NUMBER|#|NUM)\b",
    "date": r"\b(INVOICE\s*DATE|BILL\s*DATE|DATE|DT)\b",
    "subtotal": r"(AMOUNT\s*BEFORE\s*TAX|TAXABLE\s*(VALUE|AMOUNT|AMT)|SUB\s*-?\s*TOTAL|TOTAL\s*TAXABLE|GROSS\s*AMOUNT|TOTAL\s*BEFORE\s*TAX|BASIC\s*VALUE)",
    "cgst": r"C\s*-?\s*GST", "sgst": r"(S|UT)\s*-?\s*GST", "igst": r"I\s*-?\s*GST",
    "hsn": r"(HSN|SAC)", "total": r"(GRAND\s*TOTAL|TOTAL\s*BILL\s*AMOUNT|NET\s*(PAYABLE|AMOUNT|AMT)|INVOICE\s*(VALUE|TOTAL)|TOTAL\s*AMOUNT|BILL\s*AMOUNT|AMOUNT\s*PAYABLE|\bTOTAL\b)",
    "gst_word": r"(GST|TAX\s*INVOICE)", "round": r"ROUND",
}
RATES = {0.25, 1.5, 2.5, 3, 5, 6, 9, 12, 14, 18, 28}
COMPANY = r"(PVT|PRIVATE|LTD|LIMITED|TRADERS|TRADING|ENTERPRISE|AGENC|DISTRIB|STORES?|CORPORATION|FOODS|MART|& SONS|AND SONS|COMPANY|INDUSTRIES|MARKETING|SUPPLIERS?)"

def feats(d):
    o = json.loads((POOL / "ocr" / f"{d}.json").read_text()); ws = o["words"]; H = o["height"]
    L = lines_of(ws); txt_lines = [" ".join(w["text"] for w in l) for l in L]; T = "\n".join(txt_lines).upper()
    f = {"document_id": d, "n_words": len(ws), "mean_conf": round(float(np.mean([w["conf"] for w in ws])), 3) if ws else 0.0, "size": [o["width"], o["height"]]}
    G = gstins(ws); f["gstins"] = [(g[0], g[1], g[2]) for g in G]; f["n_gstin"] = len(G)
    f["gstin_checksum_ok"] = sum(g[1] for g in G)
    for k, p in KW.items(): f["has_" + k] = bool(re.search(p, T))
    f["has_hsn_digits"] = bool(re.search(r"\b\d{4,8}\b", T)) and f["has_hsn"]
    rs = set()
    for m in re.finditer(r"(\d{1,2}(?:\.\d{1,2})?)\s*%", T):
        try:
            v = float(m.group(1))
            if v in RATES: rs.add(v)
        except ValueError: pass
    f["rates"] = sorted(rs); f["multi_rate"] = len(rs) >= 2
    f["dates"] = sorted(set(re.findall(r"\b\d{1,2}[-/.]\w{2,3}[-/.]\d{2,4}\b", T)))[:6]
    f["multi_page"] = bool(re.search(r"PAGE\s*\d+\s*(OF|/)\s*\d+", T))
    top = [t for t, l in zip(txt_lines, L) if (np.mean([(w["bbox"][1] + w["bbox"][3]) / 2 for w in l]) < 0.28 * H)]
    comp = [t for t in top if re.search(COMPANY, t.upper()) and len(t) > 6]
    f["company_line"] = comp[0] if comp else None
    f["header_text"] = " | ".join(top)[:300]
    inv = None
    for l in txt_lines:
        m = re.search(r"(?:INVOICE|INV|BILL)\s*(?:NO|NUMBER|#)\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9/\-]{3,})", l.upper())
        if m and any(c.isdigit() for c in m.group(1)): inv = (m.group(1), l); break
    f["invoice_no_candidate"] = inv
    f["tokens"] = sorted(set(re.findall(r"[A-Z]{3,}", T)))
    return f

def supplier_key(f):
    """Seller evidence: first checksum-valid (else first) GSTIN in reading order -> its 7-char PAN+state-free core (PAN) identifies the legal entity."""
    for g, ok, _ in f["gstins"]:
        if ok: return g[2:12]  # PAN
    if f["gstins"]: return f["gstins"][0][0][2:12]
    return None

def main():
    ori = json.loads((POOL / "orientation.json").read_text())
    F = [feats(d) for d in sorted(ori)]
    for f in F: f.update(orientation=ori[f["document_id"]]["rotation_ccw"], margin=ori[f["document_id"]]["orientation_margin"])
    # ---- layout clusters: TF-IDF over form tokens (df>=3, <=60%) + agglomerative cosine ----
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import AgglomerativeClustering
    docs = [" ".join(f["tokens"]) for f in F]
    tv = TfidfVectorizer(min_df=3, max_df=0.6, token_pattern=r"[A-Z]{3,}", lowercase=False)
    X = tv.fit_transform(docs).toarray()
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=0.9, metric="cosine", linkage="average").fit_predict(X)
    for f, l in zip(F, lab): f["layout_cluster"] = int(l)
    # near-duplicate (text) inside the pool
    S = (X @ X.T) / (np.linalg.norm(X, axis=1)[:, None] * np.linalg.norm(X, axis=1)[None] + 1e-9)
    for i, f in enumerate(F):
        f["text_dup_of"] = None
        for j in range(i):
            if S[i, j] > 0.97 and F[j]["n_words"] > 40 and len(set(f["tokens"]) ^ set(F[j]["tokens"])) < 6 and f["invoice_no_candidate"] and F[j]["invoice_no_candidate"] and f["invoice_no_candidate"][0] == F[j]["invoice_no_candidate"][0]:
                f["text_dup_of"] = F[j]["document_id"]; break
    for f in F: f["supplier_pan"] = supplier_key(f)
    json.dump(F, open(POOL / "pool_features.json", "w"), ensure_ascii=False)
    return F

if __name__ == "__main__":
    main()
