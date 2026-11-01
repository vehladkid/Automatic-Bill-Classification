"""Builds the Tier-A annotation file from the annotator's decisions (datasets/interim/v2_pool/annot/decisions.jsonl).
A decision = the value the annotator READ OFF THE IMAGE (+ optional line-label hint / manual bbox / note). This script only (a) finds the matching OCR word box so the value gets a bbox,
(b) compares with the original OCR/regex suggestion (SUGGESTION_ONLY) to record agreement/correction, (c) emits the export format of the review tool. Task B model predictions are never used."""
import sys, json, re, difflib, collections as C
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import normalize, parse_amount, parse_date, norm_id, gstin_checksum_ok, FIELDS
from v2_pool_analysis import lines_of
ROOT = Path(__file__).resolve().parents[2]; POOL = ROOT / "datasets/interim/v2_pool"; ANN = POOL / "annot"
CAN = ["merchant_name", "gstin", "invoice_number", "invoice_date", "subtotal", "tax_total", "cgst", "sgst", "igst", "hsn_sac", "total_amount"]  # annotated in this pilot (payment_method/address/currency: not annotated)

def bb(ws): return [min(w["bbox"][0] for w in ws), min(w["bbox"][1] for w in ws), max(w["bbox"][2] for w in ws), max(w["bbox"][3] for w in ws)]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i; return i / u if u else 0

def substring_box(L, text):
    """value printed inside a longer OCR word (OCR glued neighbours): estimate the sub-box by character proportion (flagged 'ocr_substring_estimate')."""
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    for line in L:
        for w in line:
            wt = re.sub(r"[^A-Z0-9]", "", w["text"].upper())
            i = wt.find(t) if len(t) >= 4 else -1
            if i >= 0 and len(wt) >= len(t):
                x0, x1 = w["bbox"][0], w["bbox"][2]; n = len(wt)
                return [int(x0 + (x1 - x0) * i / n), w["bbox"][1], int(x0 + (x1 - x0) * (i + len(t)) / n), w["bbox"][3]]
    return None

def find_box(L, field, value, hint=None, occ="last"):
    typ = FIELDS[field]; cands = []
    target = normalize(field, value) if typ != "text" else re.sub(r"[^A-Z0-9]", "", str(value).upper())
    for line in L:
        lt = " ".join(w["text"] for w in line)
        if hint and not re.search(hint, lt, re.I): continue
        n = len(line)
        for i in range(n):
            for j in range(i, min(n, i + (9 if typ == "text" else 4))):
                ws = line[i:j + 1]; raw = " ".join(w["text"] for w in ws)
                if typ == "amount":
                    if j > i: break
                    v = parse_amount(raw)
                    if v is not None and target is not None and abs(v - target) < 0.005: cands.append((1.0, ws))
                elif typ == "date":
                    try: v = parse_date(raw)
                    except Exception: v = None
                    if v and v == target: cands.append((1.0, ws))
                elif typ == "text":
                    r = difflib.SequenceMatcher(None, re.sub(r"[^A-Z0-9]", "", raw.upper()), target).ratio()
                    if r >= 0.8: cands.append((r, ws))
                else:  # id / gstin / code
                    t = re.sub(r"[^A-Z0-9]", "", raw.upper())
                    if not t: continue
                    if target and (target == t or (len(target) >= 6 and target in t and len(t) <= len(target) + 12)): cands.append((1.0 if target == t else 0.95, ws))
                    elif target and len(t) >= 5 and abs(len(t) - len(target)) <= 2 and difflib.SequenceMatcher(None, t, target).ratio() >= 0.86: cands.append((0.7, ws))
    if not cands: return None, None
    best = max(c[0] for c in cands); top = [c for c in cands if c[0] == best]
    if typ == "text" and len(top) > 1: top.sort(key=lambda c: -len(c[1]))
    pick = top[-1] if occ == "last" else top[0] if occ == "first" else top[min(int(occ), len(top) - 1)]
    return bb(pick[1]), ("ocr_exact" if best >= 0.95 else "ocr_fuzzy")

def main():
    dec = [json.loads(l) for l in open(ANN / "decisions.jsonl") if l.strip()]
    rev = {r["document_id"]: r for r in json.load(open(POOL / "review_candidates.json"))}
    ori = json.load(open(POOL / "orientation.json")); out = []
    for d in dec:
        did = d["document_id"]; o = json.load(open(POOL / "ocr" / f"{did}.json")); L = lines_of(o["words"]); rv = rev[did]
        rec = {"document_id": did, "tier": "A", "supplier": d.get("supplier") or rv["supplier_candidate"], "working_image": rv["working_image"], "original_image": rv["original_image"],
               "annotation_status": "unusable" if d.get("unusable") else "verified", "annotator": "claude (AI-assisted; image crops read by the model, not an independent human)",
               "verified_against_image": not d.get("unusable"), "notes": d.get("notes", ""), "fields": {}, "extra": d.get("extra", {})}
        for f in CAN:
            spec = d.get("fields", {}).get(f); sug = rv["candidates"].get(f, []); top = sug[0] if sug else None
            rcd = {"value": None, "bbox": None, "status": "null", "source": "human_null", "document_id": did, "original_suggestion": top["value"] if top else None, "original_suggestion_bbox": top["bbox"] if top else None,
                   "final_value": None, "verified_against_image": True, "agrees_with_top_suggestion": None, "suggestion_corrected": False, "bbox_corrected": False, "bbox_source": None, "note": ""}
            if spec is None and f == "hsn_sac" and d.get("hsn_sac_list"):
                rcd["note"] = "unsupported: invoice has %d unique HSN/SAC codes; canonical hsn_sac is scalar -> null (full list in extra.hsn_sac_list)" % len({h['value'] if isinstance(h, dict) else h for h in d["hsn_sac_list"]}); rcd["status"] = "unsupported_multi_value"
            if spec is not None and spec.get("value") is not None:
                v = spec["value"]; box = spec.get("bbox"); bs = "manual" if box else None
                if not box and spec.get("cand_box") and top: box, bs = top["bbox"], "suggestion_bbox"
                if not box and spec.get("locate"):
                    box = substring_box(L, spec["locate"]); bs = "ocr_substring_estimate" if box else None
                if not box: box, bs = find_box(L, f, v, spec.get("hint"), spec.get("occ", "last"))
                if not box and spec.get("hint"): box, bs = find_box(L, f, v, None, spec.get("occ", "last"))
                if not box: box = substring_box(L, str(v)); bs = "ocr_substring_estimate" if box else None
                if not box and f == "gstin":  # OCR misread a few characters: align on the readable 8-char prefix, flag as partial
                    box = substring_box(L, str(v)[:8]) or substring_box(L, str(v)[-8:]); bs = "ocr_partial_match" if box else None
                nv = normalize(f, v); sv = normalize(f, top["value"]) if top else None
                rcd.update(value=str(v), final_value=str(v), bbox=box, bbox_source=bs or "not_found", note=spec.get("note", ""))
                if f == "gstin": rcd["gstin_checksum_ok"] = gstin_checksum_ok(v)
                same_any = any(normalize(f, c["value"]) == nv for c in sug)
                rcd["agrees_with_top_suggestion"] = (sv == nv) if top else None
                rcd["suggestion_corrected"] = bool(top) and sv != nv
                rcd["accepted_from_alternative_suggestion"] = bool(top) and sv != nv and same_any
                if box and top and sv == nv: rcd["bbox_corrected"] = iou(box, top["bbox"]) < 0.5
                elif box and bs == "manual": rcd["bbox_corrected"] = True
                rcd["status"] = ("accepted" if (top and sv == nv and not rcd["bbox_corrected"]) else "edited"); rcd["source"] = "suggestion_accepted" if rcd["status"] == "accepted" else "human_edit"
                if not box: rcd["note"] += " [no bbox: value not located in OCR words]"
            elif spec is not None: rcd["note"] = spec.get("note", "")  # explicit null with reason
            rec["fields"][f] = rcd
        if d.get("hsn_sac_list"):
            lst = []; used = set()
            for h in d["hsn_sac_list"]:
                hv = h["value"] if isinstance(h, dict) else h; box, bs = (h.get("bbox"), "manual") if isinstance(h, dict) and h.get("bbox") else find_box(L, "hsn_sac", hv, None, "first")
                if box and tuple(box) in used and bs != "manual": box, bs = None, "not_located_separately"
                if box: used.add(tuple(box))
                lst.append({"value": hv, "bbox": box, "bbox_source": bs})
            rec["extra"]["hsn_sac_list"] = lst
        c, g = rec["fields"]["cgst"], rec["fields"]["sgst"]
        if c["bbox"] and c["bbox"] == g["bbox"]:  # OCR glued the two numbers into one word or only one occurrence found: shared box is ambiguous, say so
            g["bbox"] = None; g["bbox_source"] = "glued_with_cgst_not_boxed"; c["bbox_source"] = (c["bbox_source"] or "") + "+glued_with_sgst"; g["bbox_ambiguous"] = c["bbox_ambiguous"] = True  # QA: shared box is wrong for sgst
        out.append(rec)
    json.dump(out, open(ANN / "annotations_tierA.json", "w"), indent=1, ensure_ascii=False); print("built", len(out), "docs")
if __name__ == "__main__": main()
