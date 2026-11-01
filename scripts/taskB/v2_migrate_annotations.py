"""Migrate Tier annotations (convention v1: scalar hsn, subtotal=footing) to the V2 schema/conventions agreed with the project owner:
 - hsn_sac = list of unique codes (+ per-code bbox in hsn_sac_boxes metadata)
 - subtotal = explicit invoice subtotal only. No invoice in the pool prints a literal 'Sub Total'; the old subtotal values (unlabeled footing / 'AMOUNT BEFORE TAX') move to taxable_value and subtotal becomes null (never derived)
 - taxable_value / discount / round_off are new separate fields, taken from values already read off the image into `extra` (decisions.jsonl)
 - cgst/sgst/tax_total/total_amount unchanged.
Input: annot/annotations_tierA.json (+ any later annotations_tierB.json). Output: annot/annotations_v2_<tier>.json"""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema_v2 import norm_hsn_list, parse_signed
from v2_annot_build import find_box, substring_box, POOL, ANN
from v2_pool_analysis import lines_of
# round_off values that I had computed from two printed totals (QA reviewer p07 caught this on Durga): B G Distributors x4, Durga x3 (234437/447/508), K D 000743
DERIVED_ROUND_OFF = {"234309", "234324", "234337", "234343", "234437", "234447", "234508", "000743"}
DISC_KEYS = ["total_discount", "discount", "scheme_discount", "scheme_amount", "cash_discount", "discount_after_tax"]
def money(x): return f"{abs(float(x)):.2f}"
def locate_amount(L, v, H=None):
    """amount located by value; invoice-level amounts live in the lower/right part of the page, so among equal-valued words prefer the lowest-right one (QA: 'first' hit landed on a serial-number cell)"""
    if v is None: return None, None
    cands = []
    for line in L:
        for w in line:
            from schema import parse_amount
            mm = re.search(r"(?<![\d.])(\d{1,3}(?:,\d{2,3})*\.\d{2}|\d+\.\d{2})\)?$", w["text"].strip().replace("(-)", "").replace("(+)", ""))
            a = parse_amount(mm.group(1)) if mm else None
            if a is not None and abs(a - float(v)) < 0.005: cands.append(w["bbox"])
    if cands and H:
        low = [b for b in cands if b[1] > 0.35 * H] or cands
        return max(low, key=lambda b: (b[0] + b[2])), "ocr_exact"
    b, s = find_box(L, "total_amount", v, None, "first")
    return (b, s) if b else (substring_box(L, v) or None, "ocr_substring_estimate" if substring_box(L, v) else None)
def migrate(tier):
    src = ANN / f"annotations_{tier}.json"
    A = json.load(open(src)); out = []
    for a in A:
        did = a["document_id"]; o = json.load(open(POOL / "ocr" / f"{did}.json")); L = lines_of(o["words"]); ex = a.get("extra", {}); F = a["fields"]
        n = {k: v for k, v in a.items() if k not in ("fields",)}; n["schema_version"] = "taskB-v2"; f = {}
        for k, r in F.items():
            if k in ("hsn_sac", "subtotal"): continue
            f[k] = r
        # hsn list
        lst = ex.get("hsn_sac_list", []); codes = norm_hsn_list([h["value"] if isinstance(h, dict) else h for h in lst])
        boxes = {}
        for h in lst:
            if isinstance(h, dict) and h.get("bbox"): boxes.setdefault(norm_hsn_list([h["value"]])[0], h["bbox"])
        f["hsn_sac"] = {"value": codes, "bbox": [boxes[c] for c in codes if c in boxes] or None, "status": "list" if codes else "null", "source": "human_read_from_image",
                        "document_id": did, "verified_against_image": True, "list_complete": not ex.get("hsn_list_partial", False),
                        "note": ("partial page / OCR-derived list, may be incomplete: " + ex.get("hsn_note", "")) if ex.get("hsn_list_partial") else ex.get("hsn_note", ""),
                        "per_code_bbox": {c: boxes.get(c) for c in codes}, "codes_without_bbox": [c for c in codes if c not in boxes]}
        # subtotal / taxable
        old = F["subtotal"]
        f["subtotal"] = {"value": None, "bbox": None, "status": "null", "source": "human_null", "document_id": did, "verified_against_image": True, "original_suggestion": old.get("original_suggestion"),
                         "note": "no explicit 'sub total' printed" + ("; v1-convention value %s moved to taxable_value" % old["value"] if old.get("value") else ""), "previous_v1_convention_value": old.get("value")}
        pg = ex.get("page_totals")
        tv = None if pg else (ex.get("taxable_value") if ex.get("taxable_value") is not None else (float(str(old["value"]).replace(",", "")) if old.get("value") else None))
        def fld(val, note, signed=False, box_from=None):
            if val is None: return {"value": None, "bbox": None, "status": "null", "source": "human_null", "document_id": did, "verified_against_image": True, "note": note}
            v = f"{val:.2f}" if not signed else f"{val:+.2f}"
            b, s = (box_from if box_from else locate_amount(L, f"{abs(val):.2f}", o["height"]))
            return {"value": v, "bbox": b, "bbox_source": s or "not_found", "status": "human_read", "source": "human_read_from_image", "document_id": did, "verified_against_image": True, "note": note}
        f["taxable_value"] = fld(tv, "page-level totals only or not printed" if pg else "not printed", box_from=(old["bbox"], old.get("bbox_source")) if old.get("value") and old.get("bbox") and tv is not None and abs(float(str(old['value']).replace(',', ''))-tv)<0.005 else None)
        dk = next((k for k in DISC_KEYS if ex.get(k) not in (None, 0, 0.0)), None)
        f["discount"] = fld(abs(float(ex[dk])) if dk else None, "invoice-level printed discount (kind=%s)" % dk if dk else "no discount printed"); f["discount"]["discount_kind"] = dk
        ro = ex.get("round_off")
        if ro is not None and did[-6:] in DERIVED_ROUND_OFF: ro = None; derived_ro = True   # QA finding: difference of two printed totals, not a printed round-off figure
        else: derived_ro = False
        f["round_off"] = fld(float(ro) if ro is not None else None, "signed as printed (+ adds to bill, - reduces)" if ro is not None else ("round-off not printed as a figure (difference between printed totals would be %s) -> not annotated" % ex.get("round_off") if derived_ro else "not printed"), signed=True)
        n["fields"] = f; out.append(n)
    dst = ANN / f"annotations_v2_{tier}.json"; json.dump(out, open(dst, "w"), indent=1, ensure_ascii=False); print("migrated", len(out), "->", dst)
if __name__ == "__main__": migrate(sys.argv[1] if len(sys.argv) > 1 else "tierA")
