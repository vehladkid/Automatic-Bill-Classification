"""Compare blind-reviewer annotations with the project annotations (annotations_v2_tierA.json) on the QA pages. Disagreements are written to qa/disagreements.json (separate from the annotations)."""
import sys, json, glob, collections as C
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema_v2 import normalize, values_match, FIELDS, norm_hsn_list
ROOT = Path(__file__).resolve().parents[2]; POOL = ROOT / "datasets/interim/v2_pool"; Q = POOL / "qa"
QF = ["merchant_name", "gstin", "invoice_number", "invoice_date", "subtotal", "taxable_value", "discount", "round_off", "tax_total", "cgst", "sgst", "igst", "hsn_sac", "total_amount"]
def nv(f, v):
    if f == "hsn_sac": return norm_hsn_list(v)
    if v in (None, "", []): return None
    return normalize(f, v)
def same(f, a, b):
    if f == "hsn_sac": return set(a) == set(b)
    if a is None or b is None: return a is None and b is None
    if f == "merchant_name":
        import difflib; return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.9
    return values_match(f, a, b)
def main(reviewer_prefix="reviewer*", out="disagreements.json"):
    mp = json.load(open(Q / "page_map_PRIVATE.json")); A = {a["document_id"]: a for a in json.load(open(POOL / "annot/annotations_v2_tierA.json"))}
    rows = []; dis = []
    for pid, m in mp.items():
        fs = glob.glob(str(Q / "blind_review" / f"{reviewer_prefix}_{pid}.json"))
        if not fs: continue
        rv = json.load(open(fs[0]))["fields"]; mine = A[m["document_id"]]["fields"]
        for f in QF:
            a = nv(f, mine[f]["value"]); b = nv(f, rv.get(f, {}).get("value"))
            ok = same(f, a, b); nulla, nullb = (not a), (not b)
            rows.append({"page": pid, "family": m["family"], "field": f, "agree": ok, "null_agree": nulla == nullb, "mine_null": nulla, "rev_null": nullb})
            if not ok: dis.append({"page": pid, "document_id": m["document_id"], "family": m["family"], "field": f, "claude": mine[f]["value"], "reviewer": rv.get(f, {}).get("value"), "reviewer_note": rv.get(f, {}).get("note"), "claude_note": mine[f].get("note")})
    json.dump({"disagreements": dis, "rows": rows}, open(Q / out, "w"), indent=1, ensure_ascii=False)
    n = len(rows); print("pages", len({r["page"] for r in rows}), "field comparisons", n)
    print("value agreement", sum(r["agree"] for r in rows), "/", n, round(sum(r["agree"] for r in rows) / max(n, 1), 3))
    print("null-decision agreement", sum(r["null_agree"] for r in rows), "/", n)
    pf = C.defaultdict(lambda: [0, 0])
    for r in rows: pf[r["field"]][1] += 1; pf[r["field"]][0] += r["agree"]
    for f, (x, y) in pf.items(): print(f"  {f:15s} {x}/{y}")
    pg = C.Counter(d["page"] for d in dis); print("pages with >=1 disagreement", len(pg), dict(pg)); print("field-level disagreements", len(dis))
    return dis
if __name__ == "__main__": main(*sys.argv[1:])
