"""Builds the blind packet for the independent HUMAN QA (10 pages) and scores it when returned.
 build : writes qa/HUMAN_QA/{pages/pXX/*, answer_sheet.csv, INSTRUCTIONS.md}; no project annotation values are included.
 score : python v2_qa_human_packet.py score qa/HUMAN_QA/answer_sheet_filled.csv"""
import sys, csv, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]; Q = ROOT / "datasets/interim/v2_pool/qa"; H = Q / "HUMAN_QA"
FIELDS = ["merchant_name", "gstin", "invoice_number", "invoice_date", "subtotal", "taxable_value", "discount", "round_off", "tax_total", "cgst", "sgst", "igst", "hsn_sac", "total_amount"]
INS = """# Independent QA of 10 invoice pages
You are checking invoice photos. Do not look at any other project file. For each page folder (pages/pXX): overview.jpg, top/middle/bottom crops, full_res.jpg (zoom in as needed), and overlay_*.jpg (same page with coloured boxes drawn by the annotator, labelled with field names).

Fill answer_sheet.csv, one row per page x field:
- value: what is printed on the invoice for that field (seller GSTIN only; dates ISO YYYY-MM-DD; numbers with 2 decimals; hsn_sac = all unique codes separated by space). Leave EMPTY if the field is not printed / not legible on this page, and write why in note.
- bbox_ok: for fields that have a coloured box in the overlay: Y (tight on the right value), L (right value but much too big/cut), N (wrong place). Leave empty if no box drawn.
Rules: subtotal = only an explicitly labelled 'sub total' (never taxable value / gross / amount before tax). taxable_value = invoice-level total taxable amount. discount = invoice-level total discount (non-zero). round_off = a printed round-off figure, signed (not a difference you compute). tax_total = printed total tax only. cgst/sgst = invoice-level totals (never one line's tax; empty if several rates and no printed total). total_amount = final payable. Never derive a value; empty beats a guess.
Return the filled CSV; scoring is automatic.
"""
def build():
    mp = json.load(open(Q / "page_map_PRIVATE.json")); (H / "pages").mkdir(parents=True, exist_ok=True)
    for pid in mp:
        d = H / "pages" / pid; d.mkdir(exist_ok=True)
        for f in (Q / "blind_pages" / pid).glob("*.jpg"): shutil.copy(f, d / f.name)
        for f in (Q / "bbox_pages" / pid).glob("overlay_*.jpg"): shutil.copy(f, d / f.name)
    with open(H / "answer_sheet.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["page", "field", "value", "bbox_ok", "note"])
        for pid in mp:
            for fl in FIELDS: w.writerow([pid, fl, "", "", ""])
    (H / "INSTRUCTIONS.md").write_text(INS); print("packet ->", H)
def score(path):
    import v2_qa_compare as qc
    from schema_v2 import values_match
    mp = json.load(open(Q / "page_map_PRIVATE.json")); A = {a["document_id"]: a for a in json.load(open(ROOT / "datasets/interim/v2_pool/annot/annotations_v2_tierA.json"))}
    R = {(r["page"], r["field"]): r for r in csv.DictReader(open(path))}
    rows = []; dis = []; bb = {"Y": 0, "L": 0, "N": 0}
    for pid, m in mp.items():
        for fl in FIELDS:
            r = R.get((pid, fl), {}); mine = A[m["document_id"]]["fields"][fl]
            a = qc.nv(fl, mine["value"]); b = qc.nv(fl, (r.get("value") or "").replace(" ", ", ") if fl == "hsn_sac" else (r.get("value") or None))
            ok = qc.same(fl, a, b); rows.append((ok, (not a) == (not b)))
            if not ok: dis.append((pid, fl, mine["value"], r.get("value")))
            if r.get("bbox_ok") in bb: bb[r["bbox_ok"]] += 1
    n = len(rows); print("value agreement", sum(x[0] for x in rows), "/", n, "| null-decision agreement", sum(x[1] for x in rows), "/", n, "| bbox verdicts", bb)
    print("page-level disagreements", len({d[0] for d in dis}), "field-level", len(dis)); [print(" ", d) for d in dis]
if __name__ == "__main__":
    build() if len(sys.argv) < 2 or sys.argv[1] == "build" else score(sys.argv[2])
