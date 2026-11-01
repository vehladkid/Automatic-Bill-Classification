"""Read-only inventory of real documents already on disk: which files are in manifest v2.1, which are not and why (exact duplicate of a manifested file / undecodable / not selected into any manifest), and what is needed to make each excluded group usable.
Never reads datasets/interim/v2_pool or any Task B V2 annotation asset (only counted by name). Writes reports/doc_type/licensing_and_collection/local_inventory.{json,md}. No network, no file changes outside that directory."""
import sys, json, collections, hashlib
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import ROOT, DS, META, read_jsonl, sha256_file
OUT = ROOT / "reports/doc_type/licensing_and_collection"; IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}
ROOTS = ["public/sroie", "public/receiptbench", "interim/my_receipts/images", "public/indian_grocery_gst", "public/funsd", "public/rvl_cdip_160", "public/invoices_receipts_pt", "interim/hf_medical_bills/images", "public/in_financial_docs", "public/cord", "real", "indian", "indian_all/real"]
def main():
    rows = read_jsonl(META / "doc_type_manifest_v2_1.jsonl"); known = {p for r in rows for p in r["pages"]}; sha_to_doc = {s: r["doc_id"] for r in rows for s in r["page_sha256"]}; res = {}
    for root in ROOTS:
        base = DS / root
        if not base.exists(): continue
        files = [f for f in base.rglob("*") if f.is_file() and f.suffix.lower() in IMG]; c = collections.Counter(); ex = []
        for f in files:
            rel = str(f.relative_to(DS)); sha = sha256_file(f)
            if rel in known: c["in_manifest"] += 1; continue
            if sha in sha_to_doc: c["exact_duplicate_of_manifested_file"] += 1; continue
            try:
                if f.suffix.lower() != ".pdf": Image.open(f).verify()
            except Exception: c["undecodable"] += 1; continue
            c["unmanifested_unique_decodable"] += 1; ex.append(rel) if len(ex) < 3 else None
        res[root] = dict(image_or_pdf_files=len(files), **c, examples_unmanifested=ex)
    real = [r for r in rows if not r["synthetic"]]; by = collections.defaultdict(collections.Counter)
    for r in real: by[r["source"]]["rows"] += 1; by[r["source"]]["approved"] += r["approved_for_training"]; by[r["source"]]["flagged_for_review"] += r["review_status"] == "flagged_for_review"; by[r["source"]]["needs_mapping_review"] += r["review_status"] == "needs_mapping_review"; by[r["source"]]["no_ocr"] += not r["ocr_available"]
    groups = [
      dict(group="real docs excluded only by licence/provenance", docs={s: by[s]["rows"] for s in ("in_financial", "sroie", "hf_medical", "user_real", "receiptbench") if s in by}, action="none possible locally; each needs the exact permission in licensing_decisions.md (draft requests in PERMISSION_REQUESTS.md). Do not use until approved.", recoverable_now=False),
      dict(group="approved real docs whose label is only a collection-rule mapping", docs=sum(by[s]["needs_mapping_review"] for s in by if s not in ("in_financial", "sroie", "hf_medical", "user_real", "receiptbench")), action="human review of mapping_review_queue_v2_1.jsonl (286 records: 78 cue conflicts + spot checks)", recoverable_now=False),
      dict(group="real docs flagged for review (OCR cue conflicts)", docs=sum(by[s]["flagged_for_review"] for s in by), action="human decision per record; keep out of evaluation until resolved", recoverable_now=False),
      dict(group="real docs without OCR text", docs=sum(by[s]["no_ocr"] for s in by), action="in_financial only (blocked source); no OCR spent on unapproved data", recoverable_now=False)]
    unman = {k: v["unmanifested_unique_decodable"] for k, v in res.items() if v.get("unmanifested_unique_decodable")}
    out = dict(disk_roots=res, real_by_source={k: dict(v) for k, v in by.items()}, groups=groups, unmanifested_unique_decodable_by_root=unman, v2_pool_note="datasets/interim/v2_pool belongs to the blocked Task B V2 annotation track: not inspected", raw_manifest_dedup=json.loads((META / "dedup_stats.json").read_text()))
    (OUT / "local_inventory.json").write_text(json.dumps(out, indent=1))
    L = ["# Local data inventory (read-only)\n", "| disk root | files | in manifest | exact duplicate of a manifested file | undecodable | unmanifested unique |", "|---|---|---|---|---|---|"]
    for k, v in res.items(): L.append(f"| {k} | {v['image_or_pdf_files']} | {v.get('in_manifest', 0)} | {v.get('exact_duplicate_of_manifested_file', 0)} | {v.get('undecodable', 0)} | {v.get('unmanifested_unique_decodable', 0)} |")
    L += ["", "## Real documents by source (manifest v2.1)", "| source | rows | approved | flagged | needs mapping review | no OCR |", "|---|---|---|---|---|---|"] + [f"| {k} | {v['rows']} | {v['approved']} | {v['flagged_for_review']} | {v['needs_mapping_review']} | {v['no_ocr']} |" for k, v in by.items()]
    L += ["", "## Groups and exact action", "| group | docs | action |", "|---|---|---|"] + [f"| {g['group']} | {g['docs']} | {g['action']} |" for g in groups]
    (OUT / "local_inventory.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
if __name__ == "__main__": main()
