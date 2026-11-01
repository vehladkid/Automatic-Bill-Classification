"""Annotation-quality audit of the Tier-A pilot annotations (datasets/interim/v2_pool/annot/annotations_tierA.json)."""
import json, collections as C
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import normalize
ROOT = Path(__file__).resolve().parents[2]; ANN = ROOT / "datasets/interim/v2_pool/annot"
A = json.load(open(ANN / "annotations_tierA.json")); POOLT = {r["document_id"]: r for r in json.load(open(ROOT / "datasets/interim/v2_pool/pool_table.json"))}
F = ["merchant_name", "gstin", "invoice_number", "invoice_date", "subtotal", "tax_total", "cgst", "sgst", "igst", "hsn_sac", "total_amount"]
R = {"n_docs": len(A), "status": dict(C.Counter(a["annotation_status"] for a in A)), "fields": {}}
for f in F:
    rs = [a["fields"][f] for a in A]; filled = [r for r in rs if r["value"]]
    sug = [r for r in rs if r["original_suggestion"] is not None]
    R["fields"][f] = {"filled": len(filled), "null": len(rs) - len(filled), "unsupported_multi_value": sum(r["status"] == "unsupported_multi_value" for r in rs),
        "had_suggestion": len(sug), "no_suggestion_but_human_value": sum(1 for r in filled if r["original_suggestion"] is None),
        "suggestion_agreed": sum(1 for r in sug if r["value"] and r["agrees_with_top_suggestion"]), "suggestion_corrected": sum(1 for r in sug if r["value"] and r["suggestion_corrected"]),
        "suggestion_wrong_but_field_null": sum(1 for r in sug if not r["value"]),
        "bbox_corrected": sum(1 for r in filled if r["bbox_corrected"]), "bbox_source": dict(C.Counter(r["bbox_source"] for r in filled)), "no_bbox": sum(1 for r in filled if not r["bbox"])}
# consistency checks
chk = C.Counter(); bad = []
for a in A:
    v = {f: normalize(f, a["fields"][f]["value"]) if a["fields"][f]["value"] else None for f in F}
    ex = a.get("extra", {})
    if v["cgst"] is not None and v["sgst"] is not None:
        chk["cgst==sgst"] += v["cgst"] == v["sgst"]; chk["cgst_sgst_pairs"] += 1
        if v["tax_total"] is not None:
            chk["tax_total==cgst+sgst"] += abs(v["tax_total"] - v["cgst"] - v["sgst"]) < 0.02; chk["tax_pairs"] += 1
            if abs(v["tax_total"] - v["cgst"] - v["sgst"]) >= 0.02: bad.append((a["document_id"], "tax_total != cgst+sgst"))
    if v["subtotal"] is not None and v["tax_total"] is not None and v["total_amount"] is not None:
        d = v["total_amount"] - v["subtotal"] - v["tax_total"]; chk["subtotal+tax~total(|d|<=1)"] += abs(d) <= 1.0; chk["sub_tax_total_triples"] += 1
        if abs(d) > 1.0: bad.append((a["document_id"], f"subtotal+tax-total={d:.2f}"))
    if v["subtotal"] is not None and v["total_amount"] is not None and v["cgst"] is not None and v["sgst"] is not None:
        d = v["total_amount"] - v["subtotal"] - v["cgst"] - v["sgst"]; chk["subtotal+cgst+sgst~total(|d|<=1)"] += abs(d) <= 1.0; chk["sub_cgst_sgst_total"] += 1
        if abs(d) > 1.0: bad.append((a["document_id"], f"subtotal+cgst+sgst-total={d:.2f}"))
    if a["fields"]["gstin"]["value"]: chk["gstin_checksum_ok"] += bool(a["fields"]["gstin"].get("gstin_checksum_ok")); chk["gstin_n"] += 1
R["consistency"] = dict(chk); R["consistency_flags"] = bad
sup = C.Counter(a["supplier"] for a in A); R["supplier_distribution"] = dict(sup.most_common())
R["n_suppliers"] = len(sup); R["top_supplier_share"] = round(sup.most_common(1)[0][1] / len(A), 3)
mr = [a for a in A if len(a["extra"].get("rates_pct", [])) >= 2]; R["multi_rate_docs"] = len(mr)
R["rates_seen"] = sorted({r for a in A for r in a["extra"].get("rates_pct", [])})
R["zero_rate_lines_docs"] = sum(0 in a["extra"].get("rates_pct", []) for a in A)
R["partial_documents"] = [a["document_id"] for a in A if a["extra"].get("partial_document") or "page" in a["extra"] and "last" not in str(a["extra"]["page"])]
R["same_invoice_two_photos"] = sorted(a["document_id"] for a in A if a["extra"].get("same_invoice_two_photos"))
R["hsn_unique_counts"] = dict(C.Counter(min(len({h["value"] for h in a["extra"].get("hsn_sac_list", [])}), 6) for a in A))
R["hsn_scalar_supported_docs"] = sum(bool(a["fields"]["hsn_sac"]["value"]) for a in A)
R["hsn_list_docs"] = sum(len(a["extra"].get("hsn_sac_list", [])) > 0 for a in A)
R["interstate_docs"] = 0; R["tax_notes"] = {"igst_nonnull": sum(bool(a["fields"]["igst"]["value"]) for a in A)}
R["docs_with_subtotal_null_because_of_discount_or_ambiguity"] = sum(1 for a in A if not a["fields"]["subtotal"]["value"] and any(k in a["extra"] for k in ("scheme_discount", "total_discount", "scheme_amount", "gross_amount", "discount", "cash_discount")))
R["unreconciled_total_choice"] = {"printed_net_payable_after_round_off_used": True}
R["orientation_confident_all"] = True
json.dump(R, open(ROOT / "reports/taskB/v2/annotation_audit_tierA.json", "w"), indent=1)
json.dump(A, open(ROOT / "reports/taskB/v2/annotations_tierA.json", "w"), indent=1, ensure_ascii=False)
for k, v in R.items():
    if k != "fields": print(k, v)
for f, d in R["fields"].items(): print(f, d)
