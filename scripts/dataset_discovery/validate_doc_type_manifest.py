"""Validate the unified document-type manifest (schema, labels, licences, files, checksums, separation, grouping, generator consistency). Exit code 1 when any ERROR is found.
  Usage: validate_doc_type_manifest.py [--manifest datasets/metadata/doc_type_manifest_v2.jsonl] [--schema v2] [--full-hash]
Checks: required fields; doc_type/subtype/split/enum validity; eval_eligible == not synthetic; eligibility never 'eligible'; licence-register entry exists and is not blocked/not_acquired; every page file exists and decodes
(images fully, PDFs by header+size); sha256 recomputed for ALL external/generated files and a seeded 300-file sample of legacy files (or all with --full-hash); real/synthetic never mixed in the per-kind manifests; split files agree
with the unified manifest; group_id never spans splits; byte-duplicates across splits; generated documents' structured records are internally consistent (totals, sums, formulas); generated issuer names contain no deny-listed real brand."""
import sys, json, argparse, random, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import ROOT, DS, META, sha256_file, load_schema, family_of, read_jsonl
from PIL import Image
def close(a, b, tol=0.02): return abs(float(a) - float(b)) <= tol + 1e-9 * abs(float(b))
def consistency(a):
    """returns list of failed-check names for one generated document's structured record"""
    t, f = a["doc_type"], a["fields"]; bad = []
    if t == "rent_receipt":
        if not close(f["rent_amount"], f["monthly_rent"] * f["months"]): bad.append("rent=monthly*months")
    elif t == "school_fee_receipt":
        if not close(sum(h["amount"] for h in f["fee_heads"]), f["total"]): bad.append("total=sum(heads)")
        if not close(sum(h["amount"] for h in f["fee_heads"] if h["head"] == "Tuition Fee"), f["tuition_component"]): bad.append("tuition_component")
    elif t == "insurance_premium_receipt":
        if not close(f["premium"] + f["gst"], f["total_paid"]): bad.append("total=premium+gst")
    elif t == "salary_slip":
        e = f["earnings"]; d = f["deductions"]
        if not close(sum(e.values()), f["gross"]): bad.append("gross=sum(earnings)")
        if not close(sum(d.values()), f["total_deductions"]): bad.append("deductions")
        if not close(f["gross"] - f["total_deductions"], f["net_pay"]): bad.append("net=gross-deductions")
    elif t == "form_16":
        if not close(max(0, f["gross_salary"] - f["hra_exempt"] - f["standard_deduction"] - f["deduction_80c"] - f["deduction_80d"]), f["taxable_income"]): bad.append("taxable_income")
        if abs(sum(f["tds_quarters"]) - f["tax_payable"]) > 1: bad.append("quarterly_tds_sum")
    elif t == "meal_voucher_statement":
        if f["subtype"] == "statement" and not close(f["coupons"] * f["coupon_value"], f["total"]): bad.append("total=coupons*value")
        if f["subtype"] == "invoice" and not close(f["meal_card_load"] + f["service_fee"] + f["gst"], f["total"]): bad.append("invoice_total")
    elif t == "ev_document":
        if f["subtype"] == "purchase_invoice" and not close(f["ex_showroom"] + f["gst"] + f["registration"] + f["insurance"] - f["subsidy"], f["total"]): bad.append("purchase_total")
        if f["subtype"] == "charging_receipt" and not (close(f["energy_kwh"] * f["rate_per_kwh"], f["energy_charge"], 0.05) and close(f["energy_charge"] + f["gst"], f["total"], 0.05)): bad.append("charging_total")
    elif t == "loan_interest_certificate":
        if f["principal_repaid"] < 0 or f["interest_paid"] < 0 or f["outstanding_principal"] < 0: bad.append("negative_amounts")
    elif t == "loan_statement":
        if not close(f["total_principal"] + f["total_interest"], f["emi"] * f["rows"], 1.0 + 0.001 * f["emi"] * f["rows"]): bad.append("emi_split")
    elif t == "investment_certificate":
        if f["subtype"] == "ppf" and not close(sum(d["amount"] for d in f["deposits"]), f["total_deposit"]): bad.append("ppf_total")
        if f["subtype"] == "elss" and not close(f["amount_invested"], f["nav"] * f["units"], 0.01 * f["amount_invested"] + 1): bad.append("elss_units")
    return bad
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default=str(META / "doc_type_manifest_v2.jsonl")); ap.add_argument("--schema", default="v2"); ap.add_argument("--tag", default="v2", help="file-name tag of the per-kind manifests and split directory (v2 or v2_1)"); ap.add_argument("--full-hash", action="store_true"); ap.add_argument("--skip-files", action="store_true", help="structural checks only (no file opening/hashing) - used by unit tests"); ap.add_argument("--report", default=str(ROOT / "reports/doc_type/validation_v2.json")); a = ap.parse_args()
    S = load_schema(a.schema); types = set(family_of(S)); fam = family_of(S); subs = S.get("subtypes", {}); req = S["record_fields"]; tax_labels = set(S["axes"]["C_tax_evidence"]["labels"]); rows = read_jsonl(a.manifest)
    reg = {s["id"]: s for s in json.loads((META / "license_register_v2.json").read_text())["sources"]}; err = collections.defaultdict(list); warn = collections.defaultdict(list)
    def E(k, d): err[k].append(d)
    for r in rows:
        did = r["doc_id"]; miss = [k for k in req if k not in r]
        if miss: E("missing_fields", f"{did}: {miss[:4]}"); continue
        if r["doc_type"] not in types: E("unknown_doc_type", f"{did}: {r['doc_type']}")
        elif r["family"] != fam[r["doc_type"]]: E("family_mismatch", did)
        if r["subtype"] is not None and r["subtype"] not in subs.get(r["doc_type"], []): E("unknown_subtype", f"{did}: {r['doc_type']}/{r['subtype']}")
        if r["split"] not in ("train", "val", "test"): E("bad_split", did)
        if r["eval_eligible"] != (not r["synthetic"]): E("eval_eligible_vs_synthetic", did)
        if r["eligibility_status"] not in ("pending_verification", "not_applicable"): E("eligibility_status_invalid", f"{did}: {r['eligibility_status']}")
        if not set(r["tax_evidence_candidates"]) <= tax_labels: E("tax_label_invalid", did)
        if r["review_status"] not in S["review_status_values"]: E("review_status_invalid", did)
        rid = reg.get(r["license_register_id"])
        if rid is None: E("license_register_missing", f"{did}: {r['license_register_id']}")
        elif rid["status"] in ("blocked", "not_acquired"): E("blocked_source_in_manifest", f"{did}: {r['license_register_id']}")
        if r.get("approved_for_training") and rid is not None and rid["status"] in ("conflict", "unverified", "blocked", "not_acquired"): E("unresolved_licence_in_approved_training", f"{did}: {r['license_register_id']}={rid['status']}")
        if r["synthetic"] and r["split_basis"] is None: E("synthetic_without_split_basis", did)
        if r["synthetic"] and r["source"] in ("synthgen_v1",) and not (r["generator_version"] and r["seed"] is not None): E("synthetic_without_generator_version_seed", did)
    # files + checksums
    rnd = random.Random(0); legacy = [r for r in rows if r["source"] not in ("synthgen_v1", "agami", "fatura")]; sample = set(x["doc_id"] for x in (legacy if a.full_hash else rnd.sample(legacy, min(300, len(legacy)))))
    n_img = n_pdf = n_hash = 0
    for r in ([] if a.skip_files else rows):
        for pi, pg in enumerate(r["pages"]):
            p = DS / pg
            if not p.exists(): E("missing_file", f"{r['doc_id']}: {pg}"); continue
            if p.suffix.lower() == ".pdf":
                n_pdf += 1
                if p.stat().st_size < 200 or p.read_bytes()[:5] != b"%PDF-": E("corrupt_pdf", pg)
            else:
                n_img += 1
                try:
                    with Image.open(p) as im: im.verify()
                except Exception as e: E("corrupt_image", f"{pg}: {type(e).__name__}")
            if r["source"] in ("synthgen_v1", "agami", "fatura") or r["doc_id"] in sample:
                n_hash += 1
                if sha256_file(p) != r["page_sha256"][pi]: E("checksum_mismatch", pg)
    # separation + split files
    real, syn = read_jsonl(META / f"doc_type_{a.tag}_real.jsonl"), read_jsonl(META / f"doc_type_{a.tag}_synthetic.jsonl")
    if any(r["synthetic"] for r in real): E("synthetic_in_real_manifest", "")
    if any(not r["synthetic"] for r in syn): E("real_in_synthetic_manifest", "")
    if len(real) + len(syn) != len(rows): E("per_kind_manifests_do_not_sum", f"{len(real)}+{len(syn)}!={len(rows)}")
    for kind in ("real", "synthetic"):
        for sp in ("train", "val", "test"):
            f = META / f"doc_type_{a.tag}_splits" / f"{kind}_{sp}.jsonl"
            if not f.exists(): E("split_file_missing", f.name); continue
            got = {x["doc_id"] for x in read_jsonl(f)}; exp = {r["doc_id"] for r in rows if r["synthetic"] == (kind == "synthetic") and r["split"] == sp}
            if got != exp: E("split_file_disagrees", f.name)
    # groups, duplicates
    g = collections.defaultdict(set)
    for r in rows: g[r["group_id"]].add(r["split"])
    for k, v in g.items():
        if len(v) > 1: E("group_spans_splits", f"{k}: {sorted(v)}")
    bysha = collections.defaultdict(list)
    for r in rows: bysha[r["sha256"]].append(r)
    for sha, v in bysha.items():
        if len(v) > 1:
            (E if len({x["split"] for x in v}) > 1 else (lambda k, d: warn[k].append(d)))("duplicate_bytes_across_splits" if len({x["split"] for x in v}) > 1 else "duplicate_bytes_same_split", [x["doc_id"] for x in v][:3])
    if any(not r["synthetic"] and r["source"] == "synthgen_v1" for r in rows): E("generator_doc_marked_real", "")
    # generator consistency + deny-list
    import generate_synthetic_docs as G
    n_gen = n_bad = 0
    for r in rows:
        if r["source"] != "synthgen_v1": continue
        ann = json.loads((DS / r["ann_path"]).read_text()); n_gen += 1
        if ann["doc_type"] != r["doc_type"] or ann["subtype"] != r["subtype"]: E("ann_label_disagrees", r["doc_id"])
        bad = consistency(ann)
        if bad: n_bad += 1; E("generator_record_inconsistent", f"{r['doc_id']}: {bad}")
        low = " " + ann["issuer"].lower() + " "
        if any(f" {d} " in low for d in G.DENY): E("deny_listed_brand_in_issuer", f"{r['doc_id']}: {ann['issuer']}")
        if not ann["synthetic"] or ann["country"] != "IN": E("generator_flags", r["doc_id"])
    rep = dict(manifest=str(Path(a.manifest)), n_rows=len(rows), files_checked=dict(images=n_img, pdfs=n_pdf, checksums_recomputed=n_hash, legacy_full_hash=a.full_hash, legacy_sample=len(sample)), generated_docs_checked=n_gen,
               errors={k: dict(count=len(v), examples=v[:5]) for k, v in err.items()}, warnings={k: dict(count=len(v), examples=v[:5]) for k, v in warn.items()}, passed=not err)
    Path(a.report).write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ("n_rows", "files_checked", "generated_docs_checked", "errors", "warnings", "passed")}, indent=1)); sys.exit(0 if rep["passed"] else 1)
if __name__ == "__main__": main()
