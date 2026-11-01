"""Validate a real-document COLLECTION manifest (JSONL) against reports/doc_type/licensing_and_collection/collection_manifest_schema_v1.json, enforce the approval gates and privacy rules, and report coverage versus collection_targets_v1.json.
Never reads document files. Exit code 1 on any error.   Usage: collection_manifest.py MANIFEST.jsonl [--report out.json]"""
import sys, re, json, argparse, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; DIR = ROOT / "reports/doc_type/licensing_and_collection"
SCHEMA = json.loads((DIR / "collection_manifest_schema_v1.json").read_text()); TARGETS = json.loads((DIR / "collection_targets_v1.json").read_text())
PII = {"PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "AADHAAR": re.compile(r"\b[2-9][0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}\b"), "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "PHONE": re.compile(r"(?<![0-9])(?:\+?91[ -]?)?[6-9][0-9]{9}(?![0-9])"),
       "LONG_NUMBER": re.compile(r"[0-9][0-9 -]{7,}[0-9]"), "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "GSTIN": re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b")}
HEX64 = re.compile(r"^[0-9a-f]{64}$"); HEX16 = re.compile(r"^[0-9a-f]{16}$"); ID = re.compile(r"^COL-[0-9a-f]{8}-[0-9a-f]{4}$"); IG = re.compile(r"^IG-[0-9a-f]{10}$"); DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
def strings(o):
    if isinstance(o, str): yield o
    elif isinstance(o, dict):
        for v in o.values(): yield from strings(v)
    elif isinstance(o, list):
        for v in o: yield from strings(v)
def validate(rows):
    err = collections.defaultdict(list); E = lambda k, d: err[k].append(d); req = SCHEMA["required"]; seen = set(); red = set(); shas = set(); grp = collections.defaultdict(set)
    for r in rows:
        did = str(r.get("doc_id")); miss = [k for k in req if k not in r]
        if miss: E("missing_fields", f"{did}: {miss}"); continue
        extra = sorted(set(r) - set(req) - set(SCHEMA.get("optional", [])))
        if extra: E("unknown_fields", f"{did}: {extra}")
        if not ID.match(did): E("bad_doc_id", did)
        if did in seen: E("duplicate_doc_id", did)
        seen.add(did)
        if r["doc_type"] not in SCHEMA["document_type_enum"]: E("bad_doc_type", f"{did}: {r['doc_type']}")
        if r["expense_category"] is not None and r["expense_category"] not in SCHEMA["expense_category_enum"]: E("bad_expense_category", f"{did}: {r['expense_category']}")
        if not isinstance(r["tax_evidence"], list) or not set(r["tax_evidence"]) <= set(SCHEMA["tax_evidence_enum"]): E("bad_tax_evidence", did)
        if r["real"] is not True: E("synthetic_or_non_real_in_collection", did)
        if r["permission_basis"] not in SCHEMA["permission_basis_enum"]: E("bad_permission_basis", did)
        if not isinstance(r["permission_ref"], str) or not r["permission_ref"].strip(): E("missing_permission_ref", did)
        if not IG.match(str(r["issuer_group_id"])): E("bad_issuer_group_id", did)
        if r["split"] not in SCHEMA["split_enum"]: E("bad_split", did)
        if r["annotation_status"] not in SCHEMA["annotation_status_enum"]: E("bad_annotation_status", did)
        if r["sensitive_review_status"] not in SCHEMA["sensitive_review_status_enum"]: E("bad_sensitive_review_status", did)
        if not isinstance(r["ambiguity_flags"], list) or not set(r["ambiguity_flags"]) <= set(SCHEMA["ambiguity_flag_enum"]): E("bad_ambiguity_flags", did)
        if r["eligibility_status"] not in ("pending_verification", "not_applicable"): E("eligibility_status_invalid", f"{did}: {r['eligibility_status']}")
        if not re.match(SCHEMA["country_pattern"], str(r["country"])): E("bad_country", did)
        if not DATE.match(str(r["created"])): E("bad_date", did)
        if r["split"] in ("train", "val", "test"):
            if r["sensitive_review_status"] != "redacted_verified": E("gate_split_without_verified_redaction", did)
            if not r["redacted_copy"]: E("gate_split_without_redacted_copy", did)
            if r["split"] in ("val", "test") and r["annotation_status"] not in ("double_annotated", "adjudicated"): E("gate_eval_split_without_double_annotation", did)
            if r["split"] in ("val", "test") and "specimen_or_template" in r["ambiguity_flags"]: E("gate_specimen_in_eval_split", did)
        if r.get("redacted_sha256") is not None:
            if not HEX64.match(str(r["redacted_sha256"])): E("bad_redacted_sha256", did)
            elif r["redacted_sha256"] in shas: E("duplicate_redacted_sha256", did)
            shas.add(r["redacted_sha256"])
        if r.get("redacted_phash") is not None and not HEX16.match(str(r["redacted_phash"])): E("bad_redacted_phash", did)
        if r["redacted_copy"]:
            if r["redacted_copy"] in red: E("duplicate_redacted_copy", did)
            red.add(r["redacted_copy"])
            if r["sensitive_review_status"] != "redacted_verified": E("redacted_copy_without_verified_status", did)
            if str(r["redacted_copy"]).startswith("/") or ".." in str(r["redacted_copy"]): E("redacted_copy_must_be_relative", did)
        grp[r["issuer_group_id"]].add(r["split"] if r["split"] != "unassigned" else None)
        for s in strings({k: v for k, v in r.items() if k not in ("doc_id", "issuer_group_id", "permission_ref", "created", "redacted_sha256", "redacted_phash")}):
            for name, rx in PII.items():
                if rx.search(s): E("personal_data_pattern_" + name, f"{did}: field text matches {name}")
    for g, sp in grp.items():
        if len({s for s in sp if s}) > 1: E("issuer_group_spans_splits", g)
    return {k: v for k, v in err.items()}
def coverage(rows):
    ok = [r for r in rows if r.get("split") in ("train", "val", "test") and r.get("sensitive_review_status") == "redacted_verified"]; cnt = collections.Counter((r["doc_type"], r["split"]) for r in ok); mn = TARGETS["per_class_minimum"]; out = []
    for p in TARGETS["priority_order"]:
        t = p["doc_type"]; row = dict(rank=p["rank"], doc_type=t, **{s: cnt[(t, s)] for s in ("train", "val", "test")}, target=dict(train=mn["train"], val=mn["val"], test=mn["test"]))
        row["ready_for_per_class_metric"] = row["val"] >= mn["val"] and row["test"] >= mn["test"]; out.append(row)
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("--report"); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]; err = validate(rows); cov = coverage(rows); rep = dict(n_rows=len(rows), errors=err, passed=not err, coverage_vs_targets=cov)
    if a.report: Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)); sys.exit(1 if err else 0)
if __name__ == "__main__": main()
