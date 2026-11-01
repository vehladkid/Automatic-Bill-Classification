"""Build the DOCUMENT-TYPE evaluation/training manifest (doc-type-schema-v1) from what is already in the project plus verified external downloads.
Deterministic and re-runnable. Does NOT modify datasets/metadata/manifest.jsonl, any split, any annotation, the V1 snapshot or the V2 track.
 * Source labels are mapped by rule (config/doc_type_schema_v1.json::source_label_map); the ORIGINAL labels are kept beside the mapped one. No visual re-labelling.
 * Splits are INHERITED from the existing manifest (so the frozen test split stays untouched); external synthetic data is train-only and never eval-eligible.
 * eval_eligible = real (non-synthetic) document. Synthetic documents are a pipeline check, never a result.
Outputs: datasets/metadata/doc_type_manifest_v1.jsonl, doc_type_manifest_v1_stats.json (per-class x split x real/synthetic x Indian, imbalance, requested-category coverage)."""
import sys, json, hashlib, collections, subprocess, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; META = ROOT / "datasets/metadata"
SCHEMA = json.loads((ROOT / "config/doc_type_schema_v1.json").read_text()); MAP = SCHEMA["source_label_map"]
FAMILY = {t: f for f, ts in SCHEMA["axes"]["A_document_type"]["families"].items() for t in ts}
DERIVE = SCHEMA["axes"]["C_tax_evidence"]["default_derivation"]
REQUESTED = {   # requested business category -> document types whose REAL labelled examples would establish coverage
    "EV-related expense/benefit documents": ["ev_document"], "Sodexo / meal vouchers": ["meal_voucher_statement"], "HRA / rent receipts + rental agreements": ["rent_receipt", "rental_agreement"],
    "School-fee receipts": ["school_fee_receipt"], "Section 80C evidence": ["investment_certificate"], "Section 80D / insurance premium receipts": ["insurance_premium_receipt"],
    "Loan principal/interest certificates": ["loan_interest_certificate"], "Form 16": ["form_16"], "Salary slips": ["salary_slip"], "Bank statements": ["bank_statement"], "Invoices": ["invoice"], "Receipts": ["receipt"],
    "Utility bills": ["utility_bill"], "Cheques": ["cheque"]}
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def map_row(r):
    if r["source_dataset"] == "in_financial" and r.get("subcategory") in MAP["in_financial.subcategory"]:
        return MAP["in_financial.subcategory"][r["subcategory"]], "in_financial.subcategory", "source_dataset_label"
    if r["source_dataset"] == "rvl_cdip":
        return ("invoice" if r["document_type"] == "invoice" else "other"), "rvl_cdip.invoice_vs_other", "source_dataset_label"
    dt = MAP["manifest.document_type"].get(r["document_type"])
    return dt, "manifest.document_type", ("synthetic_generator" if r["synthetic"] else "collection_rule")
rows = [json.loads(l) for l in open(META / "manifest.jsonl")]; out = []
for r in rows:
    dtp, rule, basis = map_row(r)
    if dtp is None: continue
    out.append(dict(doc_id=r["document_id"], source=r["source_dataset"], file_path=r["file_path"], sha256=r["sha256"], doc_type=dtp, family=FAMILY[dtp], secondary_types=[], mapping_rule=rule, label_basis=basis,
                    original=dict(document_type=r["document_type"], expense_category=r["expense_category"], subcategory=r.get("subcategory"), label_quality=r["label_quality"], label_source=r["label_source"]),
                    expense_category=r["expense_category"] if r["expense_category"] != "unlabeled" else None, tax_evidence_candidates=DERIVE.get(dtp, []), eligibility_status="pending_verification" if dtp in DERIVE else "not_applicable",
                    country=r["country"], language=r["language"], synthetic=bool(r["synthetic"]), license=r["license"], group_id=r["group_id"], split=r["split"], eval_eligible=not r["synthetic"],
                    ocr_available=r["ocr_available"], ann_path=r.get("ann_path"), n_pages=r.get("n_pages")))
# ---- external, verified downloads (train-only synthetic) ----
AG = ROOT / "datasets/public/agami_bank_statements"
if AG.exists():
    prov = {json.loads(l)["file"].split("/")[-1] + json.loads(l)["revision"]: json.loads(l) for l in open(META / "provenance_external.jsonl")} if (META / "provenance_external.jsonl").exists() else {}
    for p in sorted(AG.rglob("*.pdf")):
        t = p.parent.name.replace("India_Bank_Statement_", "")
        out.append(dict(doc_id=f"agami__{t}__{p.stem}", source="agami_indian_bank_statements", file_path=str(p.relative_to(ROOT / "datasets")), sha256=sha256(p), doc_type="bank_statement", family="financial_statement", secondary_types=[],
                        mapping_rule="AgamiAI/Indian-Bank-Statements.*", label_basis="synthetic_generator", original=dict(template=t, source_label="bank statement"), expense_category=None, tax_evidence_candidates=[], eligibility_status="not_applicable",
                        country="IN", language="en", synthetic=True, license="apache-2.0 (HF card: AgamiAI/Indian-Bank-Statements, synthetic business current-account statements)", group_id=f"agami__{t}", split="train", eval_eligible=False,
                        ocr_available=False, ann_path=None, n_pages=None))
out.sort(key=lambda x: x["doc_id"])
with open(META / "doc_type_manifest_v1.jsonl", "w") as f:
    for x in out: f.write(json.dumps(x, sort_keys=True) + "\n")
# ---- stats ----
cell = lambda x: ("synthetic" if x["synthetic"] else "real") + ("_IN" if x["country"] == "IN" else "")
tab = collections.defaultdict(lambda: collections.Counter())
for x in out: tab[x["doc_type"]][f"{x['split']}|{cell(x)}"] += 1
classes = {}
for t in sorted(tab):
    c = tab[t]; real = {s: sum(v for k, v in c.items() if k.startswith(s + "|real")) for s in ("train", "val", "test")}; ind = {s: c.get(f"{s}|real_IN", 0) for s in ("train", "val", "test")}
    classes[t] = dict(family=FAMILY[t], real=real, real_indian=ind, synthetic=sum(v for k, v in c.items() if "synthetic" in k), total=sum(c.values()))
pol = SCHEMA["coverage_requirements"]
def verdict(c):
    if c is None or sum(c["real"].values()) == 0: return "NO REAL LABELLED DATA"
    ok_tr = c["real"]["train"] >= pol["min_real_train_per_class_to_train"]; ok_ev = c["real"]["val"] >= pol["min_real_val_per_class_to_report_metrics"] and c["real"]["test"] >= pol["min_real_test_per_class_to_report_metrics"]
    ind = min(c["real_indian"].values()) >= pol["min_real_indian_docs_per_class_for_indian_claims"]
    return "trainable+evaluable" if ok_tr and ok_ev else ("trainable, too few val/test for per-class metrics" if ok_tr else "too few real examples to train or evaluate") + ("" if ind else "; insufficient Indian documents for Indian-specific claims")
cov = {k: [dict(doc_type=t, **(classes.get(t) or dict(real={"train": 0, "val": 0, "test": 0}, real_indian={"train": 0, "val": 0, "test": 0}, synthetic=0, total=0)), verdict=verdict(classes.get(t))) for t in v] for k, v in REQUESTED.items()}
real_tot = {t: sum(c["real"].values()) for t, c in classes.items() if sum(c["real"].values())}
stats = dict(schema_version=SCHEMA["schema_version"], n_docs=len(out), builder_sha256=sha256(Path(__file__)), schema_sha256=sha256(ROOT / "config/doc_type_schema_v1.json"), source_manifest_sha256=sha256(META / "manifest.jsonl"),
             policy=pol, classes=classes, requested_category_coverage=cov, imbalance=dict(largest_real_class=max(real_tot, key=real_tot.get), largest=max(real_tot.values()), smallest_real_class=min(real_tot, key=real_tot.get), smallest=min(real_tot.values()),
             ratio=round(max(real_tot.values()) / min(real_tot.values()), 1)), inherited_splits=True, note="splits inherited from datasets/metadata/manifest.jsonl; external synthetic is train-only and not eval-eligible")
(META / "doc_type_manifest_v1_stats.json").write_text(json.dumps(stats, indent=1)); print(f"manifest rows: {len(out)}")
print(f"{'doc_type':28s} real tr/va/te      Indian tr/va/te   synth  verdict")
for t, c in classes.items(): print(f"{t:28s} {c['real']['train']:4d}/{c['real']['val']:3d}/{c['real']['test']:3d}   {c['real_indian']['train']:4d}/{c['real_indian']['val']:3d}/{c['real_indian']['test']:3d}   {c['synthetic']:5d}  {verdict(c)}")
print("\nrequested categories with NO real labelled data:", [k for k, v in cov.items() if all(x['verdict'] == 'NO REAL LABELLED DATA' for x in v)])
print("imbalance:", stats["imbalance"])
