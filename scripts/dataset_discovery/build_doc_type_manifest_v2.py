"""Build the UNIFIED document-type manifest v2 (schema doc-type-schema-v2) - deterministic, re-runnable, never modifies existing datasets, manifests, splits, snapshots or the V2 annotation track.
Inputs : datasets/metadata/manifest.jsonl (existing real + older synthetic), AgamiAI bank statements, FATURA subset, generated corpus datasets/synthetic_doctype/v1 (synthgen-v1).
Outputs: datasets/metadata/doc_type_manifest_v2.jsonl (all), doc_type_v2_real.jsonl, doc_type_v2_synthetic.jsonl, doc_type_v2_splits/*, doc_type_v2_stats.json
Rules  : source labels mapped by rule only (original label kept); real documents inherit the existing split (protected test documents stay test-only and are flagged); synthetic documents are never eval-eligible and are
         split by issuer/template groups; synthetic and real are reported separately everywhere."""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import ROOT, DS, META, sha256_file, load_schema, family_of, synthetic_split, template_split, read_jsonl, write_jsonl
from make_doc_type_splits import write_split_files
SCH = load_schema("v2"); MAP = SCH["source_label_map"]; FAM = family_of(SCH); DERIVE = SCH["axes"]["C_tax_evidence"]["default_derivation"]
REG_ID = {"rvl_cdip": "rvl_cdip", "agami": "agami_indian_bank_statements"}
def reg_id(src): return REG_ID.get(src, src)
def base(**kw):
    kw.setdefault("secondary_types", []); kw["family"] = FAM[kw["doc_type"]]; kw["tax_evidence_candidates"] = DERIVE.get(kw["doc_type"], []); kw["eligibility_status"] = "pending_verification" if kw["doc_type"] in DERIVE else "not_applicable"
    kw.setdefault("generator_version", None); kw.setdefault("seed", None); kw.setdefault("layout", None); kw.setdefault("ann_path", None); kw.setdefault("ocr_available", False); kw.setdefault("subtype", None); kw.setdefault("expense_category", None)
    kw["license_register_id"] = reg_id(kw["source"]); return kw
def map_existing(r):
    if r["source_dataset"] == "in_financial" and r.get("subcategory") in MAP["in_financial.subcategory"]: return MAP["in_financial.subcategory"][r["subcategory"]], "in_financial.subcategory", "source_dataset_label"
    if r["source_dataset"] == "rvl_cdip": return ("invoice" if r["document_type"] == "invoice" else "other"), "rvl_cdip.invoice_vs_other", "source_dataset_label"
    return MAP["manifest.document_type"][r["document_type"]], "manifest.document_type", ("synthetic_generator" if r["synthetic"] else "collection_rule")
rows = []; excluded = []
for r in read_jsonl(META / "manifest.jsonl"):
    try: dtp, rule, basis = map_existing(r)
    except KeyError: excluded.append(dict(doc_id=r["document_id"], reason=f"no mapping for document_type {r['document_type']!r}")); continue
    syn = bool(r["synthetic"])
    rows.append(base(doc_id=r["document_id"], source=r["source_dataset"], file_path=r["file_path"], pages=[r["file_path"]], n_pages=r.get("n_pages") or 1, sha256=r["sha256"], page_sha256=[r["sha256"]], doc_type=dtp, mapping_rule=rule, label_basis=basis,
                    original=dict(document_type=r["document_type"], expense_category=r["expense_category"], subcategory=r.get("subcategory"), label_quality=r["label_quality"], label_source=r["label_source"]),
                    expense_category=None if r["expense_category"] == "unlabeled" else r["expense_category"], country=r["country"], language=r["language"], synthetic=syn, license=r["license"], group_id=r["group_id"], split=r["split"], split_basis="inherited_from_manifest.jsonl",
                    eval_eligible=not syn, in_protected_task_test=(r["split"] == "test" and not syn), review_status=("generator_exact" if syn else ("source_label" if basis == "source_dataset_label" else "needs_mapping_review")),
                    label_confidence=("generator_exact" if syn else ("source" if basis == "source_dataset_label" else "collection_rule")), ocr_available=bool(r["ocr_available"]), ann_path=r.get("ann_path")))
# ---- AgamiAI bank statements (verified download, train-only synthetic)
AG = DS / "public/agami_bank_statements"
for p in sorted(AG.rglob("*.pdf")) if AG.exists() else []:
    t = p.parent.name.replace("India_Bank_Statement_", ""); js = p.with_suffix(".json"); sha = sha256_file(p)
    rows.append(base(doc_id=f"agami__{t}__{p.stem}", source="agami", file_path=str(p.relative_to(DS)), pages=[str(p.relative_to(DS))], n_pages=None, sha256=sha, page_sha256=[sha], doc_type="bank_statement", subtype=t.lower(), mapping_rule="AgamiAI/Indian-Bank-Statements.*", label_basis="synthetic_generator",
                     original=dict(template=t, source_label="bank statement"), country="IN", language="en", synthetic=True, license="Apache-2.0 (HF card AgamiAI/Indian-Bank-Statements)", group_id=f"agami__{t}", split="train", split_basis="train_only_single_generator", eval_eligible=False,
                     in_protected_task_test=False, review_status="generator_exact", label_confidence="generator_exact", ann_path=str(js.relative_to(DS)) if js.exists() else None))
# ---- FATURA subset (template-grouped split)
FI = DS / "public/fatura/subset_index.jsonl"
if FI.exists():
    fr = read_jsonl(FI); temps = sorted({x["template"] for x in fr})
    for x in fr:
        rows.append(base(doc_id=x["doc_id"], source="fatura", file_path=x["file"], pages=[x["file"]], n_pages=1, sha256=x["sha256"], page_sha256=[x["sha256"]], doc_type="invoice", subtype="fatura_template", mapping_rule="fatura.*", label_basis="synthetic_generator",
                         original=dict(template=x["template"], instance=x["instance"], archive_member=x["archive_member"]), country=None, language="en", synthetic=True, license="CC-BY-4.0 (Zenodo 8261508)", group_id=f"fatura__t{x['template']}", split=template_split(x["template"], temps),
                         split_basis="template_grouped_35_7_8", eval_eligible=False, in_protected_task_test=False, review_status="generator_exact", label_confidence="generator_exact"))
# ---- generated Indian documents (synthgen-v1)
GD = DS / "synthetic_doctype/v1"
for f in sorted((GD / "ann").glob("*.json")) if GD.exists() else []:
    a = json.loads(f.read_text()); pg = [str((GD / p).relative_to(DS)) for p in a["pages"]]; ph = [sha256_file(DS / p) for p in pg]
    rows.append(base(doc_id=a["doc_id"], source="synthgen_v1", file_path=pg[0], pages=pg, n_pages=a["n_pages"], sha256=ph[0], page_sha256=ph, doc_type=a["doc_type"], subtype=a["subtype"], mapping_rule="synthgen-v1.doc_type", label_basis="synthetic_generator",
                     original=dict(title=a["title"], issuer=a["issuer"], degrade=a["degrade"]["mode"], layout_note=a["layout_note"]), country="IN", language=a["language"], synthetic=True, license="generated by this project; fictional data", group_id=a["issuer_group"],
                     split=synthetic_split(a["issuer_group"], a["layout"]), split_basis="synthetic_issuer_grouped_70_15_15_layoutD_test_only", eval_eligible=False, in_protected_task_test=False, review_status="generator_exact", label_confidence="generator_exact",
                     generator_version=a["generator_version"], seed=a["doc_seed"], layout=a["layout"], ocr_available=True, ann_path=str((GD / "ann" / f.name).relative_to(DS))))
rows.sort(key=lambda r: r["doc_id"]); ids = [r["doc_id"] for r in rows]; assert len(ids) == len(set(ids)), "duplicate doc_id"
write_jsonl(META / "doc_type_manifest_v2.jsonl", rows); write_jsonl(META / "doc_type_v2_real.jsonl", [r for r in rows if not r["synthetic"]]); write_jsonl(META / "doc_type_v2_synthetic.jsonl", [r for r in rows if r["synthetic"]])
nsp = write_split_files(rows)
# ---- stats
def cnt(sel, key): return dict(collections.Counter(key(r) for r in sel).most_common())
real = [r for r in rows if not r["synthetic"]]; syn = [r for r in rows if r["synthetic"]]; pol = SCH["coverage_requirements"]
per = {}
for t in sorted({r["doc_type"] for r in rows}):
    rr = [r for r in real if r["doc_type"] == t]; ss = [r for r in syn if r["doc_type"] == t]
    per[t] = dict(family=FAM[t], real={s: sum(r["split"] == s for r in rr) for s in ("train", "val", "test")}, real_indian={s: sum(r["split"] == s and r["country"] == "IN" for r in rr) for s in ("train", "val", "test")},
                  synthetic={s: sum(r["split"] == s for r in ss) for s in ("train", "val", "test")}, real_total=len(rr), synthetic_total=len(ss), subtypes_real=cnt(rr, lambda r: r["subtype"]), subtypes_synthetic=cnt(ss, lambda r: r["subtype"]))
def readiness(c):
    real_ok_tr = c["real"]["train"] >= pol["min_real_train_per_class_to_train"]; real_ok_ev = c["real"]["val"] >= pol["min_real_val_per_class_to_report_metrics"] and c["real"]["test"] >= pol["min_real_test_per_class_to_report_metrics"]
    if c["real_total"] == 0: return "NOT READY for real-world claims: no real labelled documents; synthetic-only development possible" if c["synthetic_total"] else "NOT READY: no data"
    if real_ok_tr and real_ok_ev: return "READY (real train/val/test meet policy)" + ("" if min(c["real_indian"].values()) >= pol["min_real_indian_docs_per_class_for_indian_claims"] else "; Indian-specific claims need more Indian docs")
    return "PARTIAL: real examples below policy (train>=150, val>=30, test>=30)" + ("; synthetic available for development" if c["synthetic_total"] else "")
for t, c in per.items(): c["readiness"] = readiness(c)
stats = dict(schema_version=SCH["schema_version"], n_docs=len(rows), n_real=len(real), n_synthetic=len(syn), excluded=excluded, per_type=per, real_by_source=cnt(real, lambda r: r["source"]), synthetic_by_source=cnt(syn, lambda r: r["source"]), real_by_country=cnt(real, lambda r: r["country"]), real_by_language=cnt(real, lambda r: r["language"]),
             synthetic_by_country=cnt(syn, lambda r: r["country"]), synthetic_by_language=cnt(syn, lambda r: r["language"]), review_status=cnt(rows, lambda r: r["review_status"]), splits_files=nsp, protected_test_docs=sum(r["in_protected_task_test"] for r in rows),
             synthetic_layout=cnt([r for r in syn if r["layout"]], lambda r: r["layout"]), real_class_imbalance=dict(largest=max((c["real_total"], t) for t, c in per.items())[::-1], smallest_nonzero=min((c["real_total"], t) for t, c in per.items() if c["real_total"])[::-1]))
(META / "doc_type_v2_stats.json").write_text(json.dumps(stats, indent=1)); print(f"unified manifest: {len(rows)} docs = {len(real)} real + {len(syn)} synthetic | excluded {len(excluded)} | split files {nsp}")
print(f"{'doc_type':28s} real tr/va/te   Indian        synth tr/va/te    readiness")
for t, c in per.items(): print(f"{t:28s} {c['real']['train']:4d}/{c['real']['val']:3d}/{c['real']['test']:3d}  {sum(c['real_indian'].values()):4d}   {c['synthetic']['train']:4d}/{c['synthetic']['val']:3d}/{c['synthetic']['test']:3d}   {c['readiness'][:78]}")
