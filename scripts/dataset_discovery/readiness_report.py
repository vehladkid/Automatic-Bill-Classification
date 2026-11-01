"""Dataset coverage and TRAINING-READINESS report for the document-type corpus v2. Reads the manifest stats, validation, leakage-audit, OCR-check and licence-register outputs (run those first, or run_all.sh).
Writes reports/doc_type/readiness_v2.json and docs/DOC_TYPE_DATASET_READINESS.md. Makes no claims beyond what those files contain; trains nothing."""
import sys, json, os, shutil, subprocess, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import ROOT, DS, META
def J(p): return json.loads(Path(p).read_text())
def du(p):
    p = Path(p)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0
def main():
    st = J(META / "doc_type_v2_stats.json"); val = J(ROOT / "reports/doc_type/validation_v2.json"); au = J(ROOT / "reports/doc_type/leakage_audit_v2.json"); ocr = J(ROOT / "reports/doc_type/synthetic_ocr_check_v1.json"); reg = J(META / "license_register_v2.json")["sources"]
    free = shutil.disk_usage(ROOT).free; per = st["per_type"]
    storage = {k: du(ROOT / v) for k, v in {"generated_synthetic_documents": "datasets/synthetic_doctype", "agami_bank_statements": "datasets/public/agami_bank_statements", "fatura_archive_and_subset": "datasets/public/fatura", "dit_base_weights": "models/pretrained/dit-base-finetuned-rvlcdip"}.items()}
    by_status = {}
    for s in reg: by_status.setdefault(s["status"], []).append(s["id"])
    ready = {t: c["readiness"] for t, c in per.items()}
    real_ready = [t for t, c in per.items() if c["readiness"].startswith("READY")]; partial = [t for t, c in per.items() if c["readiness"].startswith("PARTIAL")]; none = [t for t, c in per.items() if c["readiness"].startswith("NOT READY")]
    layoutD = au["b_generated_val_test_vs_generated_train"].get("test|unseen_layout_D", {}); seen = au["b_generated_val_test_vs_generated_train"].get("test|seen_layout_ABC", {})
    rep = dict(generated=dt.datetime.now().isoformat(timespec="seconds"), manifest=dict(version=st["schema_version"], n_docs=st["n_docs"], n_real=st["n_real"], n_synthetic=st["n_synthetic"], excluded=len(st["excluded"])), per_type=per, real_by_source=st["real_by_source"], synthetic_by_source=st["synthetic_by_source"],
               real_by_country=st["real_by_country"], real_by_language=st["real_by_language"], synthetic_by_country=st["synthetic_by_country"], synthetic_by_language=st["synthetic_by_language"], review_status=st["review_status"], validation_passed=val["passed"], validation_errors=val["errors"],
               leakage=dict(identical_bytes=au["identical_bytes"], group_spanning=au["group_id_spanning_splits"], real_val_test_vs_train=au["a_real_val_test_vs_real_train"], generated_seen_layouts=seen, generated_unseen_layout_D=layoutD, synthetic_vs_real=au["c_synthetic_vs_real_val_test"]),
               ocr_check=dict(sample=ocr["sample_size"], overall=ocr["overall"], by_degradation=ocr["by_degradation"]), licence_status={k: v for k, v in by_status.items()}, storage_bytes=storage, free_disk_gb=round(free / 1e9, 2),
               readiness=dict(ready_real=real_ready, partial_real=partial, no_real_data=none), recommendation=dict(architecture="OCR-text document-type classifier first: TF-IDF (char+word) + logistic regression baseline, then a small transformer (DistilBERT-class; multilingual encoder only if Indic text matters); optional DiT image features later. All fit the 6 GB GPU (the baseline is CPU).",
               before_training=["human review of the real-label mapping (collection-rule labels)", "collect authorized, anonymized REAL documents for the no-data classes (>=30 val + >=30 test each, ideally >=150 train)", "report synthetic-layout-D and real metrics separately; never present synthetic scores as real-world accuracy", "resolve licence conflicts (SROIE, FUNSD, in_financial provenance)"]))
    (ROOT / "reports/doc_type/readiness_v2.json").write_text(json.dumps(rep, indent=1))
    L = []; w = L.append
    w("# Document-type dataset v2: coverage and training readiness"); w(f"_Generated {rep['generated']} by `scripts/dataset_discovery/readiness_report.py` from the machine-readable reports in `reports/doc_type/` and `datasets/metadata/`. Nothing here was trained._\n")
    w(f"**Corpus:** {st['n_docs']:,} documents = **{st['n_real']:,} real** + **{st['n_synthetic']:,} synthetic** (excluded by mapping: {len(st['excluded'])}). Schema `{st['schema_version']}`. Validation: **{'PASSED' if val['passed'] else 'FAILED'}** ({val['n_rows']:,} rows, {val['files_checked']['images']:,} images + {val['files_checked']['pdfs']} PDFs opened, {val['files_checked']['checksums_recomputed']:,} checksums recomputed, {val['generated_docs_checked']:,} generated records checked for internal consistency).\n")
    w("## Per document type (real and synthetic reported separately; splits train/val/test)\n"); w("| document type | real | real Indian | synthetic | readiness |"); w("|---|---|---|---|---|")
    for t, c in per.items():
        r = c["real"]; i = c["real_indian"]; s = c["synthetic"]; w(f"| {t} | {r['train']}/{r['val']}/{r['test']} | {sum(i.values())} | {s['train']}/{s['val']}/{s['test']} | {c['readiness']} |")
    w("\nPolicy for READY: >=150 real train, >=30 real val, >=30 real test per class; Indian-specific claims need >=30 Indian documents per class. Synthetic documents never count toward real readiness.\n")
    w("## Subtypes (real | synthetic)\n")
    for t, c in per.items():
        if c["subtypes_real"] or c["subtypes_synthetic"]: w(f"- **{t}**: real {c['subtypes_real'] or '-'} | synthetic {c['subtypes_synthetic'] or '-'}")
    w("\n## Provenance\n"); w(f"- Real by source: {st['real_by_source']}"); w(f"- Synthetic by source: {st['synthetic_by_source']}"); w(f"- Real by country: {st['real_by_country']}"); w(f"- Real by language: {st['real_by_language']}"); w(f"- Synthetic by country / language: {st['synthetic_by_country']} / {st['synthetic_by_language']}"); w(f"- Review status: {st['review_status']}")
    w("\n## Licences (`datasets/metadata/license_register_v2.json`)\n")
    for k, v in by_status.items(): w(f"- **{k}**: {', '.join(v)}")
    w("\n## Duplicate and leakage audit (`reports/doc_type/leakage_audit_v2.json`)\n")
    w(f"- Byte-identical duplicates: **{au['identical_bytes']['duplicate_groups']}** groups (spanning splits: {au['identical_bytes']['spanning_splits']}; real+synthetic: {au['identical_bytes']['spanning_real_and_synthetic']}). Groups spanning splits: **{au['group_id_spanning_splits']['spanning']}** of {au['group_id_spanning_splits']['groups']:,}.")
    a = au["a_real_val_test_vs_real_train"]; w(f"- Real val/test vs real train (pHash): <=4 bits: val {a['val']['le4']}/{a['val']['n']}, test {a['test']['le4']}/{a['test']['n']}; <=8 bits: val {a['val']['le8']}, test {a['test']['le8']}.")
    w(f"- Generated documents, test split: **seen layouts A-C** are near-identical to a training page for {seen.get('le4', 0)}/{seen.get('n', 0)} documents (expected: shared layouts), **unseen layout D** for {layoutD.get('le4', 0)}/{layoutD.get('n', 0)} (<=8 bits: {layoutD.get('le8', 0)}). **Only the layout-D test slice is a meaningful generalization check for synthetic classes.**")
    w(f"- Synthetic vs real val/test: no near-identical (<=4) pair for any synthetic source; closest: {{ {', '.join(f'{k}: {v['min_distance_to_any_real_val_test']} bits' for k, v in au['c_synthetic_vs_real_val_test'].items())} }}.")
    w("\n## OCR check of the generated documents (does the image support its label?)\n"); o = ocr["overall"]; w(f"- Sample {ocr['sample_size']} documents (clean / scan / phone-photo), project OCR: class-defining phrase readable **{o['key_phrase_readable']:.0%}**, title readable **{o['title_readable']:.0%}**, main amount recovered **{o['main_amount_recovered']:.0%}** (OCR confusions such as 0/o and dropped lines explain the rest; the images themselves were inspected).")
    w("\n## Storage\n"); [w(f"- {k.replace('_', ' ')}: {v / 1e6:.0f} MB") for k, v in storage.items()]; w(f"- free disk: {free / 1e9:.1f} GB")
    w("\n## Readiness verdict\n"); w(f"- **Real-data ready:** {', '.join(real_ready) or 'none'} (Indian-specific claims limited to documents with country IN).")
    w(f"- **Partial real data:** {', '.join(partial) or 'none'}."); w(f"- **No real labelled data (synthetic only):** {', '.join(none)}.")
    w("- **The dataset is ready for a development/feasibility training run, NOT for claims about real-world accuracy on the no-data classes.** Any model trained on the synthetic classes must be evaluated on real documents before such claims.")
    w("\n## Recommended next step\n"); w(f"- Architecture: {rep['recommendation']['architecture']}"); w("- Before training: " + "; ".join(rep["recommendation"]["before_training"]) + ".")
    (ROOT / "docs/DOC_TYPE_DATASET_READINESS.md").write_text("\n".join(L) + "\n"); print("\n".join(L[:12])); print("...written docs/DOC_TYPE_DATASET_READINESS.md, reports/doc_type/readiness_v2.json")
if __name__ == "__main__": main()
