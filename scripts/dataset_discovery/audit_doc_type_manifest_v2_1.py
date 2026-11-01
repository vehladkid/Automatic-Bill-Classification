"""Independent audit of manifest v2.1 (does not reuse the validator's code paths). Read-only; writes reports/doc_type/audit_v2_1.json.
 A. file existence + sha256 of EVERY page file, unique doc_id / sha256 counts, per-source counts vs files on disk
 B. real/synthetic separation, approved gate vs licence register, excluded sources absent from approved manifests
 C. original labels + mapping rules preserved: every inherited row is joined back to datasets/metadata/manifest.jsonl (doc type, expense category, split, group)
 D. protected assets: frozen Task B test hash, V1 snapshot checksums, original manifest/splits unchanged (mtime + sha256 recorded), v2 files unchanged
 E. leakage of the generated corpus: template overlap across splits; pHash<=4, pixel NCC>=0.9 and OCR/ground-truth trigram Jaccard vs nearest TRAIN page, split into held-out-template (v2.1) vs the v2 issuer split
 F. real val/test vs real train pHash (unchanged logic) and cross-domain synthetic vs real"""
import sys, json, re, collections, hashlib, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import ROOT, DS, META, CONFIG, read_jsonl, sha256_file
from audit_doc_type_manifest import phash
def fine(path):
    im = Image.open(DS / path).convert("L").resize((64, 64), Image.BILINEAR); a = np.asarray(im, dtype=np.float32).ravel(); a = (a - a.mean()) / (a.std() + 1e-6); return a / 64.0
def trig(r):
    a = json.loads((DS / r["ann_path"]).read_text()); t = re.sub(r"\d", "0", " ".join(l["text"] for l in a["lines"]).lower()); w = t.split(); return set(zip(w, w[1:], w[2:]))
def jacc(a, b): return len(a & b) / max(1, len(a | b))
def hashes(p): return sha256_file(DS / p)
def main():
    rows = read_jsonl(META / "doc_type_manifest_v2_1.jsonl"); old = {r["doc_id"]: r for r in read_jsonl(META / "doc_type_manifest_v2.jsonl")}; out = {}
    # A
    with ProcessPoolExecutor(8) as ex:
        pg = [(r["doc_id"], i, p, r["page_sha256"][i]) for r in rows for i, p in enumerate(r["pages"])]
        got = list(ex.map(hashes, [x[2] for x in pg], chunksize=64))
    miss = [x[0] for x, g in zip(pg, got) if g is None]; bad = [x[0] for x, g in zip(pg, got) if g is not None and g != x[3]]
    out["A_files"] = dict(rows=len(rows), page_files=len(pg), missing_files=len(miss), checksum_mismatch=len(bad), unique_doc_ids=len({r["doc_id"] for r in rows}), unique_page_sha256=len({x[3] for x in pg}), unique_doc_sha256=len({r["sha256"] for r in rows}),
                          per_source={s: dict(rows=n) for s, n in collections.Counter(r["source"] for r in rows).items()}, examples_missing=miss[:3], examples_bad=bad[:3])
    ps = out["A_files"]["per_source"]
    ps["synthgen_v1"]["files_on_disk"] = len(list((DS / "synthetic_doctype/v1/images").rglob("*.*"))); ps["synthgen_v1"]["index_rows"] = sum(1 for _ in open(DS / "synthetic_doctype/v1/index.jsonl"))
    ps["agami"]["files_on_disk"] = len(list((DS / "public/agami_bank_statements").rglob("*.pdf"))); ps["fatura"]["files_on_disk"] = len(list((DS / "public/fatura/subset").glob("*.jpg")))
    # B
    reg = {s["id"]: s for s in json.loads((META / "license_register_v2.json").read_text())["sources"]}
    A = [r for r in rows if r["approved_for_training"]]; manA = read_jsonl(META / "doc_type_v2_1_approved/manifest.jsonl")
    out["B_separation_and_approval"] = dict(synthetic_with_eval_eligible=sum(r["synthetic"] and r["eval_eligible"] for r in rows), real_not_eval_eligible=sum((not r["synthetic"]) and not r["eval_eligible"] for r in rows),
        real_synthetic_same_sha=len({r["sha256"] for r in rows if r["synthetic"]} & {r["sha256"] for r in rows if not r["synthetic"]}), approved_rows=len(A), approved_file_rows=len(manA), approved_file_matches=({r["doc_id"] for r in A} == {r["doc_id"] for r in manA}),
        approved_with_unresolved_licence=[r["doc_id"] for r in A if reg[r["license_register_id"]]["status"] in ("conflict", "unverified", "blocked", "not_acquired")][:5], approved_sources=sorted({r["source"] for r in A}),
        excluded_sources=sorted({r["source"] for r in rows if not r["approved_for_training"]}), blocked_or_not_acquired_sources_present=sorted({r["license_register_id"] for r in rows if reg[r["license_register_id"]]["status"] in ("blocked", "not_acquired")}),
        eligibility_status_values=dict(collections.Counter(r["eligibility_status"] for r in rows)), expense_category_equals_a_doc_type=sorted({r["doc_type"] for r in rows} & {r["expense_category"] for r in rows if r["expense_category"]}))
    # C
    base = {r["document_id"]: r for r in read_jsonl(META / "manifest.jsonl")}; c = collections.Counter(); ex_ = []
    for r in rows:
        b = base.get(r["doc_id"])
        if b is None: continue
        c["joined"] += 1
        if r["original"].get("document_type") != b["document_type"] or r["original"].get("expense_category") != b["expense_category"] or r["original"].get("label_source") != b["label_source"]: c["original_label_changed"] += 1; ex_.append(r["doc_id"])
        if r["split"] != b["split"]: c["split_differs_from_manifest_jsonl"] += 1
        if r["group_id"] != b["group_id"]: c["group_differs"] += 1
        if r["sha256"] != b["sha256"]: c["sha_differs"] += 1
    nomapping = [r["doc_id"] for r in rows if not r["mapping_rule"]]
    out["C_labels_preserved"] = dict(dict(c), inherited_rows_not_in_manifest_jsonl=sum(1 for r in rows if r["source"] not in ("agami", "fatura", "synthgen_v1") and r["doc_id"] not in base), rows_without_mapping_rule=len(nomapping), examples=ex_[:3],
        mapping_rules={k: v for k, v in collections.Counter(r["mapping_rule"] for r in rows).items()}, review_status=dict(collections.Counter(r["review_status"] for r in rows)))
    # D
    D = {}
    D["task_b_test_sha256"] = hashlib.sha256((ROOT / "datasets/processed/taskB/test.jsonl").read_bytes()).hexdigest(); D["task_b_test_expected_prefix_fd8bafb3ad13"] = D["task_b_test_sha256"].startswith("fd8bafb3ad13")
    sn = subprocess.run("sha256sum -c reports/taskB/v1_snapshot/SHA256SUMS.txt 2>&1 | awk '{print $NF}' | sort | uniq -c", shell=True, cwd=ROOT, capture_output=True, text=True).stdout.split(); D["v1_snapshot_sha256sums"] = " ".join(sn)
    D["v2_manifest_unchanged_vs_v2_1_source_fields"] = sum(1 for r in rows if old[r["doc_id"]]["sha256"] != r["sha256"])
    D["original_manifest_files"] = {f: dict(sha256=sha256_file(META / f)[:16], mtime=__import__("time").strftime("%Y-%m-%d %H:%M", __import__("time").localtime((META / f).stat().st_mtime))) for f in ("manifest.jsonl", "manifest_dedup.jsonl", "doc_type_manifest_v2.jsonl")}
    v2split = [r for r in read_jsonl(META / "doc_type_manifest_v2.jsonl")]; D["real_and_synthetic_in_splits_equal_v2_for_non_modified_sources"] = all(old[r["doc_id"]]["split"] == r["split"] for r in rows if r["source"] not in ("synthgen_v1", "agami"))
    D["protected_test_docs_all_still_test"] = all(r["split"] == "test" for r in rows if r["in_protected_task_test"]); D["protected_test_docs"] = sum(r["in_protected_task_test"] for r in rows)
    D["v2_track_dirs_untouched_note"] = "annotation/answer-sheet files not read or written by any v2.1 tool"; out["D_protected"] = D
    # E / F
    with ProcessPoolExecutor(8) as ex:
        H = np.stack(list(ex.map(phash, [r["file_path"] for r in rows], chunksize=32)))
        gi = [i for i, r in enumerate(rows) if r["source"] == "synthgen_v1"]; Fg = np.stack(list(ex.map(fine, [rows[i]["file_path"] for i in gi], chunksize=16)))
    sp = np.array([r["split"] for r in rows]); syn = np.array([r["synthetic"] for r in rows]); src = np.array([r["source"] for r in rows]); lay = np.array([r["layout"] or "" for r in rows]); typ = np.array([r["doc_type"] for r in rows])
    T = {i: trig(rows[i]) for i in gi}; gtr = [i for i in gi if sp[i] == "train"]; Fi = {i: Fg[k] for k, i in enumerate(gi)}; Ftr = np.stack([Fi[i] for i in gtr])
    def leak(split_of, label):
        res = collections.defaultdict(collections.Counter); tr = [i for i in gi if split_of(i) == "train"]; Ft = np.stack([Fi[i] for i in tr])
        for i in gi:
            s = split_of(i)
            if s == "train": continue
            kind = "layout_D" if lay[i] == "D" else "layout_ABC"; d = (H[tr] != H[i]).sum(1); ncc = Ft @ Fi[i]; cand = [t for t in tr if typ[t] == typ[i]]; jv = max(jacc(T[t], T[i]) for t in cand)
            q = res[f"{s}|{kind}"]; q["n"] += 1; q["phash_le4"] += int(d.min() <= 4); q["ncc_ge_0.90"] += int(ncc.max() >= .9); q["ncc_ge_0.95"] += int(ncc.max() >= .95); q["trigram_jaccard_ge_0.6"] += int(jv >= .6); q["same_template_in_train"] += int(any(rows[t]["template_id"] == rows[i]["template_id"] for t in tr))
        return {k: dict(v) for k, v in res.items()}
    out["E_generated_leakage"] = dict(v2_issuer_split=leak(lambda i: old[rows[i]["doc_id"]]["split"], "v2"), v2_1_template_split=leak(lambda i: rows[i]["split"], "v2.1"))
    tid = collections.defaultdict(set)
    for r in rows:
        if r["template_id"]: tid[r["template_id"]].add(r["split"])
    out["E_template_ids_spanning_splits"] = dict(templates=len(tid), spanning=sum(len(v) > 1 for v in tid.values()))
    gg = collections.defaultdict(set)
    for r in rows: gg[r["group_id"]].add(r["split"])
    out["E_group_ids_spanning_splits"] = dict(groups=len(gg), spanning=sum(len(v) > 1 for v in gg.values()))
    trr = np.where(~syn & (sp == "train"))[0]; fr = {}
    for s in ("val", "test"):
        ds_ = [int((H[trr] != H[i]).sum(1).min()) for i in np.where(~syn & (sp == s))[0]]; fr[s] = dict(n=len(ds_), le4=sum(d <= 4 for d in ds_), le8=sum(d <= 8 for d in ds_), identical=sum(d == 0 for d in ds_))
    realev = np.where(~syn & (sp != "train"))[0]; cross = {}
    for name in ("synthgen_v1", "agami", "fatura", "synthetic_in"):
        ds_ = [int((H[realev] != H[i]).sum(1).min()) for i in np.where(src == name)[0]]; cross[name] = dict(n=len(ds_), min=min(ds_), le4=sum(d <= 4 for d in ds_), le8=sum(d <= 8 for d in ds_))
    out["F_real_val_test_vs_train_phash"] = fr; out["F_synthetic_vs_real_val_test_phash"] = cross
    (ROOT / "reports/doc_type/audit_v2_1.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out, indent=1, default=str))
if __name__ == "__main__": main()
