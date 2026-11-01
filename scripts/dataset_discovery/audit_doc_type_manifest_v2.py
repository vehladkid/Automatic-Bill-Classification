"""Independent duplicate / leakage audit of the UNIFIED document-type manifest v2. Read-only; writes reports/doc_type/leakage_audit_v2.json.
 (1) byte-identical files (sha256) - across splits and across real/synthetic; (2) group_id (issuer/template/source group) spanning splits; (3) perceptual-hash (pHash 8x8, first page) nearest-neighbour analysis:
     a. REAL val/test -> nearest REAL train (per type)  b. SYNTHETIC val/test -> nearest SYNTHETIC train, split into seen layouts (A/B/C) vs the unseen layout D  c. REAL vs SYNTHETIC cross-domain contamination
 (4) generated-issuer-name reuse across splits (weak leak: same fictional name in train and test).
Note on (3b): generated documents of one layout share structure by construction, so small pHash distances between train and val/test of the SAME layout are expected and are the reason layout D is held out as test-only.
Undecodable images are counted and never treated as matches. PDFs: first page rendered at low resolution."""
import sys, json, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import ROOT, DS, META, read_jsonl
from audit_doc_type_manifest import phash
def main():
    rows = read_jsonl(META / "doc_type_manifest_v2.jsonl"); out = dict(n_docs=len(rows), n_real=sum(not r["synthetic"] for r in rows), n_synthetic=sum(r["synthetic"] for r in rows))
    by = collections.defaultdict(list)
    for r in rows: by[r["sha256"]].append(r)
    dup = [v for v in by.values() if len(v) > 1]
    out["identical_bytes"] = dict(duplicate_groups=len(dup), docs_in_groups=sum(len(v) for v in dup), spanning_splits=sum(len({r["split"] for r in v}) > 1 for v in dup), spanning_real_and_synthetic=sum(len({r["synthetic"] for r in v}) > 1 for v in dup), examples=[[r["doc_id"] for r in v][:3] for v in dup[:5]])
    g = collections.defaultdict(set)
    for r in rows: g[r["group_id"]].add(r["split"])
    out["group_id_spanning_splits"] = dict(groups=len(g), spanning=sum(len(v) > 1 for v in g.values()))
    with ProcessPoolExecutor(8) as ex: H = list(ex.map(phash, [r["file_path"] for r in rows], chunksize=32))
    ok = [i for i, h in enumerate(H) if h is not None]; out["undecodable_first_pages_skipped"] = len(rows) - len(ok); R = [rows[i] for i in ok]; M = np.stack([H[i] for i in ok]); dist = lambda i, idx: (M[idx] != M[i]).sum(1)
    syn = np.array([r["synthetic"] for r in R]); sp = np.array([r["split"] for r in R]); src = np.array([r["source"] for r in R]); lay = np.array([r["layout"] or "" for r in R]); typ = np.array([r["doc_type"] for r in R])
    # a. real
    tr = np.where(~syn & (sp == "train"))[0]; a = {}
    for s in ("val", "test"):
        c = collections.Counter(); 
        for i in np.where(~syn & (sp == s))[0]: d = int(dist(i, tr).min()); c["n"] += 1; c["le4"] += d <= 4; c["le8"] += d <= 8; c["identical"] += d == 0
        a[s] = dict(c)
    out["a_real_val_test_vs_real_train"] = a
    # b. synthetic generated (synthgen) val/test vs synthetic train, seen vs unseen layout
    gen = src == "synthgen_v1"; trg = np.where(gen & (sp == "train"))[0]; b = {}
    for s in ("val", "test"):
        for kind, sel in (("seen_layout_ABC", np.isin(lay, ["A", "B", "C"])), ("unseen_layout_D", lay == "D")):
            idx = np.where(gen & (sp == s) & sel)[0]
            if not len(idx): continue
            c = collections.Counter()
            for i in idx: d = int(dist(i, trg).min()); c["n"] += 1; c["le4"] += d <= 4; c["le8"] += d <= 8
            b[f"{s}|{kind}"] = dict(c)
    out["b_generated_val_test_vs_generated_train"] = b
    # c. cross-domain
    realev = np.where(~syn & (sp != "train"))[0]; syn_all = np.where(syn)[0]; mn = {}
    for name, idx in (("synthgen_v1", np.where(gen)[0]), ("agami", np.where(src == "agami")[0]), ("fatura", np.where(src == "fatura")[0]), ("older_synthetic_in", np.where(src == "synthetic_in")[0])):
        ds_ = [int(dist(i, realev).min()) for i in idx]; mn[name] = dict(n=int(len(idx)), min_distance_to_any_real_val_test=min(ds_) if ds_ else None, n_le4=int(sum(d <= 4 for d in ds_)), n_le8=int(sum(d <= 8 for d in ds_)))
    out["c_synthetic_vs_real_val_test"] = mn
    # 4. issuer reuse
    iss = collections.defaultdict(set)
    for r in rows:
        if r["source"] == "synthgen_v1": iss[(r["doc_type"], r["original"]["issuer"])].add(r["split"])
    out["d_generated_issuer_name_in_multiple_splits"] = dict(distinct_issuers=len(iss), spanning=sum(len(v) > 1 for v in iss.values()), note="same fictional name drawn for different documents; affects names only, not layout/content")
    (ROOT / "reports/doc_type/leakage_audit_v2.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
if __name__ == "__main__": main()
