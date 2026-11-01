"""Independent leakage/duplicate audit of datasets/metadata/doc_type_manifest_v1.jsonl (does not reuse the grouping rule that built the splits). Read-only; writes reports/doc_type/leakage_audit_v1.json.
Checks: (1) identical bytes (sha256) shared across splits or across sources; (2) group_id present in more than one split; (3) perceptual-hash (pHash 8x8) near-duplicates: for every real val/test document the
Hamming distance to its nearest TRAIN document (<=4 near-identical, <=8 similar template), reported per document type; (4) external synthetic documents vs real val/test (cross-domain contamination).
Images that cannot be decoded are counted separately and never treated as matches. PDFs: first page rendered at low resolution."""
import sys, json, collections, subprocess, tempfile, io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, imagehash
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]; DS = ROOT / "datasets"
def phash(path):
    try:
        p = DS / path
        if p.suffix.lower() == ".pdf":
            with tempfile.TemporaryDirectory() as d:
                subprocess.run(["pdftoppm", "-r", "30", "-f", "1", "-l", "1", "-png", "-singlefile", str(p), d + "/p"], capture_output=True, timeout=60); im = Image.open(d + "/p.png").convert("L"); im.load()
        else:
            im = Image.open(p).convert("L"); im.load()
        return np.array(imagehash.phash(im, 8).hash).ravel()
    except Exception: return None
if __name__ == "__main__":
    rows = [json.loads(l) for l in open(DS / "metadata/doc_type_manifest_v1.jsonl")]; out = {"n_docs": len(rows)}
    # 1 identical bytes
    by = collections.defaultdict(list)
    for r in rows: by[r["sha256"]].append(r)
    dup = [v for v in by.values() if len(v) > 1]; x_split = [v for v in dup if len({r["split"] for r in v}) > 1]; x_src = [v for v in dup if len({r["source"] for r in v}) > 1]
    out["identical_bytes"] = dict(duplicate_groups=len(dup), docs_in_duplicate_groups=sum(len(v) for v in dup), groups_spanning_splits=len(x_split), groups_spanning_sources=len(x_src),
                                  spanning_splits_examples=[[r["doc_id"] + "|" + r["split"] for r in v][:4] for v in x_split[:5]])
    # 2 groups across splits
    g = collections.defaultdict(set)
    for r in rows: g[r["group_id"]].add(r["split"])
    out["group_id_spanning_splits"] = sum(len(s) > 1 for s in g.values())
    # 3 pHash
    cand = [r for r in rows if not r["synthetic"]]
    with ProcessPoolExecutor(8) as ex: H = list(ex.map(phash, [r["file_path"] for r in cand], chunksize=32))
    ok = [i for i, h in enumerate(H) if h is not None]; out["real_docs"] = len(cand); out["undecodable_images_skipped"] = len(cand) - len(ok)
    M = np.stack([H[i] for i in ok]); R = [cand[i] for i in ok]; sp = np.array([r["split"] for r in R]); tr = np.where(sp == "train")[0]; per = collections.defaultdict(lambda: collections.Counter())
    for s in ("val", "test"):
        for i in np.where(sp == s)[0]:
            d = (M[tr] != M[i]).sum(1); m = int(d.min()); c = per[(s, R[i]["doc_type"])]; c["n"] += 1; c["le4"] += m <= 4; c["le8"] += m <= 8; c["identical"] += m == 0
    out["phash_nearest_train_neighbour"] = {f"{s}|{t}": dict(v) for (s, t), v in sorted(per.items())}
    out["phash_totals"] = {s: dict(n=int(sum(v["n"] for (a, t), v in per.items() if a == s)), le4=int(sum(v["le4"] for (a, t), v in per.items() if a == s)), le8=int(sum(v["le8"] for (a, t), v in per.items() if a == s)), identical=int(sum(v["identical"] for (a, t), v in per.items() if a == s))) for s in ("val", "test")}
    # 4 external synthetic vs real val/test
    syn = [r for r in rows if r["source"] == "agami_indian_bank_statements"]
    if syn:
        with ProcessPoolExecutor(8) as ex: SH = list(ex.map(phash, [r["file_path"] for r in syn], chunksize=8))
        SH = [h for h in SH if h is not None]; ev = np.where(sp != "train")[0]; best = [int((M[ev] != h).sum(1).min()) for h in SH] if SH and len(ev) else []
        out["external_synthetic_vs_real_val_test"] = dict(n_synthetic=len(syn), decoded=len(SH), min_hamming_to_any_real_val_test=min(best) if best else None, n_le8=sum(b <= 8 for b in best))
    (ROOT / "reports/doc_type/leakage_audit_v1.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != "phash_nearest_train_neighbour"}, indent=1))
    print("per type (val|test: n, <=4, <=8):"); [print(f"  {k:28s} n={v['n']:4d} le4={v['le4']:3d} le8={v['le8']:3d} identical={v['identical']}") for k, v in out["phash_nearest_train_neighbour"].items()]
