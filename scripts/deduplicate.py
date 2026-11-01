"""Exact (sha256) + near-duplicate (pHash & dHash) detection across ALL sources -> manifest_dedup.jsonl.
- near-dup rule: pHash<=4 OR (pHash<=6 AND dHash<=8)
- exact duplicates beyond the first copy are dropped (drop_reason='exact_duplicate')
- near-duplicate clusters + SROIE same-merchant groups are merged into one group_id; splitting is done per group
- synthetic images are excluded from near-dup search (shared templates make perceptual hashes collide by design)."""
import json, collections
import numpy as np, imagehash
from PIL import Image
from common import *

def load(): return [json.loads(l) for l in open(META / "manifest_raw.jsonl")]

def hashes(r):
    with Image.open(DS / r["file_path"]) as im:
        im = im.convert("L")
        return np.array(imagehash.phash(im, 8).hash).ravel(), np.array(imagehash.dhash(im, 8).hash).ravel()

if __name__ == "__main__":
    rows = load(); par = {}
    def find(x):
        while par.setdefault(x, x) != x: par[x] = par[par[x]]; x = par[x]
        return x
    def union(a, b): par[find(a)] = find(b)
    ok = [r for r in rows if r["image_ok"]]
    seen = {}
    for r in ok:  # exact
        r["drop_reason"] = None
        if r["sha256"] in seen: r["drop_reason"] = "exact_duplicate"; r["dup_of"] = seen[r["sha256"]]; union(r["document_id"], seen[r["sha256"]])
        else: seen[r["sha256"]] = r["document_id"]
    real = [r for r in ok if not r["synthetic"] and r["drop_reason"] is None]
    H = [hashes(r) for r in real]; P = np.stack([h[0] for h in H]); D = np.stack([h[1] for h in H])
    near = 0
    for i in range(len(real)):
        dp = (P[i + 1:] != P[i]).sum(1); dd = (D[i + 1:] != D[i]).sum(1)
        for j in np.where((dp <= 4) | ((dp <= 6) & (dd <= 8)))[0] + i + 1:  # audit (rvl/receipt leak check) showed AND-of-both missed same-merchant receipts
            union(real[i]["document_id"], real[j]["document_id"]); near += 1
    byhint = {}
    for r in real:
        if r["group_hint"] and len(r["group_hint"]) > 6: union(r["document_id"], byhint.setdefault((r["source_dataset"], r["group_hint"]), r["document_id"]))
    cross = collections.Counter()
    for r in ok: r["group_id"] = find(r["document_id"])
    members = collections.defaultdict(list)
    for r in ok: members[r["group_id"]].append(r)
    for g, ms in members.items():
        if len({m["source_dataset"] for m in ms}) > 1: cross[tuple(sorted({m["source_dataset"] for m in ms}))] += 1
    for r in rows:
        if not r["image_ok"]: r.update(drop_reason="corrupt_image", group_id=r["document_id"])
    with open(META / "manifest_dedup.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"near_duplicate_pairs": near, "exact_duplicates_dropped": sum(r.get("drop_reason") == "exact_duplicate" for r in rows),
             "corrupt_dropped": sum(r.get("drop_reason") == "corrupt_image" for r in rows),
             "multi_member_groups": sum(len(m) > 1 for m in members.values()), "largest_group": max(len(m) for m in members.values()),
             "cross_source_overlap_groups": {"+".join(k): v for k, v in cross.items()}}
    dump(META / "dedup_stats.json", stats); print(stats)
