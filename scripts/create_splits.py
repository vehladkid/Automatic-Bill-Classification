"""Group-level stratified 80/10/10 split. A group = exact/near-duplicate cluster or same-merchant SROIE family, so no
document (or template family) is ever on both sides. Augmentation is NOT materialised, so it cannot leak either."""
import json, random, collections
from common import *

def split_groups(groups, rng, frac=(0.8, 0.1, 0.1)):
    """groups: {gid: size}. Largest groups first, each to the split with the biggest remaining deficit.
    <3 groups -> all train (cannot be split without leakage or empty val/test)."""
    if len(groups) < 3: return {g: "train" for g in groups}
    names, tot = ["train", "val", "test"], sum(groups.values())
    need = {n: f * tot for n, f in zip(names, frac)}; out = {}
    ids = sorted(groups); rng.shuffle(ids); ids.sort(key=lambda g: -groups[g])
    for g in ids:
        n = max(names, key=lambda n: need[n]); out[g] = n; need[n] -= groups[g]
    return out

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(META / "manifest_dedup.jsonl")]
    ov = json.loads((META / "label_overrides.json").read_text()) if (META / "label_overrides.json").exists() else {}
    for r in rows:  # human-reviewed label fixes (also applied in normalize_labels.py); re-applied here so they can be edited without re-running dedup
        if r["document_id"] in ov:
            c = ov[r["document_id"]]["expense_category"]; r.update(expense_category=c, parent_group=CATEGORIES[c], label_quality="manual_review", label_source="label_overrides.json: " + ov[r["document_id"]]["reason"][:80])
    keep = [r for r in rows if not r.get("drop_reason")]
    strat = collections.defaultdict(lambda: collections.defaultdict(int)); gkey = {}
    for r in keep:
        k = r["expense_category"] if r["expense_category"] != "unlabeled" else f"unl:{r['source_dataset']}"
        k = (k, r["synthetic"])  # real and synthetic stratified separately so real data always reaches val/test when it can
        gkey.setdefault(r["group_id"], k); strat[gkey[r["group_id"]]][r["group_id"]] += 1
    rng = random.Random(SEED); assign = {}
    for k in sorted(strat, key=str): assign.update(split_groups(strat[k], rng))
    for r in keep: r["split"] = assign[r["group_id"]]
    with open(META / "manifest.jsonl", "w") as f:
        for r in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in ("train", "val", "test"):
        SPLITS.mkdir(parents=True, exist_ok=True)
        (SPLITS / f"{s}.txt").write_text("\n".join(r["document_id"] for r in keep if r["split"] == s) + "\n")
    print(collections.Counter(r["split"] for r in keep))
