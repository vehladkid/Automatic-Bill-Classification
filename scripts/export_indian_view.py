"""Browsable view of ALL Indian documents (country == IN), LINKED (relative symlinks, ~0 bytes; originals untouched) into datasets/indian_all/{real,synthetic}/<category>/.
Originals stay where they are. Writes datasets/indian_all/index.csv (file, category, source, license, label_quality, split, ...). Rebuilt each run."""
import json, shutil, csv, os
from common import *
OUT = DS / "indian_all"
if __name__ == "__main__":
    rows = [json.loads(l) for l in open(META / "manifest.jsonl")]
    sel = [r for r in rows if r["country"] == "IN" or (r["synthetic"] and r["source_dataset"] == "synthetic_in")]
    if OUT.exists(): shutil.rmtree(OUT)
    recs = []
    for r in sel:
        kind = "synthetic" if r["synthetic"] else "real"; cat = r["expense_category"]
        d = OUT / kind / (cat if cat != "unlabeled" else "_no_category"); d.mkdir(parents=True, exist_ok=True)
        src = DS / r["file_path"]; name = f"{r['source_dataset']}__{src.name}"; os.symlink(os.path.relpath(src, d), d / name)   # relative link: no duplicate bytes, still browsable
        recs.append({"file": str((d / name).relative_to(OUT)), "kind": kind, "category": cat, "source_dataset": r["source_dataset"], "document_type": r["document_type"],
                     "label_quality": r["label_quality"], "merchant": r.get("merchant") or "", "license": r["license"], "split": r["split"], "original_path": r["file_path"], "document_id": r["document_id"]})
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    import collections
    c = collections.Counter((x["kind"], x["category"]) for x in recs)
    for k in sorted(c): print(k, c[k])
    print("total", len(recs), "->", OUT)
