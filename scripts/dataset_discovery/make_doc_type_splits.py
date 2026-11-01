"""Write the per-split manifests from the unified document-type manifest. REAL and SYNTHETIC are kept in separate files on purpose - their metrics must never be mixed.
  datasets/metadata/doc_type_v2_splits/{real,synthetic}_{train,val,test}.jsonl     Usage: make_doc_type_splits.py [--manifest datasets/metadata/doc_type_manifest_v2.jsonl]"""
import sys, argparse, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import META, ROOT, read_jsonl, write_jsonl
def write_split_files(rows, out=None):
    out = Path(out or META / "doc_type_v2_splits"); out.mkdir(parents=True, exist_ok=True); n = collections.Counter()
    for kind in ("real", "synthetic"):
        for sp in ("train", "val", "test"):
            sel = [r for r in rows if (r["synthetic"] == (kind == "synthetic")) and r["split"] == sp]; write_jsonl(out / f"{kind}_{sp}.jsonl", sel); n[f"{kind}_{sp}"] = len(sel)
    return dict(n)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default=str(META / "doc_type_manifest_v2.jsonl")); a = ap.parse_args(); print(write_split_files(read_jsonl(a.manifest)))
