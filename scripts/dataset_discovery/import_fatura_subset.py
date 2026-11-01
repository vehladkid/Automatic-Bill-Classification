"""Deterministic, template-balanced subset of FATURA (Zenodo 8261508, CC-BY-4.0, md5-verified archive) for the document-type corpus. FATURA = 10,000 SYNTHETIC invoice images, 50 templates x 200 instances.
Takes K instances per template (default 4 -> 200 images) chosen by a stable hash, extracts only those JPEGs, and records archive path + sha256. The TEMPLATE is the leakage group (see build_doc_type_manifest_v2).
Re-runnable: skips files already extracted with a matching hash.  Usage: import_fatura_subset.py [--per-template 4]"""
import sys, json, zipfile, hashlib, re, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; ZIP = ROOT / "datasets/public/fatura/FATURA.zip"; OUT = ROOT / "datasets/public/fatura/subset"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-template", type=int, default=4); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP); by = {}
    for n in z.namelist():
        m = re.fullmatch(r"invoices_dataset_final/images/Template(\d+)_Instance(\d+)\.jpg", n)
        if m: by.setdefault(int(m[1]), []).append((int(m[2]), n))
    rows = []
    for t in sorted(by):
        pick = sorted(by[t], key=lambda x: hashlib.sha256(f"fatura|{t}|{x[0]}".encode()).hexdigest())[: a.per_template]
        for inst, name in sorted(pick):
            data = z.read(name); sha = hashlib.sha256(data).hexdigest(); dest = OUT / f"Template{t}_Instance{inst}.jpg"
            if not (dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha): dest.write_bytes(data)
            rows.append(dict(doc_id=f"fatura__t{t}_i{inst}", template=t, instance=inst, archive_member=name, file=str(dest.relative_to(ROOT / "datasets")), sha256=sha, bytes=len(data)))
    (ROOT / "datasets/public/fatura/subset_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"templates {len(by)}; extracted {len(rows)} images ({sum(r['bytes'] for r in rows) / 1e6:.1f} MB) -> {OUT}")
if __name__ == "__main__": main()
