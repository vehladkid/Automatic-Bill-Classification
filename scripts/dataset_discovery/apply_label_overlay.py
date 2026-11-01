"""Reversible label overlay. The manifests are never edited: an overlay row {change_id, doc_id, field, old, new} produces a derived row list with the new value, the preserved original (`<field>_original`) and `label_overlay_id`; revert() restores the original exactly. apply refuses any row whose current value differs from the recorded `old` (stale overlay).
CLI: apply_label_overlay.py [--overlay F] [--manifest M] [--out F]  writes only the CHANGED rows' effective labels (default label_review/effective_labels.jsonl); never writes the manifest."""
import sys, json, copy, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import ROOT, META, read_jsonl
OUT = ROOT / "reports/doc_type/licensing_and_collection/label_review"
ALLOWED_FIELDS = {"doc_type"}   # expense_category (Task A) and tax evidence are never overlaid
def apply_overlay(rows, overlay):
    out = [copy.deepcopy(r) for r in rows]; idx = {r["doc_id"]: r for r in out}
    for o in overlay:
        if o["field"] not in ALLOWED_FIELDS: raise ValueError(f"field {o['field']} may not be overlaid")
        r = idx.get(o["doc_id"])
        if r is None: raise KeyError(o["doc_id"])
        if r[o["field"]] != o["old"]: raise ValueError(f"stale overlay for {o['doc_id']}: current {r[o['field']]!r} != recorded {o['old']!r}")
        r[o["field"] + "_original"] = r[o["field"]]; r[o["field"]] = o["new"]; r["label_overlay_id"] = o["change_id"]
    return out
def revert_overlay(rows):
    out = [copy.deepcopy(r) for r in rows]
    for r in out:
        if "label_overlay_id" in r:
            for k in list(r):
                if k.endswith("_original") and k[:-9] in ALLOWED_FIELDS: r[k[:-9]] = r.pop(k)
            r.pop("label_overlay_id")
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--overlay", default=str(OUT / "label_overlay_v1.jsonl")); ap.add_argument("--manifest", default=str(META / "doc_type_manifest_v2_1.jsonl")); ap.add_argument("--out", default=str(OUT / "effective_labels.jsonl")); a = ap.parse_args()
    ov = read_jsonl(a.overlay); rows = read_jsonl(a.manifest); eff = apply_overlay(rows, ov); ids = {o["doc_id"] for o in ov}
    Path(a.out).write_text("".join(json.dumps(dict(doc_id=r["doc_id"], doc_type=r["doc_type"], doc_type_original=r["doc_type_original"], label_overlay_id=r["label_overlay_id"]), sort_keys=True) + "\n" for r in eff if r["doc_id"] in ids)); print(f"{len(ids)} overlay change(s) written to {a.out}; manifest untouched")
if __name__ == "__main__": main()
