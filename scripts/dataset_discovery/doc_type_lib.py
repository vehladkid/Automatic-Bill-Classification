"""Shared helpers for the document-type dataset tools (schema loading, hashing, deterministic grouped split rules). No heavy dependencies."""
import json, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; DS = ROOT / "datasets"; META = DS / "metadata"; CONFIG = ROOT / "config"
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def load_schema(version="v2"): return json.loads((CONFIG / f"doc_type_schema_{version}.json").read_text())
def family_of(schema): return {t: f for f, ts in schema["axes"]["A_document_type"]["families"].items() for t in ts}
def bucket(key, mod=100): return int(hashlib.sha1(str(key).encode()).hexdigest()[:8], 16) % mod
def synthetic_split(group_id, layout):
    """generated documents: layout D (unseen-layout check) is test-only; otherwise the issuer group decides 70/15/15 so one issuer never spans splits"""
    if layout == "D": return "test"
    b = bucket(group_id); return "train" if b < 70 else ("val" if b < 85 else "test")
def template_split(template_id, all_templates):
    """FATURA: whole templates go to one split (35/7/8 of 50 by stable hash order)"""
    order = sorted(all_templates, key=lambda t: hashlib.sha1(f"fatura-template|{t}".encode()).hexdigest()); n = len(order); i = order.index(template_id)
    return "train" if i < round(n * 0.7) else ("val" if i < round(n * 0.84) else "test")
def read_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f]
def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
