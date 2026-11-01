"""Read-only metadata probe for Hugging Face datasets/models. Facts come from the Hub API (card license, gating, file list with sizes, tags) - nothing is downloaded except JSON metadata.
  .venv/bin/python scripts/dataset_discovery/hf_probe.py search "rent receipt" [--kind datasets|models] [--limit 8]
  .venv/bin/python scripts/dataset_discovery/hf_probe.py probe aharley/rvl_cdip nielsr/funsd [--kind datasets] [--out datasets/metadata/external_catalog.json]
  .venv/bin/python scripts/dataset_discovery/hf_probe.py probe 8261508 5113009 --kind zenodo --out datasets/metadata/external_catalog.json      (Zenodo record ids)
Every probed record is stamped with the retrieval date so the catalog is auditable. IPv4 is forced (this machine has no IPv6 route)."""
import sys, json, argparse, time, socket, datetime as dt
import requests, urllib3.util.connection as uc
uc.allowed_gai_family = lambda: socket.AF_INET
H = "https://huggingface.co/api"
def get(url, **p):
    for i in range(4):
        try:
            r = requests.get(url, params=p, timeout=30)
            if r.status_code == 200: return r.json()
            if r.status_code in (401, 403, 404): return {"_http": r.status_code}
        except Exception as e: err = e
        time.sleep(2 * (i + 1))
    return {"_error": str(err) if 'err' in dir() else "failed"}

def search(q, kind="datasets", limit=8):
    j = get(f"{H}/{kind}", search=q, limit=limit, sort="downloads", direction=-1)
    return [dict(id=x["id"], downloads=x.get("downloads"), likes=x.get("likes"), license=next((t.split(":", 1)[1] for t in x.get("tags", []) if t.startswith("license:")), None), gated=x.get("gated")) for x in j] if isinstance(j, list) else j

def probe_zenodo(rid):
    j = get(f"https://zenodo.org/api/records/{rid}")
    if "_http" in j or "_error" in j: return dict(id=str(rid), kind="zenodo", retrieved=dt.date.today().isoformat(), **j)
    m = j["metadata"]; files = j.get("files", [])
    return dict(id=str(rid), kind="zenodo", retrieved=dt.date.today().isoformat(), url=j.get("links", {}).get("self_html") or f"https://zenodo.org/records/{rid}", title=m.get("title"), license=(m.get("license") or {}).get("id"), access=m.get("access_right"),
                publication_date=m.get("publication_date"), size_bytes=sum(f["size"] for f in files), size_gb=round(sum(f["size"] for f in files) / 1e9, 3), n_files=len(files), files=[dict(key=f["key"], size=f["size"], checksum=f.get("checksum")) for f in files][:10], doi=j.get("doi"))

def probe(rid, kind="datasets"):
    if kind == "zenodo": return probe_zenodo(rid)
    j = get(f"{H}/{kind}/{rid}", blobs="true")
    if "_http" in j or "_error" in j: return dict(id=rid, kind=kind, retrieved=dt.date.today().isoformat(), **j)
    sib = j.get("siblings", []); size = sum((s.get("size") or 0) for s in sib); card = j.get("cardData") or {}
    tags = j.get("tags", [])
    exts = {}
    for s in sib:
        e = s["rfilename"].rsplit(".", 1)[-1].lower() if "." in s["rfilename"] else "(none)"; exts[e] = exts.get(e, 0) + 1
    out = dict(id=rid, kind=kind, retrieved=dt.date.today().isoformat(), url=f"https://huggingface.co/{kind}/{rid}", license=card.get("license") or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None),
               license_name=card.get("license_name"), gated=j.get("gated"), private=j.get("private"), downloads=j.get("downloads"), last_modified=j.get("lastModified"), sha=j.get("sha"),
               size_bytes=size, size_gb=round(size / 1e9, 3), n_files=len(sib), file_ext_counts=exts, tags=[t for t in tags if not t.startswith(("license:", "region:"))][:25],
               task_categories=card.get("task_categories"), language=card.get("language"), pipeline_tag=j.get("pipeline_tag"), library=j.get("library_name"),
               dataset_info=card.get("dataset_info"), configs=card.get("configs"))
    if kind == "models":
        out["safetensors_params"] = (j.get("safetensors") or {}).get("total"); out["config_arch"] = (j.get("config") or {}).get("architectures"); out["model_files"] = {s["rfilename"]: s.get("size") for s in sib if s["rfilename"].endswith((".safetensors", ".bin", ".onnx", ".pt")) }
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["search", "probe"]); ap.add_argument("items", nargs="+"); ap.add_argument("--kind", default="datasets"); ap.add_argument("--limit", type=int, default=8); ap.add_argument("--out"); a = ap.parse_args()
    if a.cmd == "search":
        for q in a.items:
            print(f"## {a.kind} search: {q}"); [print(f"  {r['id']:55s} dl={r['downloads']!s:>8s} lic={r['license']!s:18s} gated={r['gated']}") for r in (search(q, a.kind, a.limit) if True else [])]
    else:
        res = [probe(i, a.kind) for i in a.items]
        if a.out:
            try: old = json.load(open(a.out))
            except Exception: old = []
            byid = {(r["kind"], r["id"]): r for r in old}; byid.update({(r["kind"], r["id"]): r for r in res}); json.dump(list(byid.values()), open(a.out, "w"), indent=1)
        print(json.dumps(res, indent=1, default=str)[:6000])
