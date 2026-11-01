"""Verified, resumable downloader with provenance. Official sources only (Hugging Face Hub, Zenodo). Nothing is downloaded unless you name it; sizes/licenses are printed first (--dry-run).
  hf    : .venv/bin/python scripts/dataset_discovery/fetch_verified.py hf microsoft/dit-base-finetuned-rvlcdip --kind models --dest models/pretrained/dit-base-finetuned-rvlcdip --files config.json preprocessor_config.json model.safetensors
          .venv/bin/python scripts/dataset_discovery/fetch_verified.py hf AgamiAI/Indian-Bank-Statements --dest datasets/public/agami_bank_statements --glob "*.pdf" --limit 100
  zenodo: .venv/bin/python scripts/dataset_discovery/fetch_verified.py zenodo 8261508 --dest datasets/public/fatura        (md5 from the record)
Integrity: HF LFS files are checked against the sha256 the Hub reports; Zenodo files against the record's md5; non-LFS HF files against size only (no checksum published) - the log says which.
Provenance (source URL, repo revision, licence, size, checksum, retrieval time) is appended to datasets/metadata/provenance_external.jsonl. IPv4 forced (no IPv6 route on this machine). Interrupted downloads resume."""
import sys, os, json, hashlib, argparse, time, socket, fnmatch, datetime as dt
from pathlib import Path
import requests, urllib3.util.connection as uc
uc.allowed_gai_family = lambda: socket.AF_INET
ROOT = Path(__file__).resolve().parents[2]; PROV = ROOT / "datasets/metadata/provenance_external.jsonl"
def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def download(url, dest, size, retries=40):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True); part = dest.with_suffix(dest.suffix + ".part")
    for i in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if size and have == size: break
        try:
            r = requests.get(url, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=(20, 60), allow_redirects=True)
            if r.status_code not in (200, 206): raise RuntimeError(f"HTTP {r.status_code}")
            if have and r.status_code == 200: have = 0; part.unlink(missing_ok=True)
            with open(part, "ab" if have else "wb") as f:
                for chunk in r.iter_content(1 << 20): f.write(chunk)
        except Exception as e:
            print(f"   interrupted ({type(e).__name__}) at {part.stat().st_size if part.exists() else 0}/{size} - resuming", flush=True); time.sleep(3)
    if size and part.stat().st_size != size: raise SystemExit(f"incomplete download of {dest.name}: {part.stat().st_size}/{size}")
    part.replace(dest)
APPROVAL = None  # set from approval_gate for dataset downloads; stored with every provenance record
def log(rec):
    if APPROVAL: rec = dict(rec, approval=APPROVAL)
    PROV.parent.mkdir(parents=True, exist_ok=True)
    with open(PROV, "a") as f: f.write(json.dumps(dict(rec, retrieved=dt.datetime.now().isoformat(timespec="seconds"))) + "\n")
def hf(a):
    j = requests.get(f"https://huggingface.co/api/{a.kind}/{a.repo}", params={"blobs": "true"}, timeout=30).json()
    lic = (j.get("cardData") or {}).get("license"); sib = j["siblings"]
    pick = [s for s in sib if (a.files and s["rfilename"] in a.files) or (a.glob and fnmatch.fnmatch(s["rfilename"], a.glob))]
    if a.glob: pick = sorted(pick, key=lambda s: s["rfilename"])[: a.limit or None]
    if a.glob and a.with_sidecar_json: pick += [s for s in sib if s["rfilename"].endswith(".json") and any(s["rfilename"].rsplit(".", 1)[0] == p["rfilename"].rsplit(".", 1)[0] for p in pick)]
    tot = sum(s.get("size") or 0 for s in pick); print(f"{a.repo} rev={j['sha'][:10]} license={lic} gated={j.get('gated')} | {len(pick)} files, {tot / 1e6:.1f} MB"); free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    print(f"free disk {free / 1e9:.1f} GB"); assert tot < free * 0.6, "refusing: would use more than 60% of the free disk"
    if a.dry_run: [print("  ", s["rfilename"], s.get("size")) for s in pick[:20]]; return
    for s in pick:
        dest = Path(a.dest) / s["rfilename"]; sha = (s.get("lfs") or {}).get("sha256")
        if dest.exists() and (not s.get("size") or dest.stat().st_size == s["size"]) and (not sha or digest(dest, "sha256") == sha): print("ok (already verified)", s["rfilename"]); continue
        print("get", s["rfilename"], s.get("size"), flush=True); download(f"https://huggingface.co/{'datasets/' if a.kind == 'datasets' else ''}{a.repo}/resolve/{j['sha']}/{s['rfilename']}", dest, s.get("size"))
        ok = "sha256 verified" if sha and digest(dest, "sha256") == sha else ("SHA256 MISMATCH" if sha else "size-only (no checksum published)")
        if ok == "SHA256 MISMATCH": dest.unlink(); raise SystemExit(f"checksum mismatch for {s['rfilename']}")
        print("  ", ok); log(dict(source="huggingface", url=f"https://huggingface.co/{a.kind}/{a.repo}", revision=j["sha"], file=str(dest.relative_to(ROOT)) if dest.is_absolute() and ROOT in dest.parents else str(dest), size=s.get("size"), sha256=sha or digest(dest, "sha256"), integrity=ok, license=lic))
def zenodo(a):
    j = requests.get(f"https://zenodo.org/api/records/{a.record}", timeout=30).json(); lic = (j["metadata"].get("license") or {}).get("id"); print(j["metadata"]["title"], "license", lic, "files", [(f["key"], f["size"]) for f in j["files"]])
    if a.dry_run: return
    for f in j["files"]:
        dest = Path(a.dest) / f["key"]; md5 = f["checksum"].split(":", 1)[1]
        if not (dest.exists() and digest(dest, "md5") == md5): download(f["links"]["self"], dest, f["size"])
        if digest(dest, "md5") != md5: dest.unlink(); raise SystemExit("md5 mismatch")
        print("md5 verified", f["key"]); log(dict(source="zenodo", url=j["links"].get("self_html") or j["links"].get("html") or f"https://zenodo.org/records/{a.record}", file=str(dest), size=f["size"], md5=md5, integrity="md5 verified", license=lic))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)  # every dataset acquisition needs --source-id and passes approval_gate; models are exempt
    h = sp.add_parser("hf"); h.add_argument("repo"); h.add_argument("--kind", default="datasets"); h.add_argument("--dest", required=True); h.add_argument("--files", nargs="*"); h.add_argument("--glob"); h.add_argument("--limit", type=int); h.add_argument("--with-sidecar-json", action="store_true"); h.add_argument("--dry-run", action="store_true"); h.add_argument("--source-id")
    z = sp.add_parser("zenodo"); z.add_argument("record"); z.add_argument("--dest", required=True); z.add_argument("--dry-run", action="store_true"); z.add_argument("--source-id")
    a = ap.parse_args()
    if not (a.cmd == "hf" and a.kind == "models"):
        from approval_gate import require_approved
        if not a.source_id: raise SystemExit("REFUSED: --source-id (licence-register id) is required for dataset downloads")
        APPROVAL = require_approved(a.source_id); globals()["APPROVAL"] = APPROVAL; print("approval:", json.dumps(APPROVAL))
    hf(a) if a.cmd == "hf" else zenodo(a)
