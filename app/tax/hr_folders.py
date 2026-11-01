"""On-disk HR review folders, rebuilt from the database after every change (hr.db stays the source of truth; these are browsable symlink views):
  <outputs>/hr_review/processed/<doc_id>__<name>                 every processed bill (link to the filed original in classified/ or unclassified/)
  <outputs>/hr_review/pending_approval/<employee>/<claim_id>__*  submitted claims waiting for HR  (+ <claim_id>.json with the claim and its documents)
  <outputs>/hr_review/approved/  rejected/  needs_documents/  draft/   same layout for the other claim states
Only files this module creates inside hr_review/ are ever removed."""
import json, shutil, os
from pathlib import Path
from . import store as ST
STATE_DIR = {"submitted": "pending_approval", "approved": "approved", "rejected": "rejected", "needs_documents": "needs_documents", "draft": "draft"}

def out_dir(): return ST.db_path().parent.parent

def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try: os.symlink(os.path.relpath(src, dst.parent), dst)
    except FileExistsError: pass

def sync():
    out = out_dir(); root = out / "hr_review"; res = out / "results"
    if root.exists(): shutil.rmtree(root)                       # a pure view: rebuilt from records + hr.db
    root.mkdir(parents=True, exist_ok=True); recs = {}
    for name in ["processed", *STATE_DIR.values()]: (root / name).mkdir(exist_ok=True)      # all folders exist even while empty
    for f in sorted(res.glob("*/record.json")) if res.exists() else []:
        try: r = json.loads(f.read_text())
        except Exception: continue
        recs[r["doc_id"]] = r
        if r.get("saved_path") and (out / r["saved_path"]).exists(): _link(out / r["saved_path"], root / "processed" / Path(r["saved_path"]).name)
    for c in ST.list_claims():
        d = root / STATE_DIR.get(c["hr_status"], c["hr_status"]) / c["employee_id"]; d.mkdir(parents=True, exist_ok=True)
        docs = [dict(doc_id=i, category=(recs.get(i) or {}).get("category"), file=(recs.get(i) or {}).get("saved_path")) for i in c["docs"]]
        (d / f"{c['id']}.json").write_text(json.dumps(dict(claim=c, documents=docs), indent=1, ensure_ascii=False, default=str))
        for x in docs:
            if x["file"] and (out / x["file"]).exists(): _link(out / x["file"], d / f"{c['id']}__{Path(x['file']).name}")
    return {p.name: sum(1 for _ in p.rglob("*")) for p in root.iterdir()}
