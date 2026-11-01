"""Approval gate for acquisition: a dataset may be downloaded/imported only if its licence-register id is approved in config/approved_sources_v2.json (tier1 'usable' or tier2 'usable_with_restrictions' with known terms) AND the register status agrees. Returns the evidence record that is stored in the provenance log.
CLI: approval_gate.py SOURCE_ID   -> prints the decision and why (exit 0 approved, 1 not)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def decision(source_id, cfg_path=ROOT / "config/approved_sources_v2.json", reg_path=ROOT / "datasets/metadata/license_register_v2.json"):
    cfg = json.loads(Path(cfg_path).read_text()); reg = {s["id"]: s for s in json.loads(Path(reg_path).read_text())["sources"]}; e = reg.get(source_id)
    if e is None: return dict(approved=False, source_id=source_id, reason="not in the licence register: add an entry with primary-source evidence first")
    if source_id in cfg["excluded_until_cleared"]: return dict(approved=False, source_id=source_id, reason="excluded: " + cfg["excluded_until_cleared"][source_id])
    tier = 1 if source_id in cfg["tier1_usable"] else (2 if source_id in cfg["tier2_restricted_known_terms"] else None)
    if tier is None: return dict(approved=False, source_id=source_id, reason=f"not listed in {cfg['version']} (register status '{e['status']}')")
    want = "usable" if tier == 1 else "usable_with_restrictions"
    if e["status"] != want: return dict(approved=False, source_id=source_id, reason=f"config and register disagree: register status '{e['status']}', tier {tier} needs '{want}'")
    return dict(approved=True, source_id=source_id, tier=tier, approval_config=cfg["version"], licence=e["license"], evidence_level=e["evidence"]["level"], evidence_url=e["evidence"]["url"], evidence_retrieved=e["evidence"]["retrieved"], restriction=cfg["tier2_restricted_known_terms"].get(source_id))
def require_approved(source_id, **kw):
    d = decision(source_id, **kw)
    if not d["approved"]: raise SystemExit(f"REFUSED: source '{source_id}' is not approved for acquisition - {d['reason']}")
    return d
if __name__ == "__main__":
    d = decision(sys.argv[1]); print(json.dumps(d, indent=1)); sys.exit(0 if d["approved"] else 1)
