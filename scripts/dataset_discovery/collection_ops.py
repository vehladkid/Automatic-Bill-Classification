"""One entry point for real-document collection operations. Never prints names/numbers/file names from documents: records are referred to by opaque doc_id or batch line number.
  collection_ops.py status  [--manifest M]                  validate + per-class progress + review/redaction queues + why each record is blocked (exit 1 on validation errors)
  collection_ops.py import  BATCH.jsonl --staging DIR [--apply] [--no-ocr-check]   DRY-RUN by default; checks licence/permission gate, redaction roles, PII (metadata + local OCR of the redacted copy), duplicates (sha256 + pHash, against the collection AND the real corpus), then appends only with --apply
  collection_ops.py assign-splits [--apply]                 deterministic issuer-group split assignment (dry-run by default) that respects every gate
Batch line = JSON: file (relative to staging), doc_type, expense_category, tax_evidence, source_id, permission_basis, permission_ref, issuer_group_id, redactor_id, verifier_id (distinct opaque ids; both required for 'redacted_verified'), language, country, ambiguity_flags, reviewer_notes.
No network. OCR (if enabled) is the project's local RapidOCR engine."""
import sys, re, json, argparse, hashlib, secrets, collections, shutil, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collection_manifest as CM
ROOT = CM.ROOT; DIR = CM.DIR; DEFAULT = DIR / "collection_manifest.jsonl"; REAL_CORPUS = ROOT / "datasets/metadata/doc_type_manifest_v2_1.jsonl"; APPROVED = ROOT / "config/approved_sources_v2.json"
EXPLAIN = {"missing_fields": "required field(s) absent", "unknown_fields": "field not in the schema (free-form fields are refused so personal data cannot hide in them)", "bad_doc_id": "doc_id must look like COL-<8hex>-<4hex>", "duplicate_doc_id": "doc_id used twice",
    "bad_doc_type": "document type not in the taxonomy", "bad_expense_category": "expense category not one of the 16 Task A classes (or null)", "bad_tax_evidence": "tax evidence label not allowed", "synthetic_or_non_real_in_collection": "collection manifest accepts real documents only",
    "bad_permission_basis": "permission basis not recognised", "missing_permission_ref": "no consent/permission record id: cannot be used", "bad_issuer_group_id": "issuer group id must be IG-<10hex> (opaque HMAC, never a name)", "bad_split": "split value invalid", "bad_annotation_status": "annotation status invalid",
    "bad_sensitive_review_status": "sensitive-review status invalid", "bad_ambiguity_flags": "ambiguity flag not allowed", "eligibility_status_invalid": "tax eligibility may never be asserted", "bad_country": "country must be ISO alpha-2", "bad_date": "date must be YYYY-MM-DD",
    "gate_split_without_verified_redaction": "assigned to a split but redaction is not verified", "gate_split_without_redacted_copy": "assigned to a split but there is no redacted copy", "gate_eval_split_without_double_annotation": "val/test documents need double annotation",
    "gate_specimen_in_eval_split": "specimens/templates are not completed real documents and may not be in val/test", "duplicate_redacted_copy": "two records point to the same redacted file", "redacted_copy_without_verified_status": "redacted copy recorded before verification", "redacted_copy_must_be_relative": "redacted copy path must be relative",
    "issuer_group_spans_splits": "one issuer/template family appears in more than one split (leakage)", "duplicate_redacted_sha256": "byte-identical redacted copy already in the collection", "bad_redacted_sha256": "sha256 malformed", "bad_redacted_phash": "perceptual hash malformed"}
def explain(code): return EXPLAIN.get(code, "personal-data pattern found in a text field (" + code.replace("personal_data_pattern_", "") + "): remove it" if code.startswith("personal_data_pattern_") else code)
def load(path):
    p = Path(path); return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []
def blockers(r):
    """why a record cannot (yet) feed training/evaluation"""
    out = []
    if r["sensitive_review_status"] != "redacted_verified": out.append(("redaction", {"pending": "redaction not started", "redaction_required": "redaction/verification still to do", "rejected_unredactable": "rejected: cannot be redacted safely"}[r["sensitive_review_status"]]))
    if r["annotation_status"] == "pending": out.append(("annotation", "not annotated"))
    if r["split"] == "unassigned": out.append(("split", "no split assigned yet"))
    if r["split"] in ("val", "test") and r["annotation_status"] not in ("double_annotated", "adjudicated"): out.append(("annotation", "val/test needs double annotation"))
    amb = [f for f in r["ambiguity_flags"] if f.endswith("_ambiguous")]
    if amb: out.append(("label_review", "ambiguity flagged: " + ", ".join(amb)))
    if "specimen_or_template" in r["ambiguity_flags"]: out.append(("specimen", "specimen/template: training only, never val/test"))
    return out
def progress(rows):
    out = []; mn = CM.TARGETS["per_class_minimum"]
    for p in CM.TARGETS["priority_order"]:
        t = p["doc_type"]; rs = [r for r in rows if r["doc_type"] == t]; use = [r for r in rs if r["split"] in ("train", "val", "test") and r["sensitive_review_status"] == "redacted_verified"]
        row = dict(rank=p["rank"], doc_type=t, collected=len(rs), usable_in_splits=len(use), unique_issuers_all=len({r["issuer_group_id"] for r in rs}), unique_issuers_usable=len({r["issuer_group_id"] for r in use}), annotation=dict(collections.Counter(r["annotation_status"] for r in rs)),
                   splits={s: sum(r["split"] == s for r in use) for s in ("train", "val", "test")}, issuers_per_split={s: len({r["issuer_group_id"] for r in use if r["split"] == s}) for s in ("train", "val", "test")}, targets=dict(train=mn["train"], val=mn["val"], test=mn["test"], issuers=mn["min_distinct_issuers_per_split"]))
        row["ready_for_per_class_metric"] = row["splits"]["val"] >= mn["val"] and row["splits"]["test"] >= mn["test"] and row["issuers_per_split"]["val"] >= mn["min_distinct_issuers_per_split"]["val"] and row["issuers_per_split"]["test"] >= mn["min_distinct_issuers_per_split"]["test"]; out.append(row)
    return out
def queues(rows):
    return dict(label_review=[r["doc_id"] for r in rows if any(c == "label_review" for c, _ in blockers(r)) or (r["split"] in ("val", "test") and r["annotation_status"] == "single_annotated")], redaction_verification=[r["doc_id"] for r in rows if r["sensitive_review_status"] in ("pending", "redaction_required")],
                annotation_needed=[r["doc_id"] for r in rows if r["sensitive_review_status"] == "redacted_verified" and r["annotation_status"] == "pending"], rejected=[r["doc_id"] for r in rows if r["sensitive_review_status"] == "rejected_unredactable"])
def status(path):
    rows = load(path); err = CM.validate(rows); prog = progress(rows); q = queues(rows); bl = collections.Counter(c for r in rows for c, _ in blockers(r))
    rep = dict(manifest=str(Path(path).relative_to(ROOT)) if str(path).startswith(str(ROOT)) else "external", n_records=len(rows), passed=not err, errors={k: dict(count=len(v), meaning=explain(k), examples=[str(x).split(":")[0] for x in v[:3]]) for k, v in err.items()}, blocked_reason_counts=dict(bl), progress=prog, queues={k: dict(count=len(v), doc_ids=v[:50]) for k, v in q.items()})
    L = [f"Collection manifest: {len(rows)} records; validation {'PASSED' if not err else 'FAILED'}"] + [f"  ERROR {k} x{len(v)}: {explain(k)}" for k, v in err.items()]
    L.append(f"{'#':>2} {'class':27s} {'coll':>4} {'usable':>6} {'issuers':>7} {'train':>5} {'val':>4} {'test':>4}  ready")
    for r in prog: L.append(f"{r['rank']:>2} {r['doc_type']:27s} {r['collected']:>4} {r['usable_in_splits']:>6} {r['unique_issuers_usable']:>7} {r['splits']['train']:>5} {r['splits']['val']:>4} {r['splits']['test']:>4}  {'YES' if r['ready_for_per_class_metric'] else 'no (needs >=30 val & >=30 test from >=6 issuers each)'}")
    L.append("Queues: " + ", ".join(f"{k}={len(v)}" for k, v in q.items())); L.append("Blocked reasons: " + (", ".join(f"{k}={v}" for k, v in bl.items()) or "none"))
    return rep, "\n".join(L), err
def ocr_text(path):
    sys.path[:0] = [str(ROOT / p) for p in ("scripts", "scripts/taskA", "scripts/taskB")]
    from run_ocr_gpu import get_engine
    out, _ = get_engine()(str(path)); return "\n".join(x[1] for x in (out or []))
def phash64(path):
    import imagehash
    from PIL import Image
    return str(imagehash.phash(Image.open(path).convert("L"), hash_size=8))
def ham(a, b): return bin(int(a, 16) ^ int(b, 16)).count("1")
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
CACHE = ROOT / "datasets/interim/collection_corpus_phash.json"
def corpus_phashes():
    """pHash (16 hex) of the first page of every REAL corpus document; cached because computing it takes ~2 min"""
    if CACHE.exists(): return json.loads(CACHE.read_text())
    from concurrent.futures import ProcessPoolExecutor
    paths = [str(ROOT / "datasets" / json.loads(l)["file_path"]) for l in open(REAL_CORPUS) if not json.loads(l)["synthetic"] and not json.loads(l)["file_path"].endswith(".pdf")]
    with ProcessPoolExecutor(6) as ex: hs = list(ex.map(_safe_phash, paths, chunksize=64))
    out = [h for h in hs if h]; CACHE.write_text(json.dumps(out)); return out
def _safe_phash(p):
    try: return phash64(p)
    except Exception: return None
def corpus_hashes():
    sh, ph = set(), []
    if REAL_CORPUS.exists():
        for l in open(REAL_CORPUS):
            r = json.loads(l)
            if not r["synthetic"]: sh.update(r["page_sha256"])
    return sh
def import_batch(batch, staging, manifest_path=DEFAULT, apply=False, ocr=ocr_text, corpus_sha=None, corpus_ph=None):
    from PIL import Image
    rows = load(manifest_path); staging = Path(staging); cfg = json.loads(APPROVED.read_text()); approved = set(cfg["tier1_usable"]) | set(cfg["tier2_restricted_known_terms"]); excl = cfg["excluded_until_cleared"]
    csha = corpus_sha if corpus_sha is not None else corpus_hashes(); cph = corpus_ph if corpus_ph is not None else corpus_phashes(); have_sha = {r.get("redacted_sha256") for r in rows if r.get("redacted_sha256")}; have_ph = [(r["redacted_phash"], r["issuer_group_id"]) for r in rows if r.get("redacted_phash")]
    accepted, rejected, new, seen_sha = [], [], [], set(); group_split = {r["issuer_group_id"]: r["split"] for r in rows if r["split"] != "unassigned"}
    for n, b in enumerate(batch, 1):
        why = []; f = b.get("file")
        if not f or str(f).startswith("/") or ".." in str(f): why.append("file must be a relative path inside the staging directory")
        elif not (staging / f).is_file(): why.append("staged file not found")
        if b.get("permission_basis") == "public_licence_verified":
            sid = b.get("source_id")
            if sid in excl: why.append(f"source '{sid}' is excluded until cleared: {excl[sid][:90]}")
            elif sid not in approved: why.append(f"source '{sid}' is not in the approved-source configuration; verify the licence and add it first")
        if not b.get("permission_ref"): why.append("no permission_ref (consent/permission record id)")
        red, ver = b.get("redactor_id"), b.get("verifier_id"); verified = bool(red and ver and red != ver)
        if red and ver and red == ver: why.append("redactor and verifier must be different people")
        rec = dict(doc_id="COL-" + secrets.token_hex(4) + "-" + secrets.token_hex(2), doc_type=b.get("doc_type"), expense_category=b.get("expense_category"), tax_evidence=b.get("tax_evidence", []), real=True, source_id=b.get("source_id", ""), permission_basis=b.get("permission_basis"), permission_ref=b.get("permission_ref", ""),
                   issuer_group_id=b.get("issuer_group_id"), split="unassigned", annotation_status="pending", sensitive_review_status="redacted_verified" if verified else "redaction_required", redacted_copy=f if verified else None, ambiguity_flags=b.get("ambiguity_flags", []),
                   reviewer_notes=b.get("reviewer_notes", ""), eligibility_status="pending_verification" if b.get("tax_evidence") else "not_applicable", country=b.get("country", "IN"), language=b.get("language", "en"), created=datetime.date.today().isoformat())
        if not why:
            p = staging / f
            try:
                im = Image.open(p); im.verify()
            except Exception: why.append("staged file is not a decodable image")
        if not why and verified:
            s = sha(p); ph = phash64(p)
            if s in csha: why.append("byte-identical to a document already in the real corpus")
            elif s in have_sha or s in seen_sha: why.append("byte-identical to a document already in the collection/batch")
            else:
                near_c = [x for x in cph if ham(ph, x) <= 4] + [g for h, g in have_ph if ham(ph, h) <= 4 and g != rec["issuer_group_id"]]
                if near_c: why.append("near-duplicate (pHash<=4) of a document in the corpus/collection under a different issuer group")
            rec["redacted_sha256"], rec["redacted_phash"] = s, ph; seen_sha.add(s)
            if not why and ocr is not None:
                txt = ocr(p); hits = [k for k, rx in CM.PII.items() if rx.search(txt)]
                if hits: why.append("local OCR of the redacted copy still shows personal-data patterns (" + ", ".join(hits) + "): redact again")
        if not why:
            e = CM.validate([rec])
            for k in e: why.append(explain(k))
        g = rec["issuer_group_id"]
        if not why and g in group_split: rec["_inherits_split_note"] = group_split[g]
        if why: rejected.append(dict(line=n, reasons=why))
        else: rec.pop("_inherits_split_note", None); new.append(rec); accepted.append(dict(line=n, doc_id=rec["doc_id"], status=rec["sensitive_review_status"]))
    if apply and new:
        mp = Path(manifest_path)
        if mp.exists(): shutil.copy(mp, str(mp) + ".bak")
        with open(mp, "a") as w:
            for r in new: w.write(json.dumps(r, sort_keys=True) + "\n")
    return dict(mode="APPLIED" if apply else "DRY-RUN (nothing written)", proposed=len(batch), accepted=len([a for a in accepted if "doc_id" in a]), rejected=len(rejected), accepted_detail=[a for a in accepted if "doc_id" in a], rejected_detail=rejected), new
def assign_splits(rows, apply=False):
    by = collections.defaultdict(list)
    for r in rows: by[r["issuer_group_id"]].append(r)
    changes, waits = [], collections.Counter()
    for g, rs in by.items():
        done = {r["split"] for r in rs if r["split"] != "unassigned"}
        if len(done) > 1: waits["group_already_spans_splits"] += 1; continue
        todo = [r for r in rs if r["split"] == "unassigned"]
        if not todo: continue
        if any(r["sensitive_review_status"] != "redacted_verified" for r in todo): waits["waiting_for_verified_redaction"] += 1; continue
        if any(r["annotation_status"] == "pending" for r in todo): waits["waiting_for_annotation"] += 1; continue
        if done: target = next(iter(done))
        elif any("specimen_or_template" in r["ambiguity_flags"] for r in rs): target = "train"
        else:
            b = int(hashlib.sha1(("collection-split|" + g).encode()).hexdigest()[:8], 16) % 100; target = "train" if b < 70 else ("val" if b < 85 else "test")
        if target in ("val", "test") and any(r["annotation_status"] not in ("double_annotated", "adjudicated") for r in todo): waits["waiting_for_double_annotation_before_eval_split"] += 1; continue
        for r in todo:
            changes.append((r["doc_id"], target))
            if apply: r["split"] = target
    return changes, dict(waits)
def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd"); a1 = sp.add_parser("status"); a1.add_argument("--manifest", default=str(DEFAULT)); a1.add_argument("--report")
    a2 = sp.add_parser("import"); a2.add_argument("batch"); a2.add_argument("--staging", required=True); a2.add_argument("--manifest", default=str(DEFAULT)); a2.add_argument("--apply", action="store_true"); a2.add_argument("--no-ocr-check", action="store_true")
    a3 = sp.add_parser("assign-splits"); a3.add_argument("--manifest", default=str(DEFAULT)); a3.add_argument("--apply", action="store_true"); a = ap.parse_args(); cmd = a.cmd or "status"
    if cmd == "status":
        rep, txt, err = status(getattr(a, "manifest", DEFAULT)); print(txt)
        if getattr(a, "report", None): Path(a.report).write_text(json.dumps(rep, indent=1))
        sys.exit(1 if err else 0)
    if cmd == "import":
        res, new = import_batch(load(a.batch), a.staging, a.manifest, a.apply, None if a.no_ocr_check else ocr_text); print(json.dumps(res, indent=1)); sys.exit(0 if not res["rejected"] else 2)
    if cmd == "assign-splits":
        rows = load(a.manifest); ch, waits = assign_splits(rows, a.apply)
        if a.apply and ch:
            e = CM.validate(rows)
            if e: print("refusing to write: validation errors", list(e)); sys.exit(1)
            shutil.copy(a.manifest, a.manifest + ".bak"); Path(a.manifest).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        print(json.dumps(dict(mode="APPLIED" if a.apply else "DRY-RUN", would_assign=len(ch), waiting=waits), indent=1))
if __name__ == "__main__": main()
