"""Build document-type manifest v2.1 from v2 (v2 files are left untouched). Changes, all recorded in docs/DOC_TYPE_V2_1_CHANGES.md:
 1. synthgen_v1 split v2.1 = TEMPLATE-held-out: group_id becomes the template (doc_type x layout). Per type: layout D -> test, one rotating layout of A/B/C -> val, the other two -> train. (v2 split by issuer left same-layout pages in train and val/test: 116/161 test pages near-identical to train pages.)
 2. agami split by template (Scanned_Type2 -> val, Digital_Type2 -> test, others train); v2 had train only.
 3. derived tax-evidence candidates corrected by subtype (health insurance -> 80D only, life -> 80C only; EV: no loan-interest candidate on EV documents) and uncertain mappings flagged.
 4. approval gate from config/approved_sources_v1.json + licence register: approved_for_training / approval_tier / approval_reason.
 5. real-document label cross-check by OCR text cues: conflicting cues -> review_status 'flagged_for_review' + queue file. Real splits are NOT changed (protected Task A/B test docs stay test).
Outputs: doc_type_manifest_v2_1.jsonl, doc_type_v2_1_{real,synthetic}.jsonl, doc_type_v2_1_splits/, doc_type_v2_1_approved/ (approved subset + splits), mapping_review_queue_v2_1.jsonl, doc_type_v2_1_stats.json, config/doc_type_schema_v2_1.json"""
import sys, json, re, hashlib, collections, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import ROOT, DS, META, CONFIG, read_jsonl, write_jsonl, load_schema
HELD_VAL_LAYOUT = lambda t: "ABC"[int(hashlib.sha1(f"v2.1-val-layout|{t}".encode()).hexdigest()[:8], 16) % 3]
AGAMI = {"Scanned_Type2": "val", "Digital_Type2": "test", "Digital_Type1": "train", "Scanned_Type1": "train"}
TAX_BY_SUBTYPE = {("insurance_premium_receipt", "health"): ["section_80d_evidence"], ("insurance_premium_receipt", "life"): ["section_80c_evidence"],
                  ("ev_document", None): ["ev_benefit_evidence"]}
TAX_REVIEW = {("ev_document", "purchase_invoice"): "purchase invoice may support an 80EEB loan-interest claim only if the vehicle is financed - mapping ambiguous",
              ("school_fee_receipt", "with_other_fees"): "only the tuition component qualifies for 80C - candidate covers the whole receipt",
              ("rental_agreement", "residential"): "agreement supports HRA but is not proof of rent paid",
              ("loan_statement", "account_statement"): "loan purpose (home/education/EV) unknown - section 24(b)/80E/80EEB undecidable from type"}
CUES = {"invoice": ["invoice", "gstin", "bill to", "tax invoice"], "receipt": ["receipt", "cash memo", "thank you"], "fuel_receipt": ["petrol", "diesel", "fuel", "litre", "pump"],
        "hotel_bill": ["hotel", "room", "check-in", "check in", "guest", "folio"], "medical_bill": ["hospital", "patient", "pharmacy", "medicine", "clinic", "doctor"],
        "travel_document": ["flight", "boarding", "pnr", "passenger", "irctc", "airlines", "ticket"], "utility_bill": ["electricity", "kwh", "meter", "consumer no", "broadband", "water bill", "gas bill"]}
APPROVED_CONFIG = "approved_sources_v2.json"  # v1 reproduces the 2026-10-19 baseline approvals
def words_text(r):
    try: a = json.loads((DS / r["ann_path"]).read_text()); return " ".join(w["text"] for w in a["words"]).lower()
    except Exception: return None
def cue_count(txt, t): return sum(txt.count(c) for c in CUES[t])
def main():
    S2 = load_schema("v2"); rows = read_jsonl(META / "doc_type_manifest_v2.jsonl"); reg = {s["id"]: s for s in json.loads((META / "license_register_v2.json").read_text())["sources"]}; ap = json.loads((CONFIG / APPROVED_CONFIG).read_text())
    t1, t2 = set(ap["tier1_usable"]), set(ap["tier2_restricted_known_terms"]); excl = set(ap["excluded_until_cleared"])
    out = []; changes = collections.Counter(); queue = []
    prot = set()  # every document in a frozen Task A / Task B / document_type TEST split is protected (v2 only flagged real test docs; the 243 older synthetic test docs are in those test files too)
    for d in ("classification", "document_type", "taskB"):
        for x in read_jsonl(DS / "processed" / d / "test.jsonl"): prot.add(x["document_id"])
    for r in rows:
        r = dict(r); r["split_v2_0"] = r["split"]; r["group_id_v2_0"] = r["group_id"]; r["split_version"] = "v2.1"
        if r["source"] == "synthgen_v1":
            tid = f"synthgen__{r['doc_type']}__{r['layout']}"; r["template_id"] = tid; r["group_id"] = tid
            r["split"] = "test" if r["layout"] == "D" else ("val" if r["layout"] == HELD_VAL_LAYOUT(r["doc_type"]) else "train")
            r["split_basis"] = "template_held_out: layout D=test, rotating A/B/C=val, other two=train"
        elif r["source"] == "agami":
            t = r["original"]["template"]; r["template_id"] = f"agami__{t}"; r["group_id"] = r["template_id"]; r["split"] = AGAMI[t]; r["split_basis"] = "template_held_out: Scanned_Type2=val, Digital_Type2=test, others train"
        elif r["source"] == "fatura": r["template_id"] = r["group_id"]
        else: r["template_id"] = None
        changes[(r["source"], r["split_v2_0"], r["split"])] += r["split"] != r["split_v2_0"]
        key = (r["doc_type"], r["subtype"]); tax = TAX_BY_SUBTYPE.get(key) or TAX_BY_SUBTYPE.get((r["doc_type"], None))
        r["tax_mapping_note"] = None
        if r["source"] == "synthgen_v1":
            if tax is not None and tax != r["tax_evidence_candidates"]: r["tax_evidence_candidates_v2_0"] = r["tax_evidence_candidates"]; r["tax_evidence_candidates"] = tax; changes["tax_corrected"] += 1
            if key in TAX_REVIEW: r["tax_mapping_note"] = TAX_REVIEW[key]; changes["tax_flagged"] += 1
        if r["doc_id"] in prot and not r["in_protected_task_test"]: r["in_protected_task_test"] = True; changes["protected_flag_added"] += 1
        if r["source"] == "synthetic_in": r["split_basis"] = "inherited_from_manifest.jsonl (protected Task A/B split, UNCHANGED); same-template pages: see reports/doc_type/experiments_v3/synthetic_in_phash.json"; r["template_id"] = "synthetic_in__single_generator_template_family"
        rid = r["license_register_id"]; st = reg[rid]["status"]
        if rid in t1 and st == "usable": r["approval_tier"], r["approved_for_training"], r["approval_reason"] = 1, True, "licence status 'usable'"
        elif rid in t2 and st == "usable_with_restrictions": r["approval_tier"], r["approved_for_training"], r["approval_reason"] = 2, True, ap["tier2_restricted_known_terms"][rid]
        else: r["approval_tier"], r["approved_for_training"], r["approval_reason"] = None, False, ap["excluded_until_cleared"].get(rid, f"licence status '{st}'")
        assert not (rid in excl and r["approved_for_training"])
        out.append(r)
    # real-label cross-check (OCR text cues)
    flagged = collections.Counter()
    for r in out:
        if r["synthetic"] or r["review_status"] != "needs_mapping_review" or r["doc_type"] not in CUES: continue
        txt = words_text(r)
        if not txt: continue
        own = cue_count(txt, r["doc_type"]); others = {t: cue_count(txt, t) for t in CUES if t != r["doc_type"] and t not in ("receipt", "invoice")} if r["doc_type"] in ("receipt", "invoice") else {t: cue_count(txt, t) for t in CUES if t != r["doc_type"]}
        top, n = max(others.items(), key=lambda kv: kv[1]); strong_other = n >= (3 if r["doc_type"] in ("receipt", "invoice") else 2)
        if (own == 0 and strong_other) or (r["doc_type"] in ("receipt", "invoice") and n >= 4):
            r["review_status"] = "flagged_for_review"; r["review_reason"] = f"text cues for '{top}' ({n}) vs label '{r['doc_type']}' ({own})"; flagged[(r["source"], r["doc_type"])] += 1
            queue.append(dict(doc_id=r["doc_id"], source=r["source"], label=r["doc_type"], reason=r["review_reason"], file=r["file_path"], priority="cue_conflict"))
    rnd = random.Random(20261019); by = collections.defaultdict(list)
    for r in out:
        if not r["synthetic"] and r["review_status"] == "needs_mapping_review": by[(r["source"], r["doc_type"])].append(r)
    for k, v in sorted(by.items()):
        for r in rnd.sample(v, min(8, len(v))): queue.append(dict(doc_id=r["doc_id"], source=r["source"], label=r["doc_type"], reason="stratified spot-check of collection-rule label (8 per source x type)", file=r["file_path"], priority="spot_check"))
    # schema v2.1
    S = json.loads(json.dumps(S2)); S["version"] = "doc-type-schema-v2.1"; S["review_status_values"]["flagged_for_review"] = "OCR text cues contradict the mapped label; needs a human decision"
    S["axes"]["C_tax_evidence"]["default_derivation"]["insurance_premium_receipt"] = ["section_80d_evidence", "section_80c_evidence"]; S["axes"]["C_tax_evidence"]["subtype_derivation_v2_1"] = {"insurance_premium_receipt/health": ["section_80d_evidence"], "insurance_premium_receipt/life": ["section_80c_evidence"], "ev_document": ["ev_benefit_evidence"]}
    S["splits"] = dict(real="unchanged from v2 (inherited from datasets/metadata/manifest.jsonl; protected Task A/B test docs stay test-only)", synthetic_generated="v2.1 TEMPLATE-held-out per doc_type: layout D=test, one rotating layout of A/B/C=val, other two=train; group_id = template (doc_type x layout)", fatura="by template (unchanged)", agami="by template: Scanned_Type2=val, Digital_Type2=test, Digital_Type1+Scanned_Type1=train")
    (CONFIG / "doc_type_schema_v2_1.json").write_text(json.dumps(S, indent=1))
    write_jsonl(META / "doc_type_manifest_v2_1.jsonl", out); write_jsonl(META / "doc_type_v2_1_real.jsonl", [r for r in out if not r["synthetic"]]); write_jsonl(META / "doc_type_v2_1_synthetic.jsonl", [r for r in out if r["synthetic"]])
    (META / "doc_type_v2_1_splits").mkdir(exist_ok=True); (META / "doc_type_v2_1_approved").mkdir(exist_ok=True); A = [r for r in out if r["approved_for_training"]]
    write_jsonl(META / "doc_type_v2_1_approved/manifest.jsonl", A)
    for kind in ("real", "synthetic"):
        for sp in ("train", "val", "test"):
            write_jsonl(META / "doc_type_v2_1_splits" / f"{kind}_{sp}.jsonl", [r for r in out if r["synthetic"] == (kind == "synthetic") and r["split"] == sp])
            write_jsonl(META / "doc_type_v2_1_approved" / f"{kind}_{sp}.jsonl", [r for r in A if r["synthetic"] == (kind == "synthetic") and r["split"] == sp])
    write_jsonl(META / "mapping_review_queue_v2_1.jsonl", queue)
    C = collections.Counter
    st = dict(n=len(out), approved=len(A), excluded=len(out) - len(A), approved_by_source=C(r["source"] for r in A), excluded_by_source=C(r["source"] for r in out if not r["approved_for_training"]),
              approved_counts={f"{'synthetic' if k[0] else 'real'}|{k[1]}": v for k, v in sorted(C((r["synthetic"], r["split"]) for r in A).items())},
              splits_changed={f"{k[0]}:{k[1]}->{k[2]}": v for k, v in changes.items() if isinstance(k, tuple) and v}, tax_corrected=changes["tax_corrected"], protected_flag_added=changes["protected_flag_added"], tax_flagged=changes["tax_flagged"],
              flagged_by_source_type={f"{k[0]}|{k[1]}": v for k, v in flagged.items()}, queue_size=len(queue), queue_priority=C(q["priority"] for q in queue))
    (META / "doc_type_v2_1_stats.json").write_text(json.dumps(st, indent=1)); print(json.dumps(st, indent=1))
if __name__ == "__main__": main()
