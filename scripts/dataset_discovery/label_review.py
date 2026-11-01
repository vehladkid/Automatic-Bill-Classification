"""Evidence-based review of the 286-record mapping-review queue (datasets/metadata/mapping_review_queue_v2_1.jsonl). Local only, read-only on every dataset/manifest; writes under reports/doc_type/licensing_and_collection/label_review/.
Per record: statuses  supported | likely_incorrect | ambiguous | blocked_provenance  (blocked = source excluded by config/approved_sources_v2.json; its evidence verdict is still stored but never applied).
Evidence: (1) OCR text: distinct class-cue terms (whole-word) per document type; (2) source metadata: label_quality of the original label (manual_folder / source_label / manual_review = strong; rule_high / unlabeled = weak, itself derived from text); (3) the independent expense category as corroboration only.
Axes stay independent: only doc_type is ever proposed for change; expense_category is checked and flagged, never changed (Task A labels); tax evidence is left untouched ('insufficient evidence', not forced).
Only HIGH-confidence changes on approved sources go into the reversible overlay; MEDIUM/LOW proposals go to the human queue. No filename- or generator-convention inference. Usage: label_review.py"""
import sys, re, json, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_type_lib import ROOT, DS, META, read_jsonl
OUT = ROOT / "reports/doc_type/licensing_and_collection/label_review"; CFG = ROOT / "config/approved_sources_v2.json"
CUES = {
 "invoice": ["tax invoice", "invoice no", "invoice", "gstin", "gst no", "bill to", "hsn", "cgst", "sgst", "igst", "place of supply", "irn", "purchase order"],
 "receipt": ["receipt", "cash memo", "cash receipt", "thank you", "tendered", "cashier", "subtotal", "change due"],
 "fuel_receipt": ["petrol", "diesel", "fuel", "litre", "litres", "ltr", "nozzle", "pump", "octane", "vehicle no", "hpcl", "bpcl", "iocl", "indian oil"],
 "hotel_bill": ["hotel", "room", "check-in", "check in", "check-out", "checkout", "guest", "folio", "tariff", "night", "resort", "reservation", "arrival", "departure", "housekeeping"],
 "medical_bill": ["hospital", "patient", "pharmacy", "medicine", "clinic", "doctor", "uhid", "ip no", "admission", "discharge", "consultation", "diagnostic", "tablet", "batch", "drug licence"],
 "travel_document": ["flight", "boarding", "pnr", "passenger", "irctc", "airlines", "ticket", "train", "coach", "seat", "journey", "e-ticket", "booking id", "fare", "bus", "ride", "cab", "trip"],
 "utility_bill": ["electricity", "kwh", "units consumed", "meter", "consumer no", "consumer number", "broadband", "water bill", "gas bill", "connection", "due date", "bill date", "previous reading", "current reading", "discom", "data usage", "sewerage", "landline"]}
EXPENSE_FOR = {"fuel_receipt": {"fuel"}, "hotel_bill": {"hotel"}, "medical_bill": {"medical"}, "travel_document": {"train", "bus", "flight", "transport", "parking"}, "utility_bill": {"electricity", "water", "gas", "internet"}, "receipt": {"food", "transport", "parking", "hotel", "medical", "miscellaneous", "fuel"}, "invoice": {"miscellaneous", "rent", "food"}, "other": {"non_expense"}}
STRONG_SRC = {"manual_folder", "source_label", "manual_review"}; GENERIC = {"receipt", "invoice"}
def pat(t): return re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])")
RX = {k: [(t, pat(t)) for t in v] for k, v in CUES.items()}
def score(txt): return {k: sorted(t for t, rx in v if rx.search(txt)) for k, v in RX.items()}
def text_of(r):
    a = json.loads((DS / r["ann_path"]).read_text()); return " ".join(w["text"] for w in a["words"]).lower()
def judge(r, excluded, txt=None):
    own = r["doc_type"]; txt = txt if txt is not None else text_of(r); nw = len(txt.split()); sc = score(txt); n = {k: len(v) for k, v in sc.items()}; oq = r["original"].get("label_quality"); strong = oq in STRONG_SRC
    spec = {k: v for k, v in n.items() if k not in GENERIC}; best, bn = max(spec.items(), key=lambda kv: kv[1]) if own not in ("other",) else max(n.items(), key=lambda kv: kv[1])
    if own in GENERIC: others = {k: v for k, v in spec.items()}; bo, bon = max(others.items(), key=lambda kv: kv[1])
    else: others = {k: v for k, v in n.items() if k != own}; bo, bon = max(others.items(), key=lambda kv: kv[1])
    ownn = n.get(own, 0); exp = r["expense_category"]; exp_ok = None if exp is None else (exp in EXPENSE_FOR.get(own, set()))
    ev = dict(words=nw, own_terms=sc.get(own, []), best_other=bo, best_other_terms=sc[bo][:8], own_score=ownn, other_score=bon, source_label_quality=oq, expense_category=exp, expense_consistent_with_type=exp_ok)
    proposed, conf, reason = None, None, ""
    if own == "other":  # forms (FUNSD): 'other' means "not an expense document"; supported when no expense-document cues dominate
        if max(n.values()) <= 2: status, reason = "supported", "scanned form with no expense-document cues; source label 'FUNSD = scanned forms'"
        else: status, reason = "ambiguous", f"form carries {bo} cues ({bon}) - may be an expense document"
    elif nw < 15: status, reason = "ambiguous", f"only {nw} OCR words: insufficient text"
    elif own in GENERIC:
        ex_agree = exp is not None and exp in EXPENSE_FOR.get(bo, set()) and bo in {"fuel_receipt", "hotel_bill", "medical_bill", "travel_document", "utility_bill"}
        if bon >= 3 and ownn <= bon:
            status = "ambiguous"; reason = f"generic '{own}' with specific {bo} cues ({bon}: {', '.join(sc[bo][:4])}); type overlap - a {bo.replace('_', ' ')} may also be filed as a generic {own}"
            if bon >= 4 and ownn <= 1 and (ex_agree or exp is None): status, proposed, conf = "likely_incorrect", bo, "medium"; reason += ("; independent expense category '" + str(exp) + "' agrees" if ex_agree else "; no expense category recorded") + ""
        elif ownn == 0:
            status, reason = "ambiguous", f"no '{own}' cue term in OCR and no specific type dominates"
            if own == "invoice" and n["receipt"] >= 2: proposed, conf = "receipt", "low"; reason = f"labelled 'invoice' but OCR has 0 invoice cues and {n['receipt']} receipt cues ({', '.join(sc['receipt'][:3])})"
        else: status, reason = "supported", f"'{own}' cues {ownn}; no specific type dominates (best other {bo}={bon})"
        if status == "supported" and not ownn and not strong: status, reason = "ambiguous", "no cue for the label and the label is rule-derived"
    else:
        exp_strong = oq in STRONG_SRC; ex_prop = exp is not None and exp in EXPENSE_FOR.get(bo, set()); ex_own = exp is not None and exp in EXPENSE_FOR.get(own, set())
        if ownn >= 2 and ownn >= bon: status, reason = "supported", f"{ownn} '{own}' cue terms in OCR ({', '.join(sc[own][:4])})" + ("; expense category consistent" if ex_own else "")
        elif bon >= 3 and ownn == 0 and bon >= ownn + 3:
            if exp_strong and ex_prop and not ex_own and bon >= 3: status, proposed, conf = "likely_incorrect", bo, "high"; reason = f"OCR shows {bon} '{bo}' cue terms ({', '.join(sc[bo][:4])}) and 0 for '{own}'; the independent expense category '{exp}' ({oq}) agrees with '{bo}' and not with '{own}'"
            elif not strong and bon >= 4 and ex_prop: status, proposed, conf = "likely_incorrect", bo, "high"; reason = f"label derived by rule; OCR shows {bon} '{bo}' cue terms and 0 for '{own}'; expense category {exp} agrees"
            elif not strong and bon >= 4: status, proposed, conf = "likely_incorrect", bo, "medium"; reason = f"label derived by rule; OCR shows {bon} '{bo}' cues and 0 for '{own}'; expense category does not corroborate"
            else: status, proposed, conf = "ambiguous", bo, "low"; reason = f"source says '{own}' ({oq}{'; expense ' + exp + ' agrees' if ex_own else ''}) but OCR shows {bon} '{bo}' cues ({', '.join(sc[bo][:4])}) and none for '{own}': possible overlap (e.g. a tax invoice for a utility) or wrong source label"
        elif ownn >= 1 and ownn >= bon - 1: status, reason = "supported", f"{ownn} '{own}' cue(s); other types not stronger (best {bo}={bon})"
        else: status, reason = "ambiguous", f"weak evidence: own {ownn}, best other {bo}={bon}, label {oq}"
        if status == "supported" and ownn < 2 and not strong: status, reason = "ambiguous", "single cue term and label is rule-derived"
    ev_status = status; blocked = r["source"] in excluded
    if blocked: status = "blocked_provenance"
    return dict(status=status, evidence_verdict=ev_status, confidence=conf, proposed_doc_type=proposed, reason=reason, evidence=ev, tax_evidence="not assessed: commercial document, candidates unchanged (insufficient evidence to assert any)", expense_category_flag=(None if exp_ok in (None, True) else f"expense category '{exp}' unusual for '{own}' - review (Task A label not changed)"))
def main():
    OUT.mkdir(parents=True, exist_ok=True); cfg = json.loads(CFG.read_text()); excluded = set(cfg["excluded_until_cleared"]); man = {r["doc_id"]: r for r in read_jsonl(META / "doc_type_manifest_v2_1.jsonl")}; queue = read_jsonl(META / "mapping_review_queue_v2_1.jsonl"); res = []
    for q in queue:
        r = man[q["doc_id"]]; j = judge(r, excluded)
        res.append(dict(doc_id=r["doc_id"], source=r["source"], approved_source=r["approved_for_training"], queue_priority=q["priority"], queue_reason=q["reason"], original_labels=dict(doc_type=r["doc_type"], expense_category=r["expense_category"], tax_evidence_candidates=r["tax_evidence_candidates"], review_status=r["review_status"], source_label=r["original"]), **j))
    changes = [dict(change_id=f"LR1-{i:04d}", doc_id=x["doc_id"], field="doc_type", old=x["original_labels"]["doc_type"], proposed=x["proposed_doc_type"], confidence=x["confidence"], evidence=x["evidence"], reason=x["reason"], approved_source=x["approved_source"], status=x["status"],
                    applied=bool(x["confidence"] == "high" and x["approved_source"] and x["status"] == "likely_incorrect")) for i, x in enumerate([y for y in res if y["proposed_doc_type"]], 1)]
    for c in changes: c["applied_to"] = "overlay only (manifest untouched)" if c["applied"] else ("not applied: source excluded" if not c["approved_source"] else "not applied: confidence below high")
    overlay = [dict(change_id=c["change_id"], doc_id=c["doc_id"], field=c["field"], old=c["old"], new=c["proposed"], confidence=c["confidence"], reason=c["reason"], evidence_terms=c["evidence"]["best_other_terms"], reversible=True) for c in changes if c["applied"]]
    prio = {"likely_incorrect": 0, "ambiguous": 1}; rank = lambda x: (prio.get(x["status"], 3), 0 if x["queue_priority"] == "cue_conflict" else 1, 0 if x["proposed_doc_type"] else 1)
    hq = [dict(rank=i, doc_id=x["doc_id"], source=x["source"], current=x["original_labels"]["doc_type"], suggested=x["proposed_doc_type"], status=x["status"], why=x["reason"], approved_source=x["approved_source"]) for i, x in enumerate(sorted([y for y in res if y["status"] in ("likely_incorrect", "ambiguous") and not (y["status"] == "likely_incorrect" and y["confidence"] == "high")], key=rank), 1)]
    def w(name, rows): (OUT / name).write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    w("review_results.jsonl", res); w("proposed_changes.jsonl", changes); w("label_overlay_v1.jsonl", overlay); w("human_review_queue.jsonl", hq)
    C = collections.Counter; summ = dict(queue_records=len(res), unique_doc_ids=len({x["doc_id"] for x in res}), by_status=dict(C(x["status"] for x in res)), by_evidence_verdict_all=dict(C(x["evidence_verdict"] for x in res)), approved_sources_only=dict(C(x["status"] for x in res if x["approved_source"])),
        blocked_by_source=dict(C(x["source"] for x in res if x["status"] == "blocked_provenance")), blocked_evidence_verdict=dict(C(x["evidence_verdict"] for x in res if x["status"] == "blocked_provenance")), proposed_changes=len(changes), by_confidence=dict(C(c["confidence"] for c in changes)), applied_to_overlay=len(overlay),
        human_queue=len(hq), human_queue_by_source=dict(C(h["source"] for h in hq)), by_source_status={f"{k[0]}|{k[1]}": v for k, v in sorted(C((x["source"], x["status"]) for x in res).items())}, expense_flags=sum(bool(x["expense_category_flag"]) for x in res))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))
if __name__ == "__main__": main()
