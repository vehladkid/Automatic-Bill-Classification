"""Orchestrates LLM-assisted extraction for one document: decide whether the LLM is needed, prompt it, validate every proposal against OCR, fuse with V1.
The original OCR text and V1 output are never overwritten; everything the LLM proposes is stored with its validation verdict for audit."""
import json, time
from . import validate as V
from .fuse import fuse, LLM_FIELDS
import doc_fields as DF
EXT_FIELDS = ["landlord_name", "tenant_name", "landlord_pan", "rental_period", "institution", "student_name", "fee_type", "insurer", "policy_number", "premium_amount", "lender", "loan_type", "interest_amount", "principal_amount", "period", "rent_amount", "provider", "payment_status", "payment_reference", "student_reference", "fee_period", "vehicle_details"]
from .client import OllamaClient, LLMUnavailable, LLMBadOutput
from .service import LLMService
from .config import settings, check_privacy, PrivacyError
S = V.S
import json as _j
from pathlib import Path as _P
_TX = _j.loads((_P(__file__).resolve().parents[2] / 'config' / 'taxonomy.json').read_text())
KINDS = sorted(_TX['profiles']) + ['expense_bill', 'non_expense_document', 'other', 'unknown']      # claim-type keys of config/taxonomy.json + generic outcomes

SYSTEM = ("You read OCR output of Indian bills and invoices and return JSON only. Rules: (1) Use ONLY text that is visible in the OCR lines or the image. "
          "(2) If a field is not present, use null - never guess, never compute a value that is not printed. (3) Copy values exactly as printed. "
          "(4) For every non-null field give 'evidence': the exact OCR text it came from. (5) List obvious OCR mistakes under ocr_corrections as {ocr, corrected, reason}.")
USER_T = ("Extract these fields: {fields}. Also extract these ONLY if the document is that kind of document, else null: {ext}.\nReturn JSON: {{\"fields\": {{<field>: {{\"value\": ..., \"evidence\": \"...\"}}}}, \"ocr_corrections\": [...], "
          "\"classification\": {{\"document_kind\": one of {kinds}, \"evidence\": \"exact OCR words that show this\"}}, \"expense_details\": {{\"description\": \"what was bought or paid for\", \"items\": [\"...\"]}}}}.\nhsn_sac is a list of AT MOST 5 unique digit strings (never repeat a value; null if none printed). Dates exactly as printed. Amounts as printed numbers.\n\nOCR lines (text | box x0,y0,x1,y1 in 0-1000 page units):\n{lines}")

def ocr_lines_for_prompt(segs, size, max_chars):
    W, H = size; rows = []
    for s in sorted(segs, key=lambda s: (round(s["bbox"][1] / max(H, 1) * 60), s["bbox"][0])):
        b = [round(s["bbox"][0] / W * 1000), round(s["bbox"][1] / H * 1000), round(s["bbox"][2] / W * 1000), round(s["bbox"][3] / H * 1000)]
        rows.append(f"{s['text']} | {b[0]},{b[1]},{b[2]},{b[3]}")
    txt = "\n".join(rows); return txt[:max_chars]

def needs_llm(v1, ocr, thr=0.6):
    """Reasons why the LLM is worth calling. Empty list = V1 output looks sound and OCR is clean, skip the LLM (privacy and latency)."""
    r = []
    for f in ("merchant_name", "invoice_date", "total_amount"):
        p = v1.get(f)
        if p is None: r.append(f"{f} missing")
        elif p.get("confidence") is not None and p["confidence"] < thr: r.append(f"{f} low confidence ({p['confidence']:.2f})")
        elif isinstance(p.get("value"), (int, float)) and f == "total_amount" and p["value"] == 0: r.append("total_amount is 0")
    g = v1.get("gstin")
    if g and not S.gstin_checksum_ok(str(g["value"])): r.append("GSTIN checksum invalid")
    if ocr.mean_conf() < 0.85: r.append(f"low OCR confidence (mean {ocr.mean_conf():.2f})")
    amt = {f: v1[f]["value"] for f in ("subtotal", "tax_total", "total_amount", "cgst", "sgst", "igst") if f in v1 and isinstance(v1[f]["value"], (int, float))}
    r += [f"arithmetic: {n} ({d})" for n, ok, d in V.consistency(amt) if not ok]
    return r

def run_llm(client, segs, size, image):
    user = USER_T.format(fields=", ".join(LLM_FIELDS), ext=", ".join(EXT_FIELDS), lines=ocr_lines_for_prompt(segs, size, client.s["max_ocr_chars"]), kinds=json.dumps(KINDS))
    client.last_fallback = False
    try: return client.chat_json(SYSTEM, user, image)
    except LLMBadOutput:
        # Observed on validation data: with an image the model sometimes falls into a repetition loop and Ollama aborts the generation (empty answer). Retry ONCE text-only (OCR lines + boxes).
        if image is None: raise
        client.last_fallback = True; return client.chat_json(SYSTEM, user, None)

def enhance(segs, size, v1_fields, image=None, client=None, cfg=None, force=False):
    """Returns a dict; 'status' in off | skipped | unavailable | privacy_blocked | bad_output | ok. Never raises for LLM problems - V1 output is always still valid."""
    cfg = cfg or settings(); res = dict(status="off", model=cfg["model"], mode=cfg["mode"], reasons=[], fields=None, rejected=[], corrections=[], seconds=None, location=None)
    if cfg["mode"] == "off" and not force: return res
    try: res["location"] = check_privacy(cfg)
    except PrivacyError as e: res.update(status="privacy_blocked", error=str(e)); return res
    ocr = V.OCR(segs); reasons = needs_llm(v1_fields, ocr); res["reasons"] = reasons
    if not reasons and cfg["mode"] != "always" and not force: res["status"] = "skipped"; return res
    try:
        base = client or OllamaClient(cfg); client = base if isinstance(base, LLMService) else LLMService(base, use_cache=cfg.get("cache", True)); t0 = time.time(); js, raw, secs = run_llm(client, segs, size, image); res["cached"] = bool(getattr(client, "last_cached", False))
    except LLMUnavailable as e: res.update(status="unavailable", error=str(e)); return res
    except LLMBadOutput as e: res.update(status="bad_output", error=str(e)); return res
    res["seconds"] = secs; res["raw"] = raw[:4000]
    checked = {}; proposals = (js.get("fields") if isinstance(js, dict) else None) or {}
    for f in LLM_FIELDS:
        p = proposals.get(f)
        if not isinstance(p, dict) and p is not None: p = {"value": p}
        val = (p or {}).get("value")
        if val in (None, "", []): continue
        vals = val if isinstance(val, list) else [val]
        if f == "hsn_sac":
            ok = [v for v in (str(x) for x in vals) if v.isdigit() and 4 <= len(v) <= 8 and v in ocr.text.replace(" ", "")]
            bad = [x for x in vals if str(x) not in ok]
            if ok: checked[f] = dict(value=ok, ground=dict(status="grounded", norm=",".join(ok), seg=None, detail="digit codes found in OCR"))
            if bad: res["rejected"].append(dict(field=f, value=bad, why="HSN/SAC code not found in OCR text"))
            continue
        g = V.ground(f, val, ocr)
        if g["status"] in ("grounded", "corrected"): checked[f] = dict(value=val, ground=g)
        else: res["rejected"].append(dict(field=f, value=val, why=g["detail"], status=g["status"]))
    lc = js.get("classification") if isinstance(js, dict) else None; res["llm_classification"] = None
    if isinstance(lc, dict) and lc.get("document_kind") in KINDS:
        ev = str(lc.get("evidence") or ""); gk = V.ground("merchant_name", ev, ocr) if ev.strip() and len(ev) >= 4 else dict(status="ungrounded")
        # the LLM's opinion only counts if the evidence it quotes is really in the document (or it says the doc is a generic bill / unknown)
        res["llm_classification"] = dict(kind=lc["document_kind"], evidence=ev[:160], grounded=gk["status"] in ("grounded", "corrected") or lc["document_kind"] in ("expense_bill", "unknown", "other", "non_expense_document"))
    ext_ok = {}
    for f in EXT_FIELDS:                       # extension fields: same grounding rule, no V1 counterpart to fuse with -> always flagged unverified
        p = proposals.get(f); p = p if isinstance(p, dict) or p is None else {"value": p}; val = (p or {}).get("value")
        if val in (None, "", []): continue
        g = V.ground(f, val, ocr)
        if g["status"] in ("grounded", "corrected"): ext_ok[f] = dict(value=val if DF.ALL_FIELDS[f] != "amount" else g["norm"], source="llm(validated)", evidence=g["detail"], bbox=list(g["seg"]["bbox"]) if g.get("seg") else None)
        else: res["rejected"].append(dict(field=f, value=val, why=g["detail"], status=g["status"]))
    res["ext_fields"] = ext_ok
    v1c = {f: p for f, p in v1_fields.items() if f != "hsn_sac"}
    fused, log = fuse(v1c, {f: c for f, c in checked.items() if f != "hsn_sac"}, ocr)
    if "hsn_sac" in checked: fused["hsn_sac"] = dict(value=checked["hsn_sac"]["value"], raw_text=None, confidence=None, source="llm(validated)", origin="llm", bbox=None, evidence=["codes found in OCR"], evidence_score=0.45, level="low", needs_review=True)
    elif "hsn_sac" in v1_fields: fused["hsn_sac"] = dict(v1_fields["hsn_sac"], origin="v1", evidence=["v1 only"], evidence_score=0.4, level="low", needs_review=True)
    for c in (js.get("ocr_corrections") or [] if isinstance(js, dict) else []):
        if not isinstance(c, dict): continue
        ok, why = V.validate_correction(c, ocr); res["corrections"].append(dict(c, accepted=bool(ok), check=why))
    # Policy chosen on VALIDATION data (295 real docs): values only the LLM produced were right 24% of the time (25/103) and cut F1 (0.703 -> 0.692); where V1 and the LLM agree 87% were right.
    # Default 'suggest': LLM-only values are NOT written into the extracted fields; they are returned as reviewable suggestions. 'add' restores the old behaviour.
    res["suggestions"] = {}
    if cfg.get("llm_only", "suggest") == "suggest":
        res["suggestions"] = {f: p for f, p in fused.items() if p.get("origin") == "llm"}; fused = {f: p for f, p in fused.items() if p.get("origin") != "llm"}
    ed = js.get("expense_details") if isinstance(js, dict) else None
    res.update(status="ok", fields=fused, conflicts=log, llm_only_policy=cfg.get("llm_only", "suggest"), expense_details=ed if isinstance(ed, dict) else None,
               review_fields=sorted(f for f, p in fused.items() if p["needs_review"]), note="LLM proposals are validated against OCR; nothing the LLM said is used unless it is grounded in the document.")
    return res
