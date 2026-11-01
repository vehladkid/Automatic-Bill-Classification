"""Evidence-based fusion of V1 extraction (rules / ranker / LayoutLMv3) with validated LLM proposals.
evidence_score is a transparent heuristic in [0,1]: V1-only = V1's own isotonic-calibrated confidence (0.40 if uncalibrated); V1+LLM agreeing and grounded = noisy-OR of V1 and a 0.70 agreement weight;
LLM-only (grounded) = 0.45; conflicts and arithmetic mismatches subtract; checksum/arithmetic agreement add. Weights are in W below. It is ORDINAL and NOT a calibrated probability (unlike V1's isotonic
confidence); the evaluation script reports measured accuracy per level so the levels can be judged on data, not trusted blindly."""
from . import validate as V
S = V.S
W = dict(v1_uncalibrated=0.40, llm_agree=0.70, llm_only=0.45, llm_corrected_penalty=-0.05, checksum=0.10, arithmetic=0.10, arithmetic_mismatch=-0.20, conflict=-0.30, v1_zero_amount=-0.30)
LEVELS = [(0.75, "high"), (0.5, "medium"), (0.0, "low")]
LLM_FIELDS = ["merchant_name", "invoice_number", "invoice_date", "total_amount", "subtotal", "tax_total", "cgst", "sgst", "igst", "gstin", "currency", "hsn_sac"]

def level(x): return next(n for t, n in LEVELS if x >= t)

def fuse(v1, llm_checked, ocr):
    """v1: {field: {'value','confidence','source','bbox',...}} (Task B output). llm_checked: {field: {'value','norm','ground':{...}}} already grounded-filtered.
    Returns (fused_fields, rejected_log)."""
    out, log = {}, []
    for f in sorted(set(v1) | set(llm_checked)):
        a, b = v1.get(f), llm_checked.get(f); ev = []; sc = 0.0; review = False
        an = v1_norm(f, a) if a else None
        bn = b["ground"]["norm"] if b else None
        agree = a is not None and b is not None and S.values_match(f, an, bn, soft=S.FIELDS.get(f) in ("text", "place"))
        if a and not b:
            c = a.get("confidence"); sc = c if c is not None else W["v1_uncalibrated"]; ev.append(f"v1:{a.get('source')}" + (f" conf {c:.2f}" if c is not None else " (uncalibrated)"))
            res = dict(a, origin="v1")
            if S.FIELDS.get(f) == "amount" and isinstance(a["value"], (int, float)) and a["value"] == 0: sc += W["v1_zero_amount"]; ev.append("amount is 0 (implausible)"); review = True
        elif b and not a:
            g = b["ground"]; sc = W["llm_only"] + (W["llm_corrected_penalty"] if g["status"] == "corrected" else 0); ev.append(f"llm only, OCR-{g['status']}: {g['detail']}")
            if g.get("checksum"): sc += W["checksum"]; ev.append("GSTIN checksum valid")
            res = dict(value=bn, raw_text=b["value"], confidence=None, source="llm(validated)", bbox=V_bbox(g), origin="llm"); review = True
        elif agree:
            c = a.get("confidence"); c0 = c if c is not None else W["v1_uncalibrated"]; sc = 1 - (1 - c0) * (1 - W["llm_agree"]); ev += [f"v1:{a.get('source')} and LLM agree", f"LLM value OCR-{b['ground']['status']}"]
            if b["ground"].get("checksum"): sc += W["checksum"]; ev.append("GSTIN checksum valid")
            res = dict(a, origin="v1+llm")
        else:    # conflict - keep V1 unless V1 fails a hard validator and the LLM value passes
            ga = V.ground(f, a["value"] if S.FIELDS.get(f) in ("amount", "date", "gstin", "id") else (a.get("raw_text") or a["value"]), ocr); win = "v1"
            if ga["status"] in ("ungrounded", "invalid") and b["ground"]["status"] in ("grounded", "corrected"): win = "llm"
            if f == "gstin" and not S.gstin_checksum_ok(str(a["value"])) and b["ground"].get("checksum"): win = "llm"
            sc = (W["llm_only"] if win == "llm" else (a.get("confidence") if a.get("confidence") is not None else W["v1_uncalibrated"])) + W["conflict"]; review = True
            ev.append(f"CONFLICT: v1={a['value']!r} vs llm={b['value']!r}; kept {win}"); log.append(dict(field=f, kind="conflict", v1=a["value"], llm=b["value"], kept=win))
            res = dict(a, origin="v1(conflict)") if win == "v1" else dict(value=bn, raw_text=b["value"], confidence=None, source="llm(validated)", bbox=V_bbox(b["ground"]), origin="llm(conflict)")
            res["alternative"] = b["value"] if win == "v1" else a["value"]
        out[f] = res; res["evidence"] = ev; res["evidence_score"] = round(max(0.0, min(0.99, sc)), 3)
    # cross-field arithmetic: reward fields that take part in satisfied relations, flag violated ones
    checks = V.consistency({f: S.normalize(f, p["value"]) if not isinstance(p["value"], (int, float)) else p["value"] for f, p in out.items() if S.FIELDS.get(f) == "amount"})
    for name, ok, detail in checks:
        parts = {"tax_total = cgst+sgst+igst": ("tax_total", "cgst", "sgst", "igst"), "total = subtotal + tax (within round-off)": ("total_amount", "subtotal", "tax_total", "cgst", "sgst", "igst"), "cgst = sgst": ("cgst", "sgst")}[name]
        for k in parts:
            if k in out:
                out[k]["evidence"].append(("arithmetic ok: " if ok else "ARITHMETIC MISMATCH: ") + name + f" ({detail})")
                out[k]["evidence_score"] = round(max(0.0, min(0.99, out[k]["evidence_score"] + (W["arithmetic"] if ok else W["arithmetic_mismatch"]))), 3)
                if not ok: out[k]["needs_review"] = True
    for f, p in out.items():
        p["level"] = level(p["evidence_score"]); p["needs_review"] = bool(p.get("needs_review") or p["level"] != "high" or p["origin"] in ("llm", "llm(conflict)", "v1(conflict)"))
    return out, log

def v1_norm(f, p):
    """V1 'value' is already typed/normalised for amount/date/gstin/id/int/code/enum (scripts/taskB/predict.typed); text fields are compared normalised."""
    return S.norm_text(p["value"]) if S.FIELDS.get(f) in ("text", "place") else p["value"]

def V_bbox(g):
    s = g.get("seg"); return list(s["bbox"]) if s and s.get("bbox") else None
