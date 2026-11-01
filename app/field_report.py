"""Per-field report that keeps four things apart (never merged into one number):
  ocr_confidence       mean recognition confidence of the OCR segments under the field's box (None if no box)
  extraction           who produced the value + V1's isotonic-CALIBRATED confidence (validation-set based) or the LLM-fusion evidence level (ordinal, uncalibrated)
  validation           deterministic checks (format, date plausibility, GSTIN checksum, arithmetic) -> passed | failed | not_checked
  presence             extracted | extracted_unverified | conflict | illegible | absent (expected for this document type but not found)
classification confidence lives separately in the document profile. LLM self-reported confidence is never used."""
import datetime as dt
from llm import validate as V
S = V.S

def _overlap(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0) / max(1, (b[2] - b[0]) * (b[3] - b[1]))

def ocr_conf(bbox, segs):
    if not bbox: return None
    c = [float(s.get("conf", 1.0)) for s in segs if _overlap(bbox, s["bbox"]) > 0.3]
    return round(sum(c) / len(c), 3) if c else None

def _checks(f, v):
    t = S.FIELDS.get(f); out = []
    if t == "amount" and isinstance(v, (int, float)): out.append(("amount_positive", v > 0))
    if t == "date": out.append(("valid_date", bool(S.parse_date(str(v)) and 2000 <= int(str(v)[:4]) <= dt.date.today().year + 1)))
    if t == "gstin": out.append(("gstin_format", S.gstin_valid_format(str(v)))); out.append(("gstin_checksum", S.gstin_checksum_ok(str(v))))
    return out

def build(final, segs, expected=(), conflicts=()):
    """final: {field: {'value','confidence','source','bbox','evidence_score'?,'level'?,'needs_review'?,'origin'?}} -> {field: report}. expected: field names this document type should have."""
    amounts = {f: p["value"] for f, p in final.items() if S.FIELDS.get(f) == "amount" and isinstance(p["value"], (int, float))}
    arith = V.consistency(amounts); bad_arith = {k for n, ok, d in arith if not ok for k in {"tax_total = cgst+sgst+igst": ("tax_total", "cgst", "sgst", "igst"), "total = subtotal + tax (within round-off)": ("total_amount", "subtotal", "tax_total", "cgst", "sgst", "igst"), "cgst = sgst": ("cgst", "sgst")}[n]}
    out = {}; conflicted = {c["field"] for c in conflicts}
    for f, p in final.items():
        chk = _checks(f, p["value"]) + ([("arithmetic_consistent", f not in bad_arith)] if f in amounts and arith else [])
        failed = [n for n, ok in chk if not ok]; oc = ocr_conf(p.get("bbox"), segs)
        val = "failed" if failed else ("passed" if chk else "not_checked")
        if oc is not None and oc < 0.5: pres = "illegible"
        elif f in conflicted: pres = "conflict"
        elif p.get("needs_review") or failed or (p.get("confidence") is not None and p["confidence"] < 0.5): pres = "extracted_unverified"
        else: pres = "extracted"
        out[f] = dict(value=p["value"], presence=pres, ocr_confidence=oc,
                      extraction=dict(source=p.get("source"), calibrated_confidence=p.get("confidence") if p.get("origin", "v1") in ("v1",) or p.get("source", "").startswith(("layoutlm", "ranker")) else None,
                                      evidence_level=p.get("level"), evidence_score_is_probability=False),
                      validation=dict(status=val, failed=failed, checked=[n for n, _ in chk]), evidence=p.get("evidence"), bbox=p.get("bbox"))
    for f in expected:
        if f not in out: out[f] = dict(value=None, presence="absent", ocr_confidence=None, extraction=None, validation=dict(status="not_checked", failed=[], checked=[]), evidence=["not found in document"], bbox=None)
    return out
