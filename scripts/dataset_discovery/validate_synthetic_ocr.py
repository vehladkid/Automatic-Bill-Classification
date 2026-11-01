"""OCR sanity check of the synthetic corpus: does the rendered image actually SUPPORT its label? For a deterministic sample per document type (stratified over clean / scan / photo degradation) run the project's OCR
(RapidOCR, scripts/taskA/run_ocr_gpu.get_engine) and test (a) a class-defining phrase is readable, (b) the main amount of the document is recoverable from the OCR text. Reports by type and by degradation.
Output: reports/doc_type/synthetic_ocr_check_v1.json.   Usage: validate_synthetic_ocr.py [--per-type 8] [--dir datasets/synthetic_doctype/v1]"""
import sys, json, re, random, argparse, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts", "scripts/taskA", "scripts/taskB"): sys.path.insert(0, str(ROOT / p))
MAIN_AMOUNT = {"rent_receipt": ["rent_amount"], "rental_agreement": ["monthly_rent", "security_deposit"], "insurance_premium_receipt": ["total_paid"], "school_fee_receipt": ["total"], "investment_certificate": ["total_deposit", "amount_invested", "deposit_amount", "face_value"],
               "meal_voucher_statement": ["total"], "ev_document": ["total", "energy_charge", "monthly_lease_rental"], "loan_interest_certificate": ["interest_paid"], "loan_statement": ["emi"], "salary_slip": ["net_pay"], "form_16": ["taxable_income"]}
def inr(x):
    s = f"{x:.2f}"; ip, fp = s.split("."); 
    if len(ip) > 3:
        head, tail = ip[:-3], ip[-3:]; g = []
        while len(head) > 2: g.insert(0, head[-2:]); head = head[:-2]
        if head: g.insert(0, head)
        ip = ",".join(g + [tail])
    return ip + "." + fp
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--per-type", type=int, default=8); ap.add_argument("--dir", default="datasets/synthetic_doctype/v1"); a = ap.parse_args(); D = ROOT / a.dir
    from run_ocr_gpu import get_engine
    eng = get_engine(); rnd = random.Random(0); by = collections.defaultdict(list)
    for f in sorted((D / "ann").glob("*.json")): r = json.loads(f.read_text()); by[(r["doc_type"], r["degrade"]["mode"])].append(r)
    types = sorted({t for t, _ in by}); res = []; 
    for t in types:
        pick = []
        for m in ("clean", "scan", "photo"): pick += rnd.sample(by[(t, m)], min(len(by[(t, m)]), max(2, a.per_type // 3 + (1 if m == "clean" else 0))))
        for r in pick:
            txt = ""
            for pg in r["pages"]:
                out, _ = eng(str(D / pg)); txt += "\n".join(x[1] for x in (out or [])) + "\n"
            nt = norm(txt); hit_phrase = any(norm(k) in nt for k in r["key_phrases"]); hit_title = norm(r["title"]) in nt
            amts = [r["fields"][k] for k in MAIN_AMOUNT[t] if isinstance(r["fields"].get(k), (int, float))]; digits = re.sub(r"[,\s]", "", txt)
            hit_amt = any(inr(x).replace(",", "") in digits or f"{x:.2f}".replace(",", "") in digits for x in amts) if amts else None
            res.append(dict(doc_id=r["doc_id"], doc_type=t, mode=r["degrade"]["mode"], layout=r["layout"], phrase=hit_phrase, title=hit_title, amount=hit_amt, ocr_chars=len(txt)))
        print(t, "ok", flush=True)
    def agg(rows): n = len(rows); return dict(n=n, key_phrase_readable=round(sum(x["phrase"] for x in rows) / n, 3), title_readable=round(sum(x["title"] for x in rows) / n, 3), main_amount_recovered=round(sum(1 for x in rows if x["amount"]) / max(1, sum(1 for x in rows if x["amount"] is not None)), 3))
    out = dict(sample_size=len(res), overall=agg(res), by_type={t: agg([x for x in res if x["doc_type"] == t]) for t in types}, by_degradation={m: agg([x for x in res if x["mode"] == m]) for m in ("clean", "scan", "photo")}, by_layout={l: agg([x for x in res if x["layout"] == l]) for l in "ABCD" if any(x["layout"] == l for x in res)},
               failures=[x for x in res if not x["phrase"] or x["amount"] is False][:30])
    (ROOT / "reports/doc_type/synthetic_ocr_check_v1.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: out[k] for k in ("sample_size", "overall", "by_degradation", "by_layout")}, indent=1))
    for t, v in out["by_type"].items(): print(f"  {t:28s} n={v['n']:2d} phrase={v['key_phrase_readable']:.2f} title={v['title_readable']:.2f} amount={v['main_amount_recovered']:.2f}")
    print("failures:", [(x["doc_id"], x["mode"], x["layout"], x["phrase"], x["amount"]) for x in out["failures"][:10]])
