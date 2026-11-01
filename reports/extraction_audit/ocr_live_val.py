"""Phase 2: run the APP's OCR path on every real VALIDATION image, then (a) score OCR text against dataset ground-truth text where it exists (SROIE, CORD), (b) measure
how many ground-truth FIELD values are recoverable from the text, (c) run Task B V1 on the live OCR and on the dataset words, same scorer. Read-only on datasets/ and models/.
Needs the GPU (RapidOCR + LayoutLMv3): do not run together with the Streamlit app or Ollama.   Usage: .venv/bin/python reports/extraction_audit/ocr_live_val.py [--limit N]
Writes: live_ocr_val.json (segments per doc, cache), ocr_live_val_results.json."""
import sys, json, re, argparse, time, pickle, collections, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/"app", ROOT/"scripts", ROOT/"scripts"/"taskB", ROOT/"scripts"/"taskA"): sys.path.insert(0, str(p))
import numpy as np
from PIL import Image
from schema import *
from data import load
from evaluate import evaluate, micro
from words import split_words, reading_order
import service as SV
import llm_eval as LE
from llm import validate as V
HERE = ROOT/"reports/extraction_audit"; F = LE.FIELDS_EVAL

def lev(a, b):
    """Levenshtein distance between two sequences (Hyyro bit-parallel; Python ints)."""
    if not a: return len(b)
    if not b: return len(a)
    if len(a) < len(b): a, b = b, a
    m = len(b); peq = collections.defaultdict(int)
    for i, c in enumerate(b): peq[c] |= 1 << i
    mask = (1 << m) - 1; pv, mv, score = mask, 0, m; hb = 1 << (m - 1)
    for c in a:
        eq = peq.get(c, 0); xv = eq | mv; xh = (((eq & pv) + pv) ^ pv) | eq; ph = mv | ~(xh | pv); mh = pv & xh
        if ph & hb: score += 1
        if mh & hb: score -= 1
        ph = ((ph << 1) | 1) & mask; mh = (mh << 1) & mask; pv = mh | ~(xv | ph) & mask; mv = ph & xv
    return score

def toks(segs):
    """the SAME word split + reading order the app uses"""
    ws = reading_order(split_words(segs)); return [w["text"] for w in ws]

def text_scores(gt_segs, ocr_segs):
    g, o = toks(gt_segs), toks(ocr_segs)
    gs, os_ = "".join(g).casefold(), "".join(o).casefold()                     # whitespace-free so glued/split tokens do not dominate CER
    cer = lev(gs, os_) / max(len(gs), 1)
    vocab = {}; enc = lambda ts: "".join(chr(0x4e00 + vocab.setdefault(t.casefold(), len(vocab))) for t in ts)
    wer = lev(enc(g), enc(o)) / max(len(g), 1)
    cg, co = collections.Counter(t.casefold() for t in g), collections.Counter(t.casefold() for t in o); hit = sum((cg & co).values())     # order-independent
    return dict(cer=cer, wer=wer, tok_recall=hit / max(sum(cg.values()), 1), tok_precision=hit / max(sum(co.values()), 1), n_gt_tok=len(g), n_ocr_tok=len(o), n_gt_chars=len(gs))

_DATE_LOOSE = re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})|(\d{1,2})\s*([A-Za-z]{3})[A-Za-z]*[\s\-,.]*(\d{2,4})|(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})")
def date_loose(gt_iso, text):
    """is the GT date present in the text in ANY common printed form, tolerating lost spaces / glued neighbours (no \\b requirement)?"""
    t = re.sub(r"\s+", "", text)
    for m in _DATE_LOOSE.finditer(t):
        for cand in {m.group(0), m.group(0)[:-2] if m.group(0)[-2:].isdigit() and len(m.group(0)) > 8 else m.group(0)}:
            v = parse_date(cand)
            if v == gt_iso: return True
    return False

def recoverable(f, d, segs, text_all):
    g = d["fields"].get(f)
    if not g or g["norm"] is None: return None
    ocr = V.OCR(segs)
    if FIELDS[f] == "date": return V.ground(f, g["norm"], ocr)["status"] in ("grounded", "corrected") or date_loose(g["norm"], text_all)
    if f == "currency": return None                  # inferred label, never printed as such: not an OCR question
    if FIELDS[f] in ("amount",): return V.ground(f, g["norm"], ocr)["status"] in ("grounded", "corrected") or any(abs(parse_amount(x) - g["norm"]) < 0.011 for x in re.findall(r"\d[\d.,]*\d|\d", re.sub(r"[^0-9.,\n ]", " ", text_all)) if parse_amount(x) is not None)
    return V.ground(f, g["value"] if FIELDS[f] in ("text", "place") else g["norm"], ocr)["status"] in ("grounded", "corrected")

def to_pred(out):
    return {f: dict(value=str(p["value"]), norm=(p["value"] if FIELDS[f] in ("amount", "date", "gstin", "id", "int", "code", "enum") else norm_text(p["value"])), score=p.get("model_score"), confidence=p.get("confidence")) for f, p in out.items() if f in F and p.get("value") is not None}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int); a = ap.parse_args()
    docs = [d for d in load("val") if not d["synthetic"]]
    if a.limit: docs = docs[:a.limit]
    svc = SV.BillService().load(); print("loaded", svc.errors, svc.load_seconds, "s", flush=True)
    cache_f = HERE/"live_ocr_val.json"; live = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    t0 = time.time(); tmp = Path(tempfile.mkdtemp(prefix="ocrval_"))
    for i, d in enumerate(docs):
        did = d["document_id"]
        if did in live: continue
        im = SV._load_image(ROOT/"datasets"/d["image"]); p = tmp/"page_1.png"; im.save(p)                 # identical preparation to BillService._prepare_pages
        t1 = time.time(); segs = svc._ocr(p); live[did] = dict(segs=segs, size=list(im.size), secs=round(time.time() - t1, 2))
        if (i + 1) % 25 == 0: print(f"  OCR {i + 1}/{len(docs)} {time.time() - t0:.0f}s", flush=True); cache_f.write_text(json.dumps(live))
    cache_f.write_text(json.dumps(live))
    res = dict(n_docs=len(docs), ocr_secs_mean=round(float(np.mean([live[d['document_id']]['secs'] for d in docs])), 2))
    # ---------- (a) OCR text quality vs dataset ground-truth text (SROIE, CORD only: the only sources whose words are transcriptions) ----------
    per = {"sroie": [], "cord": []}
    for d in docs:
        if d["source_dataset"] in per: per[d["source_dataset"]].append(text_scores(LE.segs_of(d) if False else [dict(text=w, bbox=b) for w, b in zip(d["words"], d["boxes"])], live[d["document_id"]]["segs"]))
    res["ocr_text_vs_ground_truth"] = {s: {k: round(float(np.mean([x[k] for x in v])), 4) for k in ("cer", "wer", "tok_recall", "tok_precision")} | {"n_docs": len(v), "cer_median": round(float(np.median([x["cer"] for x in v])), 4), "cer_p90": round(float(np.percentile([x["cer"] for x in v], 90)), 4)} for s, v in per.items() if v}
    # ---------- (b) field-value recoverability from text: dataset words vs live RapidOCR ----------
    rec = collections.defaultdict(lambda: collections.Counter())
    for d in docs:
        did = d["document_id"]; gt_segs = [dict(text=w, bbox=b, conf=1.0) for w, b in zip(d["words"], d["boxes"])]; ls = live[did]["segs"]
        grp = "rapidocr_source" if d["source_dataset"] in ("receiptbench", "invoices_pt") else "gt_ocr_source"
        tgt, tlv = "\n".join(s["text"] for s in gt_segs), "\n".join(s["text"] for s in ls)
        for f in F:
            if f not in d["supported_fields"]: continue
            r0, r1 = recoverable(f, d, gt_segs, tgt), recoverable(f, d, ls, tlv)
            if r0 is None: continue
            c = rec[(grp, f)]; c["n"] += 1; c["in_dataset_words"] += bool(r0); c["in_live_ocr"] += bool(r1)
    res["field_recoverability"] = {f"{g}|{f}": dict(c) for (g, f), c in sorted(rec.items())}
    agg = collections.defaultdict(lambda: collections.Counter())
    for (g, f), c in rec.items():
        for k, v in c.items(): agg[g][k] += v
    res["field_recoverability_total"] = {g: dict(c) for g, c in agg.items()}
    # ---------- (c) Task B V1: live OCR vs dataset words ----------
    ex = svc.B; pl, pd_ = {}, {}
    for i, d in enumerate(docs):
        did = d["document_id"]; im = SV._load_image(ROOT/"datasets"/d["image"]).convert("RGB"); ls = live[did]["segs"]; ws = reading_order(split_words(ls))
        dl = dict(document_id=did, words=[w["text"] for w in ws], boxes=[w["bbox"] for w in ws], width=im.width, height=im.height, country="?", supported_fields=list(svc.mB.FIELDS), fields={}, labels=[])
        pl[did] = to_pred(ex.run(dl, im) if ws else {})
        dd = dict(document_id=did, words=d["words"], boxes=d["boxes"], width=d["width"], height=d["height"], country="?", supported_fields=list(svc.mB.FIELDS), fields={}, labels=[])
        pd_[did] = to_pred(ex.run(dd, Image.open(ROOT/"datasets"/d["image"]).convert("RGB")))
    (HERE/"v1_preds_live_val.json").write_text(json.dumps(pl, default=float)); (HERE/"v1_preds_datasetwords_rerun_val.json").write_text(json.dumps(pd_, default=float))
    def slc(name, sub):
        out = {}
        for sysn, pr in (("v1_dataset_words_rerun", pd_), ("v1_live_rapidocr", pl)):
            ev = evaluate(sub, pr, F)["real"]; out[sysn] = dict(micro=micro(ev, F), per_field={f: dict(f1=r["f1"], tp=r["tp"], fp=r["fp"], fn=r["fn"], n_gt=r["n_gt"]) for f, r in ev.items()})
        return out
    res["v1_end_to_end"] = {"all": slc("all", docs), "gt_ocr_source(sroie+cord)": slc("g", [d for d in docs if d["source_dataset"] in ("sroie", "cord")]), "rapidocr_source(receiptbench+invoices_pt)": slc("r", [d for d in docs if d["source_dataset"] in ("receiptbench", "invoices_pt")]),
                            "sroie": slc("s", [d for d in docs if d["source_dataset"] == "sroie"]), "cord": slc("c", [d for d in docs if d["source_dataset"] == "cord"])}
    (HERE/"ocr_live_val_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("n_docs", "ocr_secs_mean", "ocr_text_vs_ground_truth", "field_recoverability_total")}, indent=1))
    for s, v in res["v1_end_to_end"].items(): print(s, {k: (x["micro"]["f1"], x["micro"]["tp"], x["micro"]["fp"], x["micro"]["fn"]) for k, x in v.items()})
if __name__ == "__main__": main()
