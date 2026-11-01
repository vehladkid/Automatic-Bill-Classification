"""Input-truncation audit (Phase 2). Facts + diagnostics, no XLM-R training here:
(a) tokenizer behaviour (special tokens count toward max_length; which end is kept); (b) share of documents truncated per split / real-only / CLASS at 256 and 512; (c) how much text is discarded and what it looks like;
(d) does the discarded tail carry class information? production TF-IDF scored on full text vs head-only vs tail-only (no re-fit); (e) DIAGNOSTIC: TF-IDF re-fitted and scored under the SAME truncation policy as XLM-R (256 and 512 tokens).
Run low priority: OMP_NUM_THREADS=1 nice -n 19 python truncation_audit.py"""
import sys, json, re, time, collections, joblib
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "taskA"))
from xlmr_lib import *
import t1_text_baseline as T1
tr, va, te = splits(); tok = get_tokenizer(); C = json.loads((ROOT / "reports/taskA/text_baseline.json").read_text())["C"]; out = {}
# (a) tokenizer facts
e = tok(te[0]["text"], truncation=True, max_length=256); full = tok(te[0]["text"], truncation=False)
out["tokenizer"] = dict(first_id_is_bos=e["input_ids"][0] == tok.cls_token_id, last_id_is_eos=e["input_ids"][-1] == tok.sep_token_id, special_tokens_per_sequence=2, content_tokens_available_at_256=254, content_tokens_available_at_512=510, truncation_side=tok.truncation_side, example_doc_total_tokens=len(full["input_ids"]), example_tokens_kept=len(e["input_ids"]))
def ntok(rows): return np.array([len(tok(r["text"], truncation=False)["input_ids"]) for r in rows])
N = {s: ntok(rows) for s, rows in (("train", tr), ("val", va), ("test", te))}; R = {s: np.array([not r["synthetic"] for r in rows]) for s, rows in (("train", tr), ("val", va), ("test", te))}
out["share_truncated"] = {s: {f"over_{L}": dict(all_pct=float(100 * (N[s] > L).mean()), real_pct=float(100 * (N[s][R[s]] > L).mean())) for L in (256, 512)} for s in N}
# (b) per class
rows_by = {"train": tr, "val": va, "test": te}; out["per_class_test_real"] = {}; out["per_class_train"] = {}
for nm, s, mask_real in (("per_class_test_real", "test", True), ("per_class_train", "train", False)):
    for c in LABELS:
        idx = [i for i, r in enumerate(rows_by[s]) if LABELS[r["y"]] == c and (not mask_real or not r["synthetic"])]
        if idx: n = N[s][idx]; out[nm][c] = dict(n=len(idx), median_tokens=float(np.median(n)), over_256_pct=float(100 * (n > 256).mean()), over_512_pct=float(100 * (n > 512).mean()))
# (c) what is discarded (test real, truncated at 256)
disc = []; kw = re.compile(r"total|amount|gst|tax|cgst|sgst|balance|due|paid|invoice|thank", re.I)
heads, tails = [], []
for r in te:
    if r["synthetic"]: continue
    t = tok(r["text"], truncation=True, max_length=256, return_offsets_mapping=True); full_n = len(tok(r["text"], truncation=False)["input_ids"])
    if full_n <= 256: continue
    cut = t["offset_mapping"][-2][1]; head, tail = r["text"][:cut], r["text"][cut:]; disc.append(dict(frac_tokens_lost=(full_n - 256) / full_n, tail_chars=len(tail), tail_has_total_or_tax_words=bool(kw.search(tail)), head_has_total_or_tax_words=bool(kw.search(head)), y=r["y"], tail=tail, head=head, id=r["id"]))
out["discarded_test_real_256"] = dict(n_docs=len(disc), mean_frac_tokens_lost=float(np.mean([d["frac_tokens_lost"] for d in disc])), median_frac_tokens_lost=float(np.median([d["frac_tokens_lost"] for d in disc])), tail_mentions_total_tax_words_pct=float(100 * np.mean([d["tail_has_total_or_tax_words"] for d in disc])),
                                      head_mentions_total_tax_words_pct=float(100 * np.mean([d["head_has_total_or_tax_words"] for d in disc])), example_tails=[dict(label=LABELS[d["y"]], tail=d["tail"][:160]) for d in disc[:5]])
# (d) information in the tail: production TF-IDF (trained on full text) scored on full / head-only / tail-only text, for the truncated real test documents
vm = joblib.load(PROD_MODELS / "text_lr.joblib"); v, m = vm
def pred(texts):
    p = np.zeros((len(texts), NC)); p[:, m.classes_] = m.predict_proba(v.transform(texts)); return p.argmax(1)
if disc:
    ys = np.array([d["y"] for d in disc]); full_t = {r["id"]: r["text"] for r in te}; pf = pred([full_t[d["id"]] for d in disc]); ph = pred([d["head"] for d in disc]); pt = pred([d["tail"] if d["tail"].strip() else "empty" for d in disc])
    out["tail_information_production_tfidf"] = dict(n_docs=len(disc), accuracy_full_text=float((pf == ys).mean()), accuracy_head_only_256=float((ph == ys).mean()), accuracy_tail_only=float((pt == ys).mean()), note="production TF-IDF (trained on full text, not re-fitted) on the REAL test documents longer than 256 tokens; chance level = majority class share")
# (e) same-policy diagnostic: re-fit TF-IDF on truncated text
def cut_texts(rows, L):
    outt = []
    for r in rows:
        t = tok(r["text"], truncation=True, max_length=L, return_offsets_mapping=True); outt.append(dict(r, text=r["text"][:t["offset_mapping"][-2][1]] if len(t["input_ids"]) >= L else r["text"]))
    return outt
lg = lambda p: np.log(np.clip(p, 1e-6, 1)); diag = {}
for L in (256, 512):
    t0 = time.perf_counter(); trL, vaL, teL = cut_texts(tr, L), cut_texts(va, L), cut_texts(te, L); vmL = T1.fit(trL, C); pv, pt = T1.proba(vmL, vaL), T1.proba(vmL, teL); res = evaluate_model(f"TF-IDF/LR under XLM-R's {L}-token policy", lg(pv), lg(pt), va, te)
    diag[f"tfidf_{L}"] = dict(real_accuracy=res["calibrated"]["real"]["accuracy"], real_macro_f1=res["calibrated"]["real"]["macro_f1"], real_weighted_f1=res["calibrated"]["real"]["weighted_f1"], fit_seconds=time.perf_counter() - t0)
    np.savez(FIN / "outputs" / f"tfidf_trunc{L}_probs.npz", val=pv, test=pt); print(L, diag[f"tfidf_{L}"], flush=True)
out["tfidf_same_truncation_diagnostic"] = diag
(FIN / "outputs" / "truncation_audit.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps({k: out[k] for k in ("tokenizer", "share_truncated", "discarded_test_real_256", "tail_information_production_tfidf") if k in out}, indent=1, default=float)[:2800])
