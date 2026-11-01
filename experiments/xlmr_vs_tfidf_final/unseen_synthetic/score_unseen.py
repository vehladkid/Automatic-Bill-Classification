"""Scores the FROZEN models on the unseen synthetic set (inference only; no tuning/calibration/selection) -> ../outputs/unseen_synthetic/{meta.json, tfidf_probs.npy, tfidf_head256_probs.npy, xlmr_<run>_logits.npy}.
Run after OCR is complete and the GPU is free:  HF_HUB_OFFLINE=1 python unseen_synthetic/score_unseen.py synthetic|real"""
import sys, json, time
import numpy as np, joblib, torch
from pathlib import Path
SET = sys.argv[1] if len(sys.argv) > 1 else "synthetic"; FIN = Path(__file__).resolve().parents[1]; HERE = FIN / {"synthetic": "unseen_synthetic", "real": "unseen_real"}[SET]; sys.path.insert(0, str(FIN))
from xlmr_lib import *                       # LABELS, get_tokenizer, load_model, encode_docs
OUT = FIN / "outputs" / f"unseen_{SET}"; OUT.mkdir(parents=True, exist_ok=True)
man = [json.loads(l) for l in open(HERE / "manifest.jsonl")]; ocr = {j["id"]: j for j in map(json.loads, open(HERE / "ocr.jsonl"))}; assert set(ocr) == {m["id"] for m in man}, "OCR incomplete"
rows = [dict(m, text=ocr[m["id"]]["text"], condition=m.get("condition", "real")) for m in man]; y = np.array([LABELS.index(m["label"]) for m in man]); tok = get_tokenizer()
def head256(t):
    e = tok(t, truncation=True, max_length=256, return_offsets_mapping=True); n_full = len(tok(t, truncation=False)["input_ids"])
    return (t[:e["offset_mapping"][-2][1]] if n_full > 256 else t), n_full
H = [head256(r["text"]) for r in rows]; ntok = np.array([h[1] for h in H])
meta = dict(n=len(rows), labels=LABELS, y=y.tolist(), ids=[r["id"] for r in rows], condition=[r["condition"] for r in rows], group=[r["group"] for r in rows], ocr_words=[ocr[r["id"]]["n_words"] for r in rows], n_tokens=ntok.tolist(),
            over_256_pct_by_condition={c: float(100 * (ntok[[r["condition"] == c for r in rows]] > 256).mean()) for c in sorted({r["condition"] for r in rows})}, ocr_empty=int(sum(ocr[r["id"]]["n_words"] == 0 for r in rows)))
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
v, m = joblib.load(FIN / "outputs/tfidf/tfidf_lr_refit.joblib")
def proba(texts): p = np.zeros((len(texts), len(LABELS))); p[:, m.classes_] = m.predict_proba(v.transform(texts)); return p
np.save(OUT / "tfidf_probs.npy", proba([r["text"] for r in rows])); np.save(OUT / "tfidf_head256_probs.npy", proba([h[0] for h in H]))
dev = "cuda"
for run in sorted((FIN / "runs").glob("primary256_s*")):
    ck = run / "trainable_fp16.safetensors"; cfg = json.loads((run / "config.json").read_text())
    if cfg["status"] != "completed" or not ck.exists(): continue
    net = load_model(ck, len(LABELS)).to(dev).eval(); I, _ = encode_docs(tok, rows, 256); order = np.argsort([len(x) for x in I]); lg = np.zeros((len(I), len(LABELS)), np.float32)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]; L = max(len(I[j]) for j in idx); X = torch.full((len(idx), L), tok.pad_token_id); A = torch.zeros((len(idx), L), dtype=torch.long)
            for k, j in enumerate(idx): X[k, :len(I[j])] = torch.tensor(I[j]); A[k, :len(I[j])] = 1
            lg[idx] = net(input_ids=X.to(dev), attention_mask=A.to(dev)).logits.float().cpu().numpy()
    np.save(OUT / f"xlmr_{run.name}_logits.npy", lg); print(run.name, "acc", float((lg.argmax(1) == y).mean()), flush=True); del net; torch.cuda.empty_cache()
