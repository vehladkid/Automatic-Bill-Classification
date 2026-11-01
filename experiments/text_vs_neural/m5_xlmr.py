"""Model 5 - text-only pretrained Transformer: XLM-RoBERTa-base (FacebookAI/xlm-roberta-base, 278M parameters, 100-language SentencePiece vocabulary; the data contains English, German, Portuguese, Indonesian, Japanese ...).
Fine-tuned on the SAME OCR text (row["text"]) and SAME train split; model selection (epoch) and calibration on VALIDATION only; test is evaluated once with the selected epoch.
Tokenizer: XLM-R SentencePiece; max_length 256 tokens (measured at run time: median 184 tokens, about a third of the training documents are longer and are truncated at the end, i.e. the first 256 tokens are kept; see verification/truncation_check.json), dynamic padding.
Fine-tuning: all encoder layers and the classification head (the 192M-parameter word-embedding matrix is frozen so that AdamW fits in 6 GB), AdamW lr 3e-5 (wd 0.01), linear warm-up 10% then linear decay, batch 16, bf16 autocast, max 6 epochs, early stopping patience 2 on VAL real-only macro-F1, grad-clip 1.0,
class-balanced cross-entropy ('balanced' weights, as for the LR baseline). One seed only (time); no hyper-parameter search was done, so this is an untuned fine-tune."""
import sys, time, json
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import *
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, get_linear_schedule_with_warmup

CKPT = EXP / "pretrained/xlm-roberta-base"; MAXLEN, BS, LR, WD, EPOCHS, PAT, SEED0 = 256, 16, 3e-5, 0.01, 6, 2, 0
dev = "cuda"; torch.manual_seed(SEED0); np.random.seed(SEED0)
tr, va, te = splits(); tok = AutoTokenizer.from_pretrained(str(CKPT))
def enc(rows): return tok([r["text"] for r in rows], truncation=True, max_length=MAXLEN)
Etr, Eva, Ete = enc(tr), enc(va), enc(te); ytr = np.array([r["y"] for r in tr]); yva = np.array([r["y"] for r in va]); real_va = np.array([not r["synthetic"] for r in va])
print("tokens/doc median %d p90 %d | truncated at %d: train %.1f%%" % (np.median([len(x) for x in Etr["input_ids"]]), np.percentile([len(x) for x in Etr["input_ids"]], 90), MAXLEN, 100 * np.mean([len(x) >= MAXLEN for x in Etr["input_ids"]])))
def batch(E, idx):
    ids = [E["input_ids"][i] for i in idx]; L = max(len(x) for x in ids); X = torch.full((len(ids), L), tok.pad_token_id); A = torch.zeros((len(ids), L), dtype=torch.long)
    for j, x in enumerate(ids): X[j, :len(x)] = torch.tensor(x); A[j, :len(x)] = 1
    return X.to(dev), A.to(dev)
def predict(model, E, bs=64):
    model.eval(); out = []; order = np.argsort([len(x) for x in E["input_ids"]])
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]; X, A = batch(E, idx); out.append((idx, model(input_ids=X, attention_mask=A).logits.float().cpu().numpy()))
    lg = np.zeros((len(order), NC), np.float32)
    for idx, o in out: lg[idx] = o
    return lg
model = XLMRobertaForSequenceClassification.from_pretrained(str(CKPT), num_labels=NC)
model.roberta.embeddings.word_embeddings.weight.requires_grad_(False)       # 192M of the 278M parameters are the 250k-token embedding table: freezing it makes AdamW fit in 6 GB (resource-conscious setting, documented)
model = model.to(dev)
cnt = np.bincount(ytr, minlength=NC).astype(float); w = torch.tensor(np.where(cnt > 0, len(ytr) / (NC * np.maximum(cnt, 1)), 0.0), dtype=torch.float32).to(dev); lossf = nn.CrossEntropyLoss(weight=w)
nd = ["bias", "LayerNorm.weight"]; params = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in nd)], "weight_decay": WD}, {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in nd)], "weight_decay": 0.0}]
opt = torch.optim.AdamW(params, lr=LR); steps = EPOCHS * ((len(tr) + BS - 1) // BS); sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
torch.cuda.reset_peak_memory_stats(); best = (-1, None, 0); hist = []; t0 = time.perf_counter(); rng = np.random.default_rng(SEED0)
for ep in range(EPOCHS):
    model.train(); perm = rng.permutation(len(tr)); tl = 0
    for i in range(0, len(perm), BS):
        idx = perm[i:i + BS]; X, A = batch(Etr, idx)
        with torch.autocast(dev, dtype=torch.bfloat16): lg = model(input_ids=X, attention_mask=A).logits
        loss = lossf(lg.float(), torch.tensor(ytr[idx]).to(dev)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); tl += float(loss.detach()) * len(idx)
    lv = predict(model, Eva); f = cls_metrics(softmax(lv)[real_va], yva[real_va], False)["macro_f1"]; hist.append(dict(epoch=ep + 1, train_loss=tl / len(tr), val_real_macro_f1=f, elapsed_s=time.perf_counter() - t0))
    print(f"epoch {ep + 1}: train loss {tl / len(tr):.4f}  val real macro-F1 {f:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    if f > best[0]: best = (f, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1)
    elif ep + 1 - best[2] >= PAT: break
train_s = time.perf_counter() - t0; peak_train = torch.cuda.max_memory_allocated() / 2**20
model.load_state_dict(best[1]); lv, lt = predict(model, Eva), predict(model, Ete); save_logits("xlmr", lv, lt)
res = evaluate_model("XLM-RoBERTa-base (text-only)", lv, lt, va, te)
# efficiency: GPU (batch 1, warm) and CPU (4 threads) per-document latency on 40 test pages
k = [0]; one_ids = [{"input_ids": [Ete["input_ids"][i]]} for i in range(40)]
def one_gpu():
    X, A = batch(one_ids[k[0] % 40], [0]); k[0] += 1
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16): model(input_ids=X, attention_mask=A).logits.float().cpu()
torch.cuda.reset_peak_memory_stats(); lat_gpu = time_fn(one_gpu, 5, 80, torch.cuda.synchronize); peak_inf = torch.cuda.max_memory_allocated() / 2**20
model.to("cpu").float().eval(); torch.set_num_threads(4)
def one_cpu():
    ids = one_ids[k[0] % 40]["input_ids"][0]; k[0] += 1
    with torch.no_grad(): model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)).logits
lat_cpu = time_fn(one_cpu, 2, 25)
res.update(config=dict(checkpoint="FacebookAI/xlm-roberta-base", params=int(sum(p.numel() for p in model.parameters())), tokenizer="SentencePiece (XLM-R), 250k vocab", max_length=MAXLEN, truncation="keep first 256 tokens", padding="dynamic per batch",
                       finetune="all transformer layers + classifier head; the 250k-token word-embedding matrix is frozen (memory)", trainable_params=int(sum(p.numel() for p in model.parameters() if p.requires_grad)), lr=LR, weight_decay=WD, warmup="10% linear", batch=BS, precision="bf16 autocast", max_epochs=EPOCHS, patience=PAT, seed=SEED0, selection="best validation real-only macro-F1 epoch", loss="class-balanced CE"),
           train_seconds=train_s, best_epoch=best[2], history=hist, device="GPU", latency_gpu=lat_gpu, latency_cpu=lat_cpu, model_bytes=(CKPT / "model.safetensors").stat().st_size, peak_mem_mb=peak_train, peak_mem_inference_mb=peak_inf,
           note="model_bytes = size of the pretrained checkpoint (the fine-tuned weights have the same shape; they were not saved to disk to conserve space)")
save_json("m5_xlmr.json", res)
print("selected epoch", best[2], "| real test raw:", {a: round(res["raw"]["real"][a], 4) for a in ("accuracy", "macro_f1", "ece", "nll")}, "| calibrated:", {a: round(res["calibrated"]["real"][a], 4) for a in ("accuracy", "macro_f1", "ece", "nll")})
print("latency gpu %.1f ms cpu %.1f ms | train %.0fs | peak train MB %.0f" % (lat_gpu["median_ms"], lat_cpu["median_ms"], train_s, peak_train))
