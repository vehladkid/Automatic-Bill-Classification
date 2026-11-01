"""One XLM-R fine-tuning run. Output: runs/<name>_s<seed>/ {config.json, history.json, logits.npz (val,test; document level), trainable_fp16.safetensors, train.log}.
Test data is only scored ONCE, after training, with the epoch chosen on VALIDATION real-only macro-F1 (pre-registered). No test number is printed or used before that.
usage: python xlmr_run.py --name primary256 --seed 0 [--maxlen 256 --epochs 6 --patience 2 --bs 16 --accum 1 --chunk 0 --stride 64 --save 1 --lr 3e-5]"""
import argparse, sys, time, json, platform, os
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xlmr_lib import *
from transformers import get_linear_schedule_with_warmup
ap = argparse.ArgumentParser()
for k, d, t in (("name", "primary256", str), ("seed", 0, int), ("maxlen", 256, int), ("epochs", 6, int), ("patience", 2, int), ("bs", 16, int), ("accum", 1, int), ("chunk", 0, int), ("stride", 64, int), ("save", 1, int), ("lr", 3e-5, float), ("max_chunks", 6, int)): ap.add_argument("--" + k, default=d, type=t)
a = ap.parse_args(); RUN = FIN / "runs" / f"{a.name}_s{a.seed}"; RUN.mkdir(parents=True, exist_ok=True); LOG = open(RUN / "train.log", "w")
def log(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
dev = "cuda"; torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
tr, va, te = splits(); tok = get_tokenizer(); chunk = bool(a.chunk)
Itr, Otr = encode_docs(tok, tr, a.maxlen, chunk, a.stride, a.max_chunks); Iva, Ova = encode_docs(tok, va, a.maxlen, chunk, a.stride, a.max_chunks); Ite, Ote = encode_docs(tok, te, a.maxlen, chunk, a.stride, a.max_chunks)
ytr_doc = np.array([r["y"] for r in tr]); yva = np.array([r["y"] for r in va]); real_va = np.array([not r["synthetic"] for r in va]); ytr = ytr_doc[Otr]
log(f"run {a.name} seed {a.seed} | maxlen {a.maxlen} (incl. special tokens) chunk {chunk} | train samples {len(Itr)} (docs {len(tr)}), val {len(Iva)}, test {len(Ite)}")
def batch(I, idx):
    L = max(len(I[i]) for i in idx); X = torch.full((len(idx), L), tok.pad_token_id); A = torch.zeros((len(idx), L), dtype=torch.long)
    for j, i in enumerate(idx): X[j, :len(I[i])] = torch.tensor(I[i]); A[j, :len(I[i])] = 1
    return X.to(dev), A.to(dev)
def doc_logits(model, I, O, n_docs, bs=32):
    model.eval(); order = np.argsort([len(x) for x in I]); out = np.zeros((len(I), NC), np.float32)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]; X, A = batch(I, idx); out[idx] = model(input_ids=X, attention_mask=A).logits.float().cpu().numpy()
    if not chunk: return out
    agg = np.zeros((n_docs, NC), np.float32); cnt = np.zeros(n_docs)
    for k, d in enumerate(O): agg[d] += out[k]; cnt[d] += 1                       # document logit = mean of its chunk logits
    return agg / np.maximum(cnt, 1)[:, None]
torch.manual_seed(a.seed)
model = load_model(None, NC); model.roberta.embeddings.word_embeddings.weight.requires_grad_(False); model.to(dev)
n_train_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad)); n_params = int(sum(p.numel() for p in model.parameters())); log("trainable params", n_train_params, "of", n_params)
cnt = np.bincount(ytr_doc, minlength=NC).astype(float); w = torch.tensor(np.where(cnt > 0, len(ytr_doc) / (NC * np.maximum(cnt, 1)), 0.0), dtype=torch.float32).to(dev); lossf = nn.CrossEntropyLoss(weight=w)
nd = ["bias", "LayerNorm.weight"]; params = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in nd)], "weight_decay": 0.01}, {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in nd)], "weight_decay": 0.0}]
opt = torch.optim.AdamW(params, lr=a.lr); micro = a.bs // a.accum; steps_per_epoch = (len(Itr) + a.bs - 1) // a.bs; steps = a.epochs * steps_per_epoch; sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
torch.cuda.reset_peak_memory_stats(); best = (-1, None, 0); hist = []; t0 = time.perf_counter(); rng = np.random.default_rng(a.seed); status = "completed"
try:
    for ep in range(a.epochs):
        model.train(); perm = rng.permutation(len(Itr)); tl = 0.0
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            for j in range(0, len(idx), micro):
                sub = idx[j:j + micro]; X, A = batch(Itr, sub)
                with torch.autocast(dev, dtype=torch.bfloat16): lg = model(input_ids=X, attention_mask=A).logits
                loss = lossf(lg.float(), torch.tensor(ytr[sub]).to(dev)) * len(sub) / len(idx); loss.backward(); tl += float(loss.detach()) * len(idx)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
        lv = doc_logits(model, Iva, Ova, len(va)); f = cls_metrics(softmax(lv)[real_va], yva[real_va], False)["macro_f1"]; hist.append(dict(epoch=ep + 1, train_loss=tl / len(Itr), val_real_macro_f1=f, val_accuracy=float((lv.argmax(1) == yva).mean()), elapsed_s=time.perf_counter() - t0))
        log(f"epoch {ep + 1}: train loss {tl / len(Itr):.4f}  val real macro-F1 {f:.4f}  ({time.perf_counter() - t0:.0f}s)")
        if f > best[0]: best = (f, {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in dict(model.named_parameters()) and dict(model.named_parameters())[k].requires_grad}, ep + 1)
        elif ep + 1 - best[2] >= a.patience: log("early stop"); break
except torch.cuda.OutOfMemoryError as e:
    status = "FAILED: CUDA out of memory"; log(status, str(e)[:200])
train_s = time.perf_counter() - t0; peak = torch.cuda.max_memory_allocated() / 2**20; reserved = torch.cuda.max_memory_reserved() / 2**20
if status == "completed":
    model.load_state_dict(best[1], strict=False); lv = doc_logits(model, Iva, Ova, len(va)); lt = doc_logits(model, Ite, Ote, len(te))    # test scored once, with the validation-selected epoch
    np.savez(RUN / "logits.npz", val=lv, test=lt)
    ck = RUN / "trainable_fp16.safetensors"; n_saved = save_trainable(model, ck) if a.save else None
else: lv = lt = None
cfg = dict(name=a.name, seed=a.seed, status=status, args=vars(a), trainable_params=n_train_params, total_params=n_params, selected_epoch=best[2], selection="best validation real-only macro-F1 epoch (test never used)", best_val_real_macro_f1=best[0],
           train_seconds=train_s, peak_gpu_allocated_mb=peak, peak_gpu_reserved_mb=reserved, tokens_special_included_in_maxlen=True, n_train_samples=len(Itr), hardware=torch.cuda.get_device_name(0), gpu_total_mb=torch.cuda.get_device_properties(0).total_memory / 2**20,
           versions=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, transformers=__import__("transformers").__version__, numpy=np.__version__), pretrained_checkpoint=str(PRETRAINED / "model.safetensors"), checkpoint_file=str(RUN / "trainable_fp16.safetensors") if (status == "completed" and a.save) else None,
           command="python experiments/xlmr_vs_tfidf_final/xlmr_run.py " + " ".join(f"--{k} {v}" for k, v in vars(a).items()))
(RUN / "config.json").write_text(json.dumps(cfg, indent=1, default=float)); (RUN / "history.json").write_text(json.dumps(hist, indent=1, default=float))
log("done:", status, f"| selected epoch {best[2]} | val real macro-F1 {best[0]:.4f} | train {train_s:.0f}s | peak GPU {peak:.0f} MB")
