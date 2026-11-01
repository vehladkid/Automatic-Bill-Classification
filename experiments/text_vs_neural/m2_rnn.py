"""Models 2 and 4 - BiLSTM and BiLSTM + additive attention pooling, trained from scratch on the SAME OCR text (row["text"]) and the SAME train/val/test split.
usage: python m2_rnn.py {bilstm|bilstm_attn}
Tokenizer: lower-case, regex  [a-z]+ | digit-runs | single symbol ; every digit is mapped to '0' (amounts/dates share a shape); vocabulary = train tokens with freq>=2 (max 40,000) + <pad>,<unk>.
Architecture: Embedding(128) -> BiLSTM(hidden 128, 1 layer, dropout 0.4 on the pooled vector) -> [last-state concat | additive attention pooling] -> Linear(16).
Training: AdamW lr 2e-3 wd 1e-2, batch 64, max 30 epochs, class-balanced cross-entropy (same 'balanced' idea as the LR baseline), early stopping on VAL real-only macro-F1 (patience 6).
Three seeds (0,1,2); the headline run is the seed with the best VALIDATION macro-F1 (test data is never used for selection). All seeds are reported."""
import sys, re, time, json, collections
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import *

KIND = sys.argv[1] if len(sys.argv) > 1 else "bilstm"; assert KIND in ("bilstm", "bilstm_attn")
EMB, HID, LAYERS, DROP, MAXLEN, BS, LR, WD, EPOCHS, PAT, SEEDS = 128, 128, 1, 0.4, 256, 64, 2e-3, 1e-2, 30, 6, (0, 1, 2)
dev = "cuda" if torch.cuda.is_available() else "cpu"
TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
def tokenize(t): return [re.sub(r"\d", "0", w) for w in TOK.findall(t.lower())][:MAXLEN]

tr, va, te = splits(); freq = collections.Counter(w for r in tr for w in tokenize(r["text"]))
vocab = {"<pad>": 0, "<unk>": 1}
for w, c in freq.most_common(40000):
    if c >= 2: vocab[w] = len(vocab)
def encode(rows):
    X = np.zeros((len(rows), MAXLEN), np.int64); L = np.zeros(len(rows), np.int64)
    for i, r in enumerate(rows):
        ids = [vocab.get(w, 1) for w in tokenize(r["text"])] or [1]; X[i, :len(ids)] = ids; L[i] = len(ids)
    return torch.from_numpy(X), torch.from_numpy(L)
(Xtr, Ltr), (Xva, Lva), (Xte, Lte) = encode(tr), encode(va), encode(te)
ytr = torch.tensor([r["y"] for r in tr]); yva = np.array([r["y"] for r in va]); real_va = np.array([not r["synthetic"] for r in va])

class Net(nn.Module):
    def __init__(self, attn):
        super().__init__(); self.attn = attn; self.emb = nn.Embedding(len(vocab), EMB, padding_idx=0); self.rnn = nn.LSTM(EMB, HID, LAYERS, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(DROP); self.out = nn.Linear(2 * HID, NC)
        if attn: self.W = nn.Linear(2 * HID, HID); self.v = nn.Linear(HID, 1, bias=False)
    def forward(self, x, L, return_attn=False):
        m = x != 0; pk = nn.utils.rnn.pack_padded_sequence(self.emb(x), L.cpu(), batch_first=True, enforce_sorted=False)    # packed: padding never reaches the recurrence
        o_, (hn, _) = self.rnn(pk); h, _ = nn.utils.rnn.pad_packed_sequence(o_, batch_first=True, total_length=x.size(1))
        if self.attn:
            s = self.v(torch.tanh(self.W(h))).squeeze(-1).masked_fill(~m, -1e9); a = torch.softmax(s, 1); z = (a.unsqueeze(-1) * h).sum(1)
        else:   # last real state of the forward direction + first state of the backward direction (= final hidden of both directions)
            z = torch.cat([hn[-2], hn[-1]], 1); a = None
        o = self.out(self.drop(z)); return (o, a) if return_attn else o

def logits_of(net, X, L, bs=256):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs): out.append(net(X[i:i + bs].to(dev), L[i:i + bs].to(dev)).float().cpu().numpy())
    return np.concatenate(out)

cnt = np.bincount(ytr.numpy(), minlength=NC).astype(float); w_cls = torch.tensor(np.where(cnt > 0, len(ytr) / (NC * np.maximum(cnt, 1)), 0.0), dtype=torch.float32).to(dev)   # sklearn 'balanced'
def run(seed):
    torch.manual_seed(seed); np.random.seed(seed); net = Net(KIND == "bilstm_attn").to(dev); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); lossf = nn.CrossEntropyLoss(weight=w_cls)
    best = (-1, None, 0); hist = []; t0 = time.perf_counter(); g = torch.Generator().manual_seed(seed)
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(Xtr), generator=g); tl = 0
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]; opt.zero_grad(); loss = lossf(net(Xtr[b].to(dev), Ltr[b].to(dev)), ytr[b].to(dev)); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += float(loss) * len(b)
        lv = logits_of(net, Xva, Lva); f = cls_metrics(softmax(lv)[real_va], yva[real_va], False)["macro_f1"]; hist.append(dict(epoch=ep + 1, train_loss=tl / len(perm), val_real_macro_f1=f))
        if f > best[0]: best = (f, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep + 1)
        elif ep + 1 - best[2] >= PAT: break
    train_s = time.perf_counter() - t0; peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else None
    net.load_state_dict(best[1]); return net, dict(seed=seed, best_epoch=best[2], epochs_run=len(hist), val_real_macro_f1=best[0], train_seconds=train_s, peak_gpu_mb=peak, history=hist)

runs = []
for s in SEEDS:
    net, info = run(s); lv, lt = logits_of(net, Xva, Lva), logits_of(net, Xte, Lte); info["_net"] = net; info["_lv"], info["_lt"] = lv, lt; runs.append(info)
    print(f"{KIND} seed {s}: best epoch {info['best_epoch']} val real macro-F1 {info['val_real_macro_f1']:.4f} train {info['train_seconds']:.1f}s", flush=True)
best_run = max(runs, key=lambda r: r["val_real_macro_f1"]); net, lv, lt = best_run["_net"], best_run["_lv"], best_run["_lt"]
save_logits(KIND, lv, lt)
res = evaluate_model("BiLSTM" if KIND == "bilstm" else "BiLSTM + attention", lv, lt, va, te)
# efficiency of the selected model
n_par = sum(p.numel() for p in net.parameters()); torch.save(net.state_dict(), OUT / f"{KIND}_state.pt"); size = (OUT / f"{KIND}_state.pt").stat().st_size
x1, l1 = Xte[:1], Lte[:1]; sample = [(Xte[i:i + 1], Lte[i:i + 1]) for i in range(40)]; k = [0]
def one_gpu():
    x, l = sample[k[0] % 40]; k[0] += 1
    with torch.no_grad(): net(x.to(dev), l.to(dev)).cpu()
lat_gpu = time_fn(one_gpu, 10, 120, torch.cuda.synchronize) if dev == "cuda" else None
net_cpu = Net(KIND == "bilstm_attn"); net_cpu.load_state_dict({k_: v.cpu() for k_, v in net.state_dict().items()}); net_cpu.eval(); torch.set_num_threads(4)
def one_cpu():
    x, l = sample[k[0] % 40]; k[0] += 1
    with torch.no_grad(): net_cpu(x, l)
lat_cpu = time_fn(one_cpu, 5, 120)
res.update(config=dict(kind=KIND, emb=EMB, hidden=HID, layers=LAYERS, bidirectional=True, dropout=DROP, max_len=MAXLEN, batch=BS, lr=LR, weight_decay=WD, optimizer="AdamW", max_epochs=EPOCHS, patience=PAT,
                       vocab_size=len(vocab), params=n_par, selection="best validation real-only macro-F1; seed chosen by validation macro-F1", seeds=list(SEEDS), tokenizer="regex lower-case, digits->0"),
           train_seconds=best_run["train_seconds"], device="GPU", latency_gpu=lat_gpu, latency_cpu=lat_cpu, model_bytes=size, peak_mem_mb=best_run["peak_gpu_mb"], selected_seed=best_run["seed"], best_epoch=best_run["best_epoch"],
           all_seeds=[{k_: v for k_, v in r.items() if not k_.startswith("_")} for r in runs],
           seeds_test_real=[dict(seed=r["seed"], **{m: cls_metrics(softmax(r["_lt"] / fit_T(r["_lv"], np.array([q["y"] for q in va])))[[not q["synthetic"] for q in te]], np.array([q["y"] for q in te])[[not q["synthetic"] for q in te]])[m] for m in ("accuracy", "macro_f1", "ece")}) for r in runs])
save_json(f"m_{KIND}.json", res)
print("selected seed", best_run["seed"], "| real test raw:", {k_: round(res["raw"]["real"][k_], 4) for k_ in ("accuracy", "macro_f1", "ece", "nll")}, "| calibrated:", {k_: round(res["calibrated"]["real"][k_], 4) for k_ in ("accuracy", "macro_f1", "ece", "nll")})
print("latency gpu/cpu ms", lat_gpu and round(lat_gpu["median_ms"], 2), round(lat_cpu["median_ms"], 2), "| params", n_par, "| size MB", round(size / 2**20, 2), "| per-seed test (calibrated):", [(s_["seed"], round(s_["accuracy"], 3), round(s_["macro_f1"], 3)) for s_ in res["seeds_test_real"]])
