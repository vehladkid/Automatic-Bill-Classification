"""Resource / latency benchmark of ONE model in a FRESH process (so peak RSS is clean). usage: python resource_bench.py {tfidf|bilstm|bilstm_attn|xlmr|layoutlmv3} [xlmr checkpoint dir name]
Protocol: first 100 REAL test documents (fixed), CPU = 4 threads fp32, GPU = bf16 autocast for the Transformers (fp32 for the BiLSTMs, as in the pipeline), warm-up 5, repetitions 100 (XLM-R 40, LayoutLMv3 15), median + p95.
'model-only' = inputs already tokenised/vectorised; 'end-to-end' = from the stored OCR text (TF-IDF, BiLSTM, XLM-R) or from OCR words + boxes + page image (LayoutLMv3) -> tokenise/vectorise -> forward -> softmax/temperature; OCR itself is excluded for ALL models.
Throughput: 96 documents processed in batches of 1 / 8 / 32 (end-to-end). Model loading time and RSS reported separately. Nothing is trained."""
import sys, os, json, time, re, platform, collections
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4"); os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "taskA"))
from xlmr_lib import *
MODEL = sys.argv[1]; CK = sys.argv[2] if len(sys.argv) > 2 else "primary256_s0"
def status(k):
    for l in open("/proc/self/status"):
        if l.startswith(k + ":"): return int(l.split()[1]) / 1024
rss, hwm = (lambda: status("VmRSS")), (lambda: status("VmHWM"))
tr, va, te = splits(); docs = [r for r in te if not r["synthetic"]][:100]; texts = [r["text"] for r in docs]; out = dict(model=MODEL, n_docs=len(docs), threads=4, python=platform.python_version(), rss_before_model_mb=rss())
def tm(fn, n_warm=5, n=100, sync=None):
    for _ in range(n_warm): fn()
    ts = []
    for _ in range(n):
        if sync: sync()
        t = time.perf_counter(); fn()
        if sync: sync()
        ts.append((time.perf_counter() - t) * 1000)
    return dict(median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)), mean_ms=float(np.mean(ts)), sd_ms=float(np.std(ts)), n=n)
def tput(fn_batch, bs, total=96, sync=None):
    batches = [list(range(i, min(i + bs, total))) for i in range(0, total, bs)]; fn_batch(batches[0])
    if sync: sync()
    t = time.perf_counter()
    for b in batches: fn_batch(b)
    if sync: sync()
    return total / (time.perf_counter() - t)
import torch
cuda = torch.cuda.is_available(); sync = torch.cuda.synchronize if cuda else None; torch.set_num_threads(4)
def gpu_reset():
    if cuda: torch.cuda.reset_peak_memory_stats()
def gpu_peak(): return dict(peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20, peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20) if cuda else None
# ------------------------------------------------------------------ adapters: each returns (prep(list of idx)->inputs, fwd(inputs, device)->logits tensor/ndarray, load info)
t0 = time.perf_counter()
if MODEL == "tfidf":
    import joblib
    vm = joblib.load(PROD_MODELS / "text_lr.joblib"); v, m = vm; art = [PROD_MODELS / "text_lr.joblib"]; deps = ["scikit-learn", "numpy", "scipy", "joblib"]
    prep = lambda idx: v.transform([texts[i] for i in idx]); fwd = lambda X, dev: m.predict_proba(X); devices = ["cpu"]
elif MODEL in ("bilstm", "bilstm_attn"):
    import torch.nn as nn
    TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]"); tokz = lambda t: [re.sub(r"\d", "0", w) for w in TOK.findall(t.lower())][:256]
    freq = collections.Counter(w for r in tr for w in tokz(r["text"])); vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in freq.most_common(40000):
        if c >= 2: vocab[w] = len(vocab)
    class Net(nn.Module):
        def __init__(s, attn):
            super().__init__(); s.attn = attn; s.emb = nn.Embedding(len(vocab), 128, padding_idx=0); s.rnn = nn.LSTM(128, 128, 1, batch_first=True, bidirectional=True); s.drop = nn.Dropout(0.4); s.out = nn.Linear(256, 16)
            if attn: s.W = nn.Linear(256, 128); s.v = nn.Linear(128, 1, bias=False)
        def forward(s, x, L):
            m_ = x != 0; pk = nn.utils.rnn.pack_padded_sequence(s.emb(x), L.cpu(), batch_first=True, enforce_sorted=False); o_, (hn, _) = s.rnn(pk); h, _ = nn.utils.rnn.pad_packed_sequence(o_, batch_first=True, total_length=x.size(1))
            if s.attn: sc = s.v(torch.tanh(s.W(h))).squeeze(-1).masked_fill(~m_, -1e9); a_ = torch.softmax(sc, 1); z = (a_.unsqueeze(-1) * h).sum(1)
            else: z = torch.cat([hn[-2], hn[-1]], 1)
            return s.out(s.drop(z))
    net = Net(MODEL == "bilstm_attn"); sp = ROOT / "experiments/text_vs_neural/outputs" / f"{MODEL}_state.pt"; net.load_state_dict(torch.load(sp, map_location="cpu")); net.eval(); art = [sp]; deps = ["torch", "numpy"]
    def prep(idx):
        ids = [[vocab.get(w, 1) for w in tokz(texts[i])] or [1] for i in idx]; L = max(len(x) for x in ids); X = torch.zeros((len(ids), L), dtype=torch.long)
        for j, x in enumerate(ids): X[j, :len(x)] = torch.tensor(x)
        return X, torch.tensor([len(x) for x in ids])
    def fwd(inp, dev):
        with torch.no_grad(): return net(inp[0].to(dev), inp[1].to(dev))
    devices = ["cpu", "cuda"]
elif MODEL == "xlmr":
    tok = get_tokenizer(); ck = FIN / "runs" / CK / "trainable_fp16.safetensors"; net = load_model(ck if ck.exists() else None, NC).eval(); art = [PRETRAINED / "model.safetensors", ck]; deps = ["torch", "transformers", "tokenizers", "sentencepiece", "safetensors", "numpy"]
    cfgj = json.loads((FIN / "runs" / CK / "config.json").read_text()) if (FIN / "runs" / CK / "config.json").exists() else {}; MAXLEN = cfgj.get("args", {}).get("maxlen", 256); out["checkpoint"] = CK; out["maxlen"] = MAXLEN; out["checkpoint_loaded"] = ck.exists()
    def prep(idx):
        e = tok([texts[i] for i in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt"); return e["input_ids"], e["attention_mask"]
    def fwd(inp, dev):
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")): return net(input_ids=inp[0].to(dev), attention_mask=inp[1].to(dev)).logits.float()
    devices = ["cpu", "cuda"]
elif MODEL == "layoutlmv3":
    from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
    from common_a import words as row_words
    from PIL import Image
    proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False); net = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=NC); net.load_state_dict(torch.load(PROD_MODELS / "layoutlmv3.pt", map_location="cpu")); net.eval()
    art = [PROD_MODELS / "layoutlmv3.pt"]; deps = ["torch", "transformers", "Pillow", "numpy", "(page image + OCR words/boxes required at inference)"]; items = []
    for r in docs:
        im = Image.open(r["cache"]).convert("RGB"); W, H = im.size; ws = row_words(r)[:512] or [("empty", [0, 0, 1, 1])]; sx, sy = W / max(r["w"], 1), H / max(r["h"], 1)
        bx = [[int(np.clip(b[0] * sx * 1000 / W, 0, 1000)), int(np.clip(b[1] * sy * 1000 / H, 0, 1000)), int(np.clip(b[2] * sx * 1000 / W, 0, 1000)), int(np.clip(b[3] * sy * 1000 / H, 0, 1000))] for _, b in ws]; items.append((im, [t for t, _ in ws], [[min(a, c), min(b_, d), max(a, c), max(b_, d)] for a, b_, c, d in bx]))
    def prep(idx):
        e = proc([items[i][0] for i in idx], [items[i][1] for i in idx], boxes=[items[i][2] for i in idx], truncation=True, max_length=512, padding="max_length", return_tensors="pt"); return dict(e)
    def fwd(inp, dev):
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")): return net(**{k: v.to(dev) for k, v in inp.items()}).logits.float()
    devices = ["cpu", "cuda"]
out["load_seconds"] = time.perf_counter() - t0; out["rss_after_load_mb"] = rss(); out["artifact_files"] = {str(p.relative_to(ROOT)): p.stat().st_size for p in art if p and p.exists()}; out["deployment_dependencies"] = {d: (__import__(d.replace("-learn", "").replace("scikit", "sklearn").replace("Pillow", "PIL")).__version__ if not d.startswith("(") and d not in ("sentencepiece",) else "n/a") if (lambda x: True)(d) else "" for d in deps if d not in ("(page image + OCR words/boxes required at inference)",)}
if MODEL == "layoutlmv3": out["deployment_dependencies"]["requires"] = "page image + OCR words and boxes (not text alone)"
reps = {"tfidf": 100, "bilstm": 100, "bilstm_attn": 100, "xlmr": 40, "layoutlmv3": 15}[MODEL]; seq = [[i % len(docs)] for i in range(10000)]; k = [0]
def nxt():
    k[0] += 1; return seq[k[0] % len(docs)]
pre = {i: prep([i]) for i in range(len(docs))}                      # pre-computed inputs for model-only timing
for dev in devices:
    if dev == "cuda" and not cuda: continue
    if MODEL != "tfidf":
        net.to(dev)
    gpu_reset(); r = dict()
    sy = sync if dev == "cuda" else None
    r["model_only"] = tm(lambda: fwd(pre[nxt()[0]], dev), 5, reps, sy)
    r["end_to_end"] = tm(lambda: fwd(prep(nxt()), dev), 5, reps, sy)
    def safe_tput(bs, total=96):
        try: return tput(lambda idx: fwd(prep(idx), dev), bs, total, sy)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); return "OOM"
    r["throughput_docs_per_s_end_to_end"] = {bs: safe_tput(bs) for bs in (1, 8, 32)}
    if dev == "cuda":
        r["gpu_memory"] = gpu_peak(); gpu_reset(); ok = safe_tput(32, 32) != "OOM"; r["gpu_memory_batch32"] = dict(gpu_peak(), out_of_memory=not ok, note="batch 32 ran out of GPU memory while the 1.3 GB of the resident demo servers was also on the card" if not ok else "")
    out[dev] = r; print(MODEL, dev, "model-only %.2f ms | end-to-end %.2f ms | docs/s bs1/8/32:" % (r["model_only"]["median_ms"], r["end_to_end"]["median_ms"]), {b: (round(v, 1) if not isinstance(v, str) else v) for b, v in r["throughput_docs_per_s_end_to_end"].items()}, flush=True)
out["rss_peak_mb"] = hwm(); out["rss_peak_increment_over_before_model_mb"] = hwm() - out["rss_before_model_mb"]
# sanity: does the loaded model (esp. the XLM-R checkpoint) reproduce its stored predictions on these 100 docs?
if MODEL == "xlmr" and (FIN / "runs" / CK / "logits.npz").exists():
    st = np.load(FIN / "runs" / CK / "logits.npz")["test"]; idx_all = [i for i, r in enumerate(te) if not r["synthetic"]][:100]; net.to("cpu"); mine = np.concatenate([fwd(prep([i]), "cpu").numpy() for i in range(len(docs))])
    out["checkpoint_reproduces_stored_predictions"] = dict(same_argmax_share=float((mine.argmax(1) == st[idx_all].argmax(1)).mean()), max_abs_logit_diff=float(np.abs(mine - st[idx_all]).max()), note="fp32 CPU forward of the fp16-saved trainable weights vs bf16 GPU logits stored at training time")
(FIN / "outputs" / "resources").mkdir(exist_ok=True); (FIN / "outputs" / "resources" / f"{MODEL}{'_' + CK if MODEL == 'xlmr' else ''}.json").write_text(json.dumps(out, indent=1, default=float))
