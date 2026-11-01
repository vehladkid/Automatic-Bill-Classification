"""Verifies model sizes, training times, latency and artefact reproducibility. Inference/timing only - nothing is trained.
(1) file sizes + SHA-256 of downloaded checkpoints against the Hugging Face file hashes; (2) stored training-time fields vs the logs/histories they came from;
(3) saved BiLSTM weights + production TF-IDF / LayoutLMv3 checkpoints re-run on the test set must reproduce the stored test logits;
(4) latency re-measured (same protocol, shared machine -> compared within a factor of 2); XLM-R latency uses the PRETRAINED weights (same architecture; the fine-tuned weights were not saved);
(5) what cannot be reproduced."""
import sys, json, re, time, hashlib, collections, socket
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; EXP = ROOT / "experiments/text_vs_neural"; OUT = EXP / "outputs"; VER = EXP / "verification"
sys.path.insert(0, str(ROOT / "scripts" / "taskA")); sys.path.insert(0, str(EXP))
import torch, torch.nn as nn
res = dict(); MB = 2 ** 20
J = lambda n: json.loads((OUT / n).read_text())
# ---------- (1) sizes + hashes ----------
files = {"tfidf production joblib": ROOT / "models/taskA/text_lr.joblib", "bilstm state": OUT / "bilstm_state.pt", "bilstm_attn state": OUT / "bilstm_attn_state.pt", "layoutlmv3 production": ROOT / "models/taskA/layoutlmv3.pt",
         "xlm-roberta-base pretrained": EXP / "pretrained/xlm-roberta-base/model.safetensors", "trocr-small-printed": EXP / "pretrained/trocr-small-printed/model.safetensors", "crnn ocr state": OUT / "crnn_ocr_state.pt",
         "rapidocr rec onnx": ROOT / ".venv/lib/python3.12/site-packages/rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx"}
stored_bytes = {"tfidf production joblib": J("m1_tfidf.json")["model_bytes"], "bilstm state": J("m_bilstm.json")["model_bytes"], "bilstm_attn state": J("m_bilstm_attn.json")["model_bytes"], "layoutlmv3 production": J("m6_layoutlmv3.json")["model_bytes"],
                "xlm-roberta-base pretrained": J("m5_xlmr.json")["model_bytes"], "trocr-small-printed": J("ocr_trocr.json")["info"]["model_bytes"], "crnn ocr state": J("ocr_crnn.json")["info"]["model_bytes"], "rapidocr rec onnx": J("ocr_rapidocr.json")["info"]["model_bytes"]}
res["sizes"] = {k: dict(exists=p.exists(), bytes=p.stat().st_size if p.exists() else None, stored_bytes=stored_bytes[k], match=p.exists() and p.stat().st_size == stored_bytes[k], mib=round(p.stat().st_size / MB, 2) if p.exists() else None) for k, p in files.items()}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
import urllib3.util.connection as uc; uc.allowed_gai_family = lambda: socket.AF_INET
import requests
for key, repo in (("xlm-roberta-base pretrained", "FacebookAI/xlm-roberta-base"), ("trocr-small-printed", "microsoft/trocr-small-printed")):
    try:
        meta = requests.get(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=30).json(); f = next(s for s in meta["siblings"] if s["rfilename"] == "model.safetensors"); want = f.get("lfs", {}).get("sha256"); got = sha(files[key])
        res["sizes"][key].update(hf_size=f.get("size"), hf_sha256=want, local_sha256=got, sha256_match=(want == got), size_matches_hf=(f.get("size") == files[key].stat().st_size))
    except Exception as e: res["sizes"][key]["hash_check_error"] = f"{type(e).__name__}: {e}"
# ---------- (2) training-time provenance ----------
x = J("m5_xlmr.json"); lx = J("m6_layoutlmv3.json"); ref = json.loads((ROOT / "reports/taskA/layoutlmv3.json").read_text()); b = J("m_bilstm.json"); ba = J("m_bilstm_attn.json")
res["training_time"] = dict(xlmr_stored=x["train_seconds"], xlmr_history_last_elapsed=x["history"][-1]["elapsed_s"], xlmr_epochs_run=len(x["history"]), xlmr_best_epoch=x["best_epoch"], xlmr_best_equals_last_epoch=(x["best_epoch"] == len(x["history"])),
                            layoutlm_stored=lx["train_seconds"], layoutlm_log_last_sec=ref["log"][-1]["sec"], bilstm_selected_seed=b["selected_seed"], bilstm_stored=b["train_seconds"], bilstm_seed_times=[(s["seed"], s["train_seconds"]) for s in b["all_seeds"]],
                            attn_selected_seed=ba["selected_seed"], attn_stored=ba["train_seconds"], attn_seed_times=[(s["seed"], s["train_seconds"]) for s in ba["all_seeds"]])
res["training_time"]["bilstm_matches_selected_seed"] = abs(b["train_seconds"] - next(s["train_seconds"] for s in b["all_seeds"] if s["seed"] == b["selected_seed"])) < 1e-6
res["training_time"]["attn_matches_selected_seed"] = abs(ba["train_seconds"] - next(s["train_seconds"] for s in ba["all_seeds"] if s["seed"] == ba["selected_seed"])) < 1e-6
res["training_time"]["xlmr_time_consistent_with_history"] = abs(x["train_seconds"] - x["history"][-1]["elapsed_s"]) < 5
# ---------- (3) re-run saved weights -> stored logits ----------
from common_a import load, LABELS
te = load("test"); va = load("val"); tr = load("train"); TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]"); tokz = lambda t: [re.sub(r"\d", "0", w) for w in TOK.findall(t.lower())][:256]
freq = collections.Counter(w for r in tr for w in tokz(r["text"])); vocab = {"<pad>": 0, "<unk>": 1}
for w, c in freq.most_common(40000):
    if c >= 2: vocab[w] = len(vocab)
def enc(rows):
    X = np.zeros((len(rows), 256), np.int64); L = np.zeros(len(rows), np.int64)
    for i, r in enumerate(rows): ids = [vocab.get(w, 1) for w in tokz(r["text"])] or [1]; X[i, :len(ids)] = ids; L[i] = len(ids)
    return torch.from_numpy(X), torch.from_numpy(L)
Xte, Lte = enc(te)
class Net(nn.Module):
    def __init__(s, attn):
        super().__init__(); s.attn = attn; s.emb = nn.Embedding(len(vocab), 128, padding_idx=0); s.rnn = nn.LSTM(128, 128, 1, batch_first=True, bidirectional=True); s.drop = nn.Dropout(0.4); s.out = nn.Linear(256, 16)
        if attn: s.W = nn.Linear(256, 128); s.v = nn.Linear(128, 1, bias=False)
    def forward(s, x, L):
        m = x != 0; pk = nn.utils.rnn.pack_padded_sequence(s.emb(x), L.cpu(), batch_first=True, enforce_sorted=False); o_, (hn, _) = s.rnn(pk); h, _ = nn.utils.rnn.pad_packed_sequence(o_, batch_first=True, total_length=x.size(1))
        if s.attn: sc = s.v(torch.tanh(s.W(h))).squeeze(-1).masked_fill(~m, -1e9); a = torch.softmax(sc, 1); z = (a.unsqueeze(-1) * h).sum(1)
        else: z = torch.cat([hn[-2], hn[-1]], 1)
        return s.out(s.drop(z))
res["reproduce_from_saved_weights"] = {}; nets = {}
for k, attn in (("bilstm", False), ("bilstm_attn", True)):
    net = Net(attn); sd = torch.load(OUT / f"{k}_state.pt", map_location="cpu"); net.load_state_dict(sd); net.eval(); nets[k] = net
    with torch.no_grad(): lg = torch.cat([net(Xte[i:i + 256], Lte[i:i + 256]) for i in range(0, len(te), 256)]).numpy()
    st = np.load(OUT / f"{k}_logits.npz")["test"]; res["reproduce_from_saved_weights"][k] = dict(max_abs_logit_diff=float(np.abs(lg - st).max()), same_argmax_share=float((lg.argmax(1) == st.argmax(1)).mean()), vocab_rows_match=sd["emb.weight"].shape[0] == len(vocab))
import joblib
vm = joblib.load(ROOT / "models/taskA/text_lr.joblib"); v, m = vm; p = np.zeros((len(te), 16)); p[:, m.classes_] = m.predict_proba(v.transform([r["text"] for r in te])); st = np.load(OUT / "tfidf_lr_logits.npz")["test"]
res["reproduce_from_saved_weights"]["tfidf"] = dict(max_abs_logit_diff=float(np.abs(np.log(np.clip(p, 1e-6, 1)) - st).max()), same_argmax_share=float((p.argmax(1) == st.argmax(1)).mean()))
# ---------- (4) latency ----------
def tmed(fn, nw=5, nr=60, sync=None):
    for _ in range(nw): fn()
    ts = []
    for _ in range(nr):
        if sync: sync()
        t = time.perf_counter(); fn()
        if sync: sync()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))
lat = {}; sample = [r["text"] for r in te[:40]]; i = [0]
def one_tfidf(): t = sample[i[0] % 40]; i[0] += 1; m.predict_proba(v.transform([t]))
lat["tfidf_cpu"] = tmed(one_tfidf, 5, 120)
torch.set_num_threads(4); dev = "cuda"
for k, net in nets.items():
    smp = [(Xte[j:j + 1], Lte[j:j + 1]) for j in range(40)]; ii = [0]
    def cpu_one():
        x, l = smp[ii[0] % 40]; ii[0] += 1
        with torch.no_grad(): net(x, l)
    lat[f"{k}_cpu"] = tmed(cpu_one, 5, 120); g = Net(k == "bilstm_attn"); g.load_state_dict(net.state_dict()); g = g.to(dev).eval()
    def gpu_one():
        x, l = smp[ii[0] % 40]; ii[0] += 1
        with torch.no_grad(): g(x.to(dev), l.to(dev)).cpu()
    lat[f"{k}_gpu"] = tmed(gpu_one, 10, 120, torch.cuda.synchronize)
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
ck = EXP / "pretrained/xlm-roberta-base"; tk = AutoTokenizer.from_pretrained(str(ck)); ids = [tk(r["text"], truncation=True, max_length=256)["input_ids"] for r in te[:40]]
xm = XLMRobertaForSequenceClassification.from_pretrained(str(ck), num_labels=16).eval(); ii = [0]
def x_cpu():
    t = ids[ii[0] % 40]; ii[0] += 1
    with torch.no_grad(): xm(input_ids=torch.tensor([t]), attention_mask=torch.ones(1, len(t), dtype=torch.long))
lat["xlmr_cpu_pretrained_proxy"] = tmed(x_cpu, 2, 25); xm = xm.to(dev)
def x_gpu():
    t = ids[ii[0] % 40]; ii[0] += 1
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16): xm(input_ids=torch.tensor([t]).to(dev), attention_mask=torch.ones(1, len(t), dtype=torch.long).to(dev)).logits.float().cpu()
lat["xlmr_gpu_pretrained_proxy"] = tmed(x_gpu, 5, 80, torch.cuda.synchronize); del xm; torch.cuda.empty_cache()
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from common_a import words as row_words
from PIL import Image
proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False); lm = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=16); lm.load_state_dict(torch.load(ROOT / "models/taskA/layoutlmv3.pt", map_location="cpu")); lm.to(dev).eval(); docs = []; lg_lm = []
for r in te[:40]:
    im = Image.open(r["cache"]).convert("RGB"); W, H = im.size; ws = row_words(r)[:512] or [("empty", [0, 0, 1, 1])]; sx, sy = W / max(r["w"], 1), H / max(r["h"], 1)
    bx = [[int(np.clip(b[0] * sx * 1000 / W, 0, 1000)), int(np.clip(b[1] * sy * 1000 / H, 0, 1000)), int(np.clip(b[2] * sx * 1000 / W, 0, 1000)), int(np.clip(b[3] * sy * 1000 / H, 0, 1000))] for _, b in ws]; bx = [[min(a, c), min(b_, d), max(a, c), max(b_, d)] for a, b_, c, d in bx]; docs.append((im, [t for t, _ in ws], bx))
ii = [0]
def lm_one():
    im, t, bx = docs[ii[0] % 40]; ii[0] += 1; e = proc(im, t, boxes=bx, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16): return lm(**{a: q.to(dev) for a, q in e.items()}).logits.float().cpu().numpy()[0]
lat["layoutlmv3_gpu"] = tmed(lm_one, 5, 80, torch.cuda.synchronize); ii[0] = 0; mine = np.stack([lm_one() for _ in range(40)]); st = np.load(OUT / "layoutlmv3_logits.npz")["test"][:40]
res["reproduce_from_saved_weights"]["layoutlmv3_first40_test_docs"] = dict(max_abs_logit_diff=float(np.abs(mine - st).max()), same_argmax_share=float((mine.argmax(1) == st.argmax(1)).mean()), note="production checkpoint, bf16, first 40 test pages")
stored = {"tfidf_cpu": J("m1_tfidf.json")["latency_cpu"]["median_ms"], "bilstm_cpu": b["latency_cpu"]["median_ms"], "bilstm_gpu": b["latency_gpu"]["median_ms"], "bilstm_attn_cpu": ba["latency_cpu"]["median_ms"], "bilstm_attn_gpu": ba["latency_gpu"]["median_ms"],
          "xlmr_cpu_pretrained_proxy": x["latency_cpu"]["median_ms"], "xlmr_gpu_pretrained_proxy": x["latency_gpu"]["median_ms"], "layoutlmv3_gpu": lx["latency_gpu"]["median_ms"]}
res["latency"] = {k: dict(stored_ms=stored[k], remeasured_ms=lat[k], ratio=lat[k] / stored[k], within_2x=0.5 <= lat[k] / stored[k] <= 2) for k in stored}
for k, d in res["latency"].items(): print(f"latency {k:28s} stored {d['stored_ms']:8.2f}  now {d['remeasured_ms']:8.2f}  ratio {d['ratio']:.2f}  within 2x: {d['within_2x']}")
print(json.dumps(res["reproduce_from_saved_weights"], indent=1)); print(json.dumps(res["training_time"], indent=1, default=float)); print(json.dumps(res["sizes"], indent=1, default=str))
json.dump(res, open(VER / "resources_check.json", "w"), indent=1, default=str)
