"""Experiment A - OCR TEXT RECOGNITION (images -> text), evaluated with Character / Word Error Rate against GROUND-TRUTH transcriptions.
Ground truth = the project's SROIE (receipts, line-level boxes) and FUNSD (scanned forms, word-level boxes) annotations whose ocr_source == 'ground_truth' (scripts/taskA/common_a.words).
Recognition-only protocol: every GT box is cropped from the page image and each recognizer must read the crop (no detector, so reading order and box-detection errors do not mix into CER/WER).
Splits: recognizers are trained / selected on TRAIN / VAL documents of the project's split; ALL numbers below are on the TEST-split documents (11 SROIE + 18 FUNSD).
Recognizers:  R1 RapidOCR PP-OCRv4 recognizer (production, ONNX, CTC),  R2 CRNN = CNN + 2-layer BiLSTM + CTC trained here from scratch on TRAIN crops (RNN-based recognizer),
              R3 TrOCR-small-printed (Transformer encoder-decoder with attention; pretrained by Microsoft; its public checkpoint was fine-tuned on SROIE, so SROIE numbers may be contaminated -> FUNSD is the clean subset).
LexRank cannot read pixels (it ranks sentences), so it has no OCR row.
usage: python ocr_exp.py {rapidocr|crnn|trocr}"""
import sys, json, time, re
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import *
from PIL import Image
MODE = sys.argv[1]; assert MODE in ("rapidocr", "crnn", "trocr")

def gt_lines(rows):
    docs = []
    for r in rows:
        if r["source"] not in ("sroie", "funsd") or not r["ann"]: continue
        a = json.loads(Path(r["ann"]).read_text())
        if a.get("ocr_source") != "ground_truth": continue
        im = Image.open(r["image"]).convert("L"); W, H = im.size; items = []
        for w in a["words"]:
            t = w.get("text", "").strip(); b = w.get("bbox", [])
            if not t or len(b) != 4: continue
            x0, y0, x1, y1 = [int(v) for v in b]; x0, x1 = max(0, min(x0, x1) - 2), min(W, max(x0, x1) + 2); y0, y1 = max(0, min(y0, y1) - 2), min(H, max(y0, y1) + 2)
            if x1 - x0 < 6 or y1 - y0 < 6: continue
            items.append((im.crop((x0, y0, x1, y1)), t))
        docs.append(dict(id=r["id"], source=r["source"], items=items))
    return docs

def edit_distance(a, b):
    if not a: return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
def cer_wer(refs, hyps):
    ce = sum(edit_distance(h, r) for r, h in zip(refs, hyps)); cn = sum(len(r) for r in refs)
    we = sum(edit_distance(h.split(), r.split()) for r, h in zip(refs, hyps)); wn = sum(len(r.split()) for r in refs)
    return ce / max(cn, 1), we / max(wn, 1), cn, wn

tr, va, te = splits(); test_docs = gt_lines(te); n_lines = sum(len(d["items"]) for d in test_docs)
print(f"test-split GT documents: {len(test_docs)} ({sum(d['source'] == 'sroie' for d in test_docs)} SROIE, {sum(d['source'] == 'funsd' for d in test_docs)} FUNSD), {n_lines} cropped lines/words")
info = dict(model=MODE, env=env_info()); hyp = {}      # hyp[doc_id] = list of strings

if MODE == "rapidocr":
    from rapidocr_onnxruntime import RapidOCR
    import subprocess
    eng = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, intra_op_num_threads=4, inter_op_num_threads=1)
    def read(crop):
        res, _ = eng(np.asarray(crop.convert("RGB")), use_det=False, use_cls=False, use_rec=True)
        return " ".join(x[0] for x in res) if res else ""
    for d in test_docs[:2]: [read(c) for c, _ in d["items"][:5]]                       # warm-up
    t_page = [];
    for d in test_docs:
        t0 = time.perf_counter(); hyp[d["id"]] = [read(c) for c, _ in d["items"]]; t_page.append(time.perf_counter() - t0)
    used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()[0])
    base = Path(__import__("rapidocr_onnxruntime").__file__).parent / "models"; size = (base / "ch_PP-OCRv4_rec_infer.onnx").stat().st_size
    info.update(arch="PP-OCRv4 text recognizer (ONNX, CTC decoding); production engine, used without its detector", model_bytes=size, peak_mem_mb=used, mem_note="GPU memory in use by the whole process (nvidia-smi), not torch peak", device="GPU (onnxruntime-gpu)", train_seconds=None)
elif MODE == "crnn":
    import torch, torch.nn as nn
    dev = "cuda"; torch.manual_seed(SEED); np.random.seed(SEED); tr_docs, va_docs = gt_lines(tr), gt_lines(va)
    chars = sorted({c for d in tr_docs for _, t in d["items"] for c in t}); c2i = {c: i + 1 for i, c in enumerate(chars)}; print("charset", len(chars), "| train docs", len(tr_docs), "val docs", len(va_docs))
    H, MAXW = 32, 640
    def prep(crop):
        w, h = crop.size; nw = min(MAXW, max(16, int(round(w * H / h)))); a = np.asarray(crop.resize((nw, H), Image.BILINEAR), np.float32) / 255.0
        return (a - a.mean()) / (a.std() + 1e-3)                                                                   # per-crop contrast normalisation
    def pack(docs): return [(prep(c), [c2i[x] for x in t if x in c2i]) for d in docs for c, t in d["items"]]
    Dtr, Dva = [x for x in pack(tr_docs) if x[1] and x[0].shape[1] >= 4 * len(x[1]) // 1 * 0 + 1], pack(va_docs)
    def batches(D, bs, shuffle):
        idx = np.argsort([x[0].shape[1] for x in D]); chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        if shuffle: np.random.shuffle(chunks)
        for ch in chunks:
            W = max(D[i][0].shape[1] for i in ch); X = np.zeros((len(ch), 1, H, W), np.float32)
            for k, i in enumerate(ch): X[k, 0, :, :D[i][0].shape[1]] = D[i][0]
            yield ch, torch.from_numpy(X)
    def blk(i, o, pool): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True)] + ([nn.MaxPool2d(pool)] if pool else [])
    class CRNN(nn.Module):
        def __init__(s, nc):
            super().__init__(); s.cnn = nn.Sequential(*blk(1, 64, 2), *blk(64, 128, 2), *blk(128, 256, None), *blk(256, 256, (2, 1)), *blk(256, 512, None), *blk(512, 512, (2, 1)), nn.Conv2d(512, 512, (2, 1)), nn.ReLU(True))
            s.rnn = nn.LSTM(512, 256, 2, bidirectional=True, batch_first=True, dropout=0.2); s.out = nn.Linear(512, nc)
        def forward(s, x): f = s.cnn(x).squeeze(2).permute(0, 2, 1); o, _ = s.rnn(f); return s.out(o)       # (B, T, nc); T = W/4
    net = CRNN(len(chars) + 1).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    def decode(lg):
        out = []
        for seq in lg.argmax(-1).cpu().numpy():
            s, prev = [], 0
            for k in seq:
                if k != prev and k != 0: s.append(chars[k - 1])
                prev = k
            out.append("".join(s))
        return out
    def val_cer():
        net.eval(); refs, hyps = [], []
        with torch.no_grad():
            for ch, X in batches(Dva, 128, False):
                hyps += decode(net(X.to(dev))); refs += ["".join(chars[k - 1] for k in Dva[i][1]) for i in ch]
        return cer_wer(refs, hyps)[0]
    best = (9, None, 0); hist = []; t0 = time.perf_counter(); torch.cuda.reset_peak_memory_stats(); EP, PAT = 45, 8
    import os
    if os.environ.get("CRNN_REUSE") and (OUT / "crnn_ocr_state.pt").exists():      # re-score the already-trained recogniser (same weights, same seed) without retraining; training info is carried over from the first run
        prev = json.loads((OUT / "ocr_crnn.json").read_text())["info"]; best = (prev["best_val_cer"], torch.load(OUT / "crnn_ocr_state.pt"), int(re.search(r"best epoch (\d+)", prev["arch"]).group(1)))
        hist = prev["history"]; EP = 0; t0 = time.perf_counter() - prev["train_seconds"]; print("REUSE: loaded trained CRNN, best epoch", best[2])
    for ep in range(EP):
        net.train(); tl = 0; nb = 0
        for ch, X in batches(Dtr, 64, True):
            lg = net(X.to(dev)).log_softmax(-1).permute(1, 0, 2); tg = torch.cat([torch.tensor(Dtr[i][1]) for i in ch]); tl_ = torch.tensor([len(Dtr[i][1]) for i in ch]); il = torch.full((len(ch),), lg.size(0), dtype=torch.long)
            loss = ctc(lg, tg, il, tl_); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); tl += float(loss.detach()); nb += 1
        c = val_cer(); hist.append(dict(epoch=ep + 1, train_ctc=tl / nb, val_cer=c))
        if c < best[0]: best = (c, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1)
        elif ep + 1 - best[2] >= PAT: break
        if ep % 5 == 0: print(f"epoch {ep + 1}: ctc {tl / nb:.3f} val CER {c:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    train_s = time.perf_counter() - t0; peak = torch.cuda.max_memory_allocated() / 2**20; net.load_state_dict(best[1]); net.eval()
    torch.save(best[1], OUT / "crnn_ocr_state.pt"); size = (OUT / "crnn_ocr_state.pt").stat().st_size
    def read_doc(items):
        X = [torch.from_numpy(prep(c)) for c, _ in items]; hs = []
        with torch.no_grad():
            for i, x in enumerate(X): hs += decode(net(x[None, None].to(dev)))
        return hs
    for d in test_docs[:2]: read_doc(d["items"][:5])
    t_page = []
    for d in test_docs:
        torch.cuda.synchronize(); t0 = time.perf_counter(); hyp[d["id"]] = read_doc(d["items"]); torch.cuda.synchronize(); t_page.append(time.perf_counter() - t0)
    info.update(arch="CRNN: 6-conv VGG-style CNN (height 32) + 2-layer BiLSTM(256) + CTC, greedy decoding; trained from scratch on %d train crops (%d chars), best epoch %d of %d, val CER %.3f" % (len(Dtr), len(chars), best[2], len(hist), best[0]),
                model_bytes=size, peak_mem_mb=peak, device="GPU (torch)", train_seconds=train_s, history=hist, params=int(sum(p.numel() for p in net.parameters())), best_val_cer=best[0])
else:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    P = EXP / "pretrained/trocr-small-printed"; dev = "cuda"; proc = TrOCRProcessor.from_pretrained(str(P)); net = VisionEncoderDecoderModel.from_pretrained(str(P)).to(dev).eval()
    torch.cuda.reset_peak_memory_stats()
    def read_doc(items, bs=16):
        out = []
        for i in range(0, len(items), bs):
            pv = proc(images=[c.convert("RGB") for c, _ in items[i:i + bs]], return_tensors="pt").pixel_values.to(dev)
            with torch.no_grad(): ids = net.generate(pv, max_new_tokens=64, num_beams=1)
            out += proc.batch_decode(ids, skip_special_tokens=True)
        return out
    for d in test_docs[:2]: read_doc(d["items"][:8])
    t_page = []
    for d in test_docs:
        torch.cuda.synchronize(); t0 = time.perf_counter(); hyp[d["id"]] = read_doc(d["items"]); torch.cuda.synchronize(); t_page.append(time.perf_counter() - t0)
    info.update(arch="TrOCR-small-printed: ViT-style image encoder + Transformer text decoder with cross-attention (microsoft/trocr-small-printed), pretrained checkpoint, zero fine-tuning here, greedy decoding", model_bytes=(P / "model.safetensors").stat().st_size,
                peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20, device="GPU (torch)", train_seconds=None, params=int(sum(p.numel() for p in net.parameters())),
                contamination_note="the public TrOCR printed checkpoints were fine-tuned on SROIE; our SROIE documents may have been seen -> treat the SROIE subset as non-comparable")

# ---- score ----
def subset(src): return [(t, h) for d in test_docs if d["source"] in src for (_, t), h in zip(d["items"], hyp[d["id"]])]
out = dict(info=info, n_docs=len(test_docs), n_lines=n_lines, latency_ms_per_page_median=float(np.median(t_page) * 1000), pages_per_second=float(len(t_page) / sum(t_page)),
           avg_lines_per_page=n_lines / len(test_docs), note="page = all GT boxes of one test document recognised one after another (batch 1 for RapidOCR/CRNN, batches of 16 for TrOCR); detector time is NOT included")
for name, src in (("all", ("sroie", "funsd")), ("sroie", ("sroie",)), ("funsd", ("funsd",))):
    pairs = subset(src); refs, hs = [p[0] for p in pairs], [p[1] for p in pairs]; c, w, cn, wn = cer_wer(refs, hs)
    norm = lambda s: s.casefold().replace(" ", ""); cns = cer_wer([norm(r) for r in refs], [norm(h) for h in hs])[0]; wcf = cer_wer([r.casefold() for r in refs], [h.casefold() for h in hs])[1]
    out[name] = dict(cer=c, wer=w, cer_normalised=cns, wer_casefold=wcf, n_lines=len(pairs), ref_chars=cn, ref_words=wn, masked_gt_lines=sum(1 for r in refs if set(r) <= {"*"}))
    print(f"{MODE} [{name}] CER strict {c:.4f} | CER case/space-insensitive {cns:.4f} | WER strict {w:.4f} | WER case-insensitive {wcf:.4f}  ({len(pairs)} lines)")
out["samples"] = [dict(ref=t, hyp=h) for t, h in subset(("sroie", "funsd"))[:12]]
_s = sorted(subset(("sroie",)), key=lambda p: -edit_distance(p[1], p[0]) / max(len(p[0]), 1)); out["worst_sroie"] = [dict(ref=t, hyp=h) for t, h in _s[:12]]
save_json(f"ocr_{MODE}.json", out); print("latency %.1f ms/page, %.2f pages/s, size %.1f MB" % (out["latency_ms_per_page_median"], out["pages_per_second"], info["model_bytes"] / 2**20))
