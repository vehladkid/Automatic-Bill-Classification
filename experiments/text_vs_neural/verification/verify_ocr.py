"""Independent verification of Experiment A (OCR). Inference only (no training). Steps:
1 independent Levenshtein implementations tested against known values and against the experiment's DP on random strings;
2 ground-truth crops rebuilt from the split files + annotation JSON (counts, a few crops saved for visual inspection);
3 the three recognisers re-run on those crops (RapidOCR, CRNN with the SAVED weights, TrOCR with the downloaded checkpoint) and CER / WER recomputed with the independent edit distance;
4 stored vs recomputed comparison."""
import sys, json, time, re, functools, random
import numpy as np
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[3]; EXP = ROOT / "experiments/text_vs_neural"; OUT = EXP / "outputs"; VER = EXP / "verification"
# ---------- 1. edit distance: two independent implementations + the experiment's own ----------
def lev_dp(a, b):                                     # copy of ocr_exp.edit_distance (the implementation under test)
    if not a: return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
def lev_np(a, b):                                     # independent: full matrix, numpy, different loop order
    a, b = list(a), list(b); m, n = len(a), len(b); D = np.zeros((m + 1, n + 1), int); D[:, 0] = np.arange(m + 1); D[0, :] = np.arange(n + 1)
    for j in range(1, n + 1):
        for i in range(1, m + 1): D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(D[m, n])
known = [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3), ("flaw", "lawn", 2), ("same", "same", 0), (["a", "b", "c"], ["a", "c"], 1)]
res = dict(levenshtein_known_cases=[(lev_np(a, b) == d and lev_dp(a, b) == d) for a, b, d in known]); random.seed(1)
fuzz = [("".join(random.choices("abc dE", k=random.randint(0, 25))), "".join(random.choices("abc dE", k=random.randint(0, 25)))) for _ in range(400)]
res["levenshtein_fuzz_400_agree"] = all(lev_dp(a, b) == lev_np(a, b) for a, b in fuzz); print("edit-distance known cases", res["levenshtein_known_cases"], "| fuzz agreement", res["levenshtein_fuzz_400_agree"])
def cer_wer(refs, hyps):
    ce = sum(lev_np(r, h) for r, h in zip(refs, hyps)); cn = sum(len(r) for r in refs); we = sum(lev_np(r.split(), h.split()) for r, h in zip(refs, hyps)); wn = sum(len(r.split()) for r in refs); return ce / cn, we / wn
# ---------- 2. ground-truth crops (own implementation) ----------
def split_rows(name): return [json.loads(l) for l in open(ROOT / f"datasets/processed/classification/{name}.jsonl")]
man = {}
for l in open(ROOT / "datasets/metadata/manifest.jsonl"):
    m = json.loads(l); man[m["document_id"]] = m
def crops(split):
    docs = []
    for r in split_rows(split):
        if r["source"] not in ("sroie", "funsd") or not r.get("ann_path"): continue
        a = json.loads((ROOT / "datasets" / r["ann_path"]).read_text())
        if a.get("ocr_source") != "ground_truth": continue
        im = Image.open(ROOT / "datasets" / r["image"]).convert("L"); W, H = im.size; items = []
        for w in a["words"]:
            t = w.get("text", "").strip(); b = w.get("bbox", [])
            if not t or len(b) != 4: continue
            x0, y0, x1, y1 = [int(v) for v in b]; xa, xb = sorted((x0, x1)); ya, yb = sorted((y0, y1)); xa, xb, ya, yb = max(0, xa - 2), min(W, xb + 2), max(0, ya - 2), min(H, yb + 2)
            if xb - xa < 6 or yb - ya < 6: continue
            items.append((im.crop((xa, ya, xb, yb)), t))
        docs.append(dict(id=r["document_id"], source=r["source"], items=items))
    return docs
test_docs, train_docs = crops("test"), crops("train"); lines = sum(len(d["items"]) for d in test_docs)
res["test_docs"] = dict(total=len(test_docs), sroie=sum(d["source"] == "sroie" for d in test_docs), funsd=sum(d["source"] == "funsd" for d in test_docs), lines=lines, sroie_lines=sum(len(d["items"]) for d in test_docs if d["source"] == "sroie"),
                        masked_gt_lines=sum(1 for d in test_docs for _, t in d["items"] if set(t) <= {"*"}))
res["train_test_doc_overlap"] = len({d["id"] for d in test_docs} & {d["id"] for d in train_docs}); print("test GT docs", res["test_docs"], "| train docs", len(train_docs), "| overlap with test", res["train_test_doc_overlap"])
rnd = random.Random(3); (VER / "crops").mkdir(exist_ok=True)
for k, (d, (c, t)) in enumerate([(d, rnd.choice(d["items"])) for d in rnd.sample(test_docs, 6)]): c.save(VER / "crops" / f"{k}_{d['source']}_{re.sub(r'[^A-Za-z0-9]', '_', t)[:30]}.png")
flat = [(d["source"], c, t) for d in test_docs for c, t in d["items"]]
# ---------- 3. re-run recognisers ----------
import torch, torch.nn as nn
hyps = {}
from rapidocr_onnxruntime import RapidOCR
eng = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True, intra_op_num_threads=4, inter_op_num_threads=1)
def rr(c): r_, _ = eng(np.asarray(c.convert("RGB")), use_det=False, use_cls=False, use_rec=True); return " ".join(x[0] for x in r_) if r_ else ""
hyps["rapidocr"] = [rr(c) for _, c, _ in flat]
dev = "cuda" if torch.cuda.is_available() else "cpu"; chars = sorted({ch for d in train_docs for _, t in d["items"] for ch in t}); c2i = {c: i + 1 for i, c in enumerate(chars)}
def blk(i, o, pool): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True)] + ([nn.MaxPool2d(pool)] if pool else [])
class CRNN(nn.Module):
    def __init__(s, nc):
        super().__init__(); s.cnn = nn.Sequential(*blk(1, 64, 2), *blk(64, 128, 2), *blk(128, 256, None), *blk(256, 256, (2, 1)), *blk(256, 512, None), *blk(512, 512, (2, 1)), nn.Conv2d(512, 512, (2, 1)), nn.ReLU(True))
        s.rnn = nn.LSTM(512, 256, 2, bidirectional=True, batch_first=True, dropout=0.2); s.out = nn.Linear(512, nc)
    def forward(s, x): f = s.cnn(x).squeeze(2).permute(0, 2, 1); o, _ = s.rnn(f); return s.out(o)
net = CRNN(len(chars) + 1); sd = torch.load(OUT / "crnn_ocr_state.pt", map_location="cpu"); res["crnn_state_matches_architecture_and_charset"] = bool(net.load_state_dict(sd) is not None and sd["out.weight"].shape[0] == len(chars) + 1); net.to(dev).eval()
def prep(c):
    w, h = c.size; nw = min(640, max(16, int(round(w * 32 / h)))); a = np.asarray(c.resize((nw, 32), Image.BILINEAR), np.float32) / 255.0; return (a - a.mean()) / (a.std() + 1e-3)
def crnn_read(c):
    with torch.no_grad(): lg = net(torch.from_numpy(prep(c))[None, None].to(dev))
    out, prev = [], 0
    for k in lg.argmax(-1)[0].cpu().numpy():
        if k != prev and k != 0: out.append(chars[k - 1])
        prev = k
    return "".join(out)
hyps["crnn"] = [crnn_read(c) for _, c, _ in flat]
tp = ROOT / "experiments/text_vs_neural/pretrained/trocr-small-printed"
if (tp / "model.safetensors").exists():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(str(tp)); tm = VisionEncoderDecoderModel.from_pretrained(str(tp)).to(dev).eval(); out = []
    for i in range(0, len(flat), 16):
        pv = proc(images=[c.convert("RGB") for _, c, _ in flat[i:i + 16]], return_tensors="pt").pixel_values.to(dev)
        with torch.no_grad(): ids = tm.generate(pv, max_new_tokens=64, num_beams=1)
        out += proc.batch_decode(ids, skip_special_tokens=True)
    hyps["trocr"] = out
# ---------- 4. compare ----------
def subset(name, k): return [(t, h) for (s, _, t), h in zip(flat, hyps[name]) if (k is None or s == k)]
cmp = []
for name in hyps:
    S = json.loads((OUT / f"ocr_{name}.json").read_text())
    for sub, key in (("all", None), ("sroie", "sroie"), ("funsd", "funsd")):
        pairs = subset(name, key); refs, hs = [p[0] for p in pairs], [p[1] for p in pairs]; c, w = cer_wer(refs, hs); norm = lambda s: s.casefold().replace(" ", ""); cn = cer_wer([norm(r) for r in refs], [norm(h) for h in hs])[0]; wc = cer_wer([r.casefold() for r in refs], [h.casefold() for h in hs])[1]
        for q, mine, stored in (("cer", c, S[sub]["cer"]), ("wer", w, S[sub]["wer"]), ("cer_normalised", cn, S[sub]["cer_normalised"]), ("wer_casefold", wc, S[sub]["wer_casefold"])):
            cmp.append(dict(model=name, subset=sub, metric=q, stored=stored, recomputed=mine, abs_diff=abs(stored - mine), ok=abs(stored - mine) < 1e-9))
    res.setdefault("lines_scored", {})[name] = len(hyps[name])
    # the stored file's own sample lines must be reproduced by the re-run
    first = {t: h for (_, _, t), h in zip(flat, hyps[name])}; res.setdefault("stored_samples_reproduced", {})[name] = sum(1 for s in S["samples"] if first.get(s["ref"]) == s["hyp"]) / max(len(S["samples"]), 1)
res["comparison"] = cmp; bad = [c for c in cmp if not c["ok"]]; print(f"OCR metrics: {len(cmp) - len(bad)}/{len(cmp)} reproduced exactly (|diff| < 1e-9); mismatches:", [(b['model'], b['subset'], b['metric'], round(b['stored'], 5), round(b['recomputed'], 5)) for b in bad][:10])
print("stored sample lines reproduced:", res["stored_samples_reproduced"]); json.dump(dict(res, hypotheses_saved_for_audit=True), open(VER / "ocr_check.json", "w"), indent=1, default=float)
json.dump({k: [dict(ref=t, hyp=h) for (_, _, t), h in zip(flat, v)] for k, v in hyps.items()}, open(VER / "ocr_hypotheses_reproduced.json", "w"), ensure_ascii=False)
