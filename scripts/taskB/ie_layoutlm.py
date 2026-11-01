"""Model B2: LayoutLMv3-base token classification (BIO tags for every Task B field) on OCR words + boxes + page image.
Long documents use sliding windows (384 subtokens, stride 96); first-subtoken labelling. Saves best-by-val-token-F1 weights and per-word tag
probabilities for val/test (merged over windows by max) for decoding/calibration. Micro-batch 2 x accumulation 8 (6 GB card)."""
import sys, time, random, json
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ImageProcessor, LayoutLMv3ForTokenClassification
from schema import *
from data import load
ML, ST, EP, BS, ACC = 384, 96, 5, 2, 8
TAGS = ["O"] + [f"{p}-{f}" for f in FIELDS for p in ("B", "I")]; T2I = {t: i for i, t in enumerate(TAGS)}
CACHE = DS / "interim/taskB_cache"; MODELS = ROOT / "models/taskB"
tok = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base", add_prefix_space=True); imp = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def norm_boxes(d):
    W, H = max(d["width"], 1), max(d["height"], 1); out = []
    for b in d["boxes"]:
        x0, y0, x1, y1 = (int(np.clip(b[0] * 1000 / W, 0, 1000)), int(np.clip(b[1] * 1000 / H, 0, 1000)), int(np.clip(b[2] * 1000 / W, 0, 1000)), int(np.clip(b[3] * 1000 / H, 0, 1000)))
        out.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
    return out

def windows(d, with_labels=True):
    """-> list of dicts(tensors) one per window + word_ids per window"""
    words = d["words"] or ["empty"]; boxes = norm_boxes(d) if d["words"] else [[0, 0, 1, 1]]
    labels = [T2I[t] for t in d["labels"]] if (with_labels and d["words"]) else None
    enc = tok(words, boxes=boxes, word_labels=labels, truncation=True, padding="max_length", max_length=ML, stride=ST, return_overflowing_tokens=True)
    pv = imp((d["_image"] if "_image" in d else Image.open(CACHE / f"{d['document_id']}.jpg")).convert("RGB"), return_tensors="pt")["pixel_values"][0]   # _image: inference on an arbitrary page
    out = []
    for k in range(len(enc["input_ids"])):
        w = dict(input_ids=torch.tensor(enc["input_ids"][k]), attention_mask=torch.tensor(enc["attention_mask"][k]), bbox=torch.tensor(enc["bbox"][k]), pixel_values=pv)
        if labels is not None: w["labels"] = torch.tensor(enc["labels"][k])
        out.append((w, enc.word_ids(k)))
    return out

class DS_(Dataset):
    def __init__(s, docs): s.items = [(i, d) for i, d in enumerate(docs)]; s.docs = docs
    def __len__(s): return len(s.docs)
    def __getitem__(s, i): return [w for w, _ in windows(s.docs[i])]
def collate(batch):
    ws = [w for b in batch for w in b]; return {k: torch.stack([w[k] for w in ws]) for k in ws[0]}

@torch.no_grad()
def predict_probs(model, docs, dev):
    """list of [n_words, n_tags] float16 arrays (max over overlapping windows)"""
    model.eval(); res = []
    for d in docs:
        ws = windows(d, with_labels=False); n = max(len(d["words"]), 1); P = np.zeros((n, len(TAGS)), np.float32)
        for w, wid in ws:
            b = {k: v[None].to(dev) for k, v in w.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(**b).logits.float()[0]
            pr = torch.softmax(lg, -1).cpu().numpy(); seen = set()
            for t, wi in enumerate(wid):
                if wi is not None and wi not in seen: seen.add(wi); P[wi] = np.maximum(P[wi], pr[t])
        res.append(P.astype(np.float16))
    return res

def token_f1(model, docs, dev):
    P = predict_probs(model, docs, dev); tp = fp = fn = 0
    for d, p in zip(docs, P):
        if not d["words"]: continue
        pred = p.argmax(1); gold = np.array([T2I[t] for t in d["labels"]])
        tp += int(((pred == gold) & (gold > 0)).sum()); fp += int(((pred != gold) & (pred > 0)).sum()); fn += int(((pred != gold) & (gold > 0)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); return 2 * pr * rc / max(pr + rc, 1e-9), P

if __name__ == "__main__":
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED); dev = "cuda"
    tr, va, te = [d for d in load("train") if d["words"]], [d for d in load("val") if d["words"]], [d for d in load("test") if d["words"]]
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(TAGS)).to(dev)
    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]; body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": 5e-5}, {"params": head, "lr": 5e-4}], weight_decay=0.01)
    steps = EP * (len(tr) // (BS * ACC)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-5, 5e-4], total_steps=steps + 5, pct_start=0.1, anneal_strategy="linear")
    w = torch.ones(len(TAGS), device=dev); w[0] = 0.5; crit = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    dl = DataLoader(DS_(tr), batch_size=BS, shuffle=True, num_workers=8, collate_fn=collate, drop_last=True, persistent_workers=True, pin_memory=True)
    best, bad, t0, log = -1, 0, time.time(), []
    for ep in range(EP):
        model.train(); tl = 0; n = 0
        for i, b in enumerate(dl):
            y = b.pop("labels"); nw = y.shape[0]; chunks = list(range(0, nw, 2))
            for c in chunks:      # windows of one batch in chunks of 2 (memory)
                bb = {k: v[c:c + 2].to(dev, non_blocking=True) for k, v in b.items()}; yy = y[c:c + 2].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(**bb).logits.float()
                loss = crit(lg.view(-1, len(TAGS)), yy.view(-1)) / (ACC * len(chunks)); loss.backward(); tl += loss.item() * ACC
            n += 1
            if (i + 1) % ACC == 0: nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        f, _ = token_f1(model, va, dev); log.append(dict(epoch=ep + 1, train_loss=round(tl / n, 4), val_token_f1=round(f, 4), sec=round(time.time() - t0))); print(log[-1], flush=True)
        if f > best: best, bad = f, 0; torch.save(model.state_dict(), MODELS / "layoutlmv3_ie.pt")
        else:
            bad += 1
            if bad >= 2: print("early stop"); break
    model.load_state_dict(torch.load(MODELS / "layoutlmv3_ie.pt")); _, Pv = token_f1(model, va, dev); _, Pt = token_f1(model, te, dev)
    import pickle; pickle.dump({"val": {d["document_id"]: p for d, p in zip(va, Pv)}, "test": {d["document_id"]: p for d, p in zip(te, Pt)}, "tags": TAGS, "log": log}, open(MODELS / "ie_probs.pkl", "wb"))
    print("DONE", log[-1], flush=True)
