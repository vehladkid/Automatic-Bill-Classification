"""Model 3: LayoutLMv3-base (text + 2-D boxes + image) sequence classifier. Words/boxes from the stored OCR (GT or RapidOCR), reading-order sorted,
boxes normalised to 0-1000 by the ORIGINAL image size; first 512 tokens (document header carries the merchant). Image: processor 224x224.
Train-time augmentation is photometric only (geometric aug would misalign boxes). bf16 autocast, class-weighted CE, best epoch by VAL real macro-F1.
NOTE: microsoft/layoutlmv3-base weights are CC-BY-NC-SA-4.0 (non-commercial) - fine for this academic project, not for a commercial release."""
import random, time, sys
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from common_a import *
sys.path.insert(0, str(ROOT / "scripts")); from augment import augment
EP, BS, ACC, MAXLEN = 6, 2, 8, 448   # effective batch 16; larger micro-batches OOM on the 6 GB card
proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

class D(Dataset):
    def __init__(s, rows, train): s.rows, s.train = rows, train
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r = s.rows[i]; ws = words(r)[:MAXLEN]; W, H = max(r["w"] or 1, 1), max(r["h"] or 1, 1)
        if not ws: ws = [("empty", [0, 0, 1, 1])]
        boxes = [[int(np.clip(b[0] * 1000 / W, 0, 1000)), int(np.clip(b[1] * 1000 / H, 0, 1000)), int(np.clip(b[2] * 1000 / W, 0, 1000)), int(np.clip(b[3] * 1000 / H, 0, 1000))] for _, b in ws]
        boxes = [[min(a, c), min(b_, d), max(a, c), max(b_, d)] for a, b_, c, d in boxes]
        im = Image.open(r["cache"]).convert("RGB")
        if s.train: im = augment(im, random.Random(random.random()), p=0.6, geometric=False)
        enc = proc(im, [t for t, _ in ws], boxes=boxes, truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
        d = {k: v[0] for k, v in enc.items()}; d["labels"] = torch.tensor(r["y"]); return d

def collate(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}

@torch.no_grad()
def predict(model, rows, dev):
    model.eval(); out = []
    for b in DataLoader(D(rows, False), batch_size=8, num_workers=8, collate_fn=collate):
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.bfloat16): out.append(model(**b).logits.float().cpu())
    return torch.cat(out).numpy()

def real_f1(rows, logits):
    idx = [i for i, r in enumerate(rows) if not r["synthetic"]]; y = [rows[i]["y"] for i in idx]; return macro_f1(y, list(logits.argmax(1)[idx]), sorted(set(y)))

if __name__ == "__main__":
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED); dev = "cuda"
    tr, va, te = load("train"), load("val"), load("test")
    model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(LABELS)).to(dev)
    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]; body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": 4e-5}, {"params": head, "lr": 4e-4}], weight_decay=0.01)
    spe = len(tr) // (BS * ACC); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[4e-5, 4e-4], total_steps=EP * spe, pct_start=0.1, anneal_strategy="linear")
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(tr), dtype=torch.float32, device=dev), label_smoothing=0.05)
    sampler = WeightedRandomSampler([0.5 if r["synthetic"] else 1.0 for r in tr], num_samples=len(tr), replacement=True)
    dl = DataLoader(D(tr, True), batch_size=BS, sampler=sampler, num_workers=10, drop_last=True, collate_fn=collate, persistent_workers=True, pin_memory=True)
    best, bad, t0, log = -1, 0, time.time(), []
    for ep in range(EP):
        model.train(); tl = 0; n = 0
        for i, b in enumerate(dl):
            y = b.pop("labels").to(dev); b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = crit(model(**b).logits.float(), y) / ACC
            loss.backward(); tl += loss.item() * ACC; n += 1
            if (i + 1) % ACC == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        lv = predict(model, va, dev); f = real_f1(va, lv); acc = float(np.mean(lv.argmax(1) == np.array([r["y"] for r in va])))
        log.append(dict(epoch=ep + 1, train_loss=round(tl / n, 4), val_acc=round(acc, 4), val_real_macro_f1=round(f, 4), sec=round(time.time() - t0))); print(log[-1], flush=True)
        if f > best: best, bad = f, 0; torch.save(model.state_dict(), MODELS / "layoutlmv3.pt")
        else:
            bad += 1
            if bad >= 3: print("early stop"); break
    model.load_state_dict(torch.load(MODELS / "layoutlmv3.pt")); lv, lt = predict(model, va, dev), predict(model, te, dev)
    np.savez(MODELS / "logits_layoutlm.npz", val=lv, test=lt)
    rep = {"val": evaluate(va, softmax(lv)), "test": evaluate(te, softmax(lt)), "log": log, "config": dict(epochs=EP, batch=BS * ACC, max_len=MAXLEN, backbone="microsoft/layoutlmv3-base")}
    save_json(REPORTS / "layoutlmv3.json", rep)
    for k in ("val", "test"): print(k, {s: (rep[k][s]["accuracy"], rep[k][s]["macro_f1"]) for s in rep[k]})
