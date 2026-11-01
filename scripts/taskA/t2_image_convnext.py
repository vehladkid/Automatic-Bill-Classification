"""Model 2: ConvNeXt-Tiny (ImageNet-22k->1k, timm) fine-tuned on the page image (letterboxed 384x384). bf16 autocast on the dGPU.
Train-only augmentation (scripts/augment.py), class-weighted CE + label smoothing, real docs sampled 2x as often as synthetic.
Best epoch by VAL real-only macro-F1. Saves logits for fusion/temperature scaling."""
import random, time, sys
import numpy as np, torch, torch.nn as nn, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from common_a import *
sys.path.insert(0, str(ROOT / "scripts")); from augment import augment
SZ, EP, BS = 384, 14, 24
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

def letterbox(im):
    s = SZ / max(im.size); im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    c = Image.new("RGB", (SZ, SZ), (255, 255, 255)); c.paste(im, ((SZ - im.width) // 2, (SZ - im.height) // 2)); return c

class DS_(Dataset):
    def __init__(s, rows, train): s.rows, s.train, s.rng = rows, train, random.Random(SEED)
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r = s.rows[i]; im = Image.open(r["cache"]).convert("RGB")
        if s.train: im = augment(im, random.Random(random.random()), p=0.6)
        a = (np.asarray(letterbox(im), np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(a.transpose(2, 0, 1)), r["y"]

@torch.no_grad()
def predict(model, rows, dev):
    model.eval(); dl = DataLoader(DS_(rows, False), batch_size=48, num_workers=8); out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16): out.append(model(x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)).float().cpu())
    return torch.cat(out).numpy()

def real_f1(rows, logits):
    idx = [i for i, r in enumerate(rows) if not r["synthetic"]]; y = [rows[i]["y"] for i in idx]; p = logits.argmax(1)[idx]
    return macro_f1(y, list(p), sorted(set(y)))

if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    tr, va, te = load("train"), load("val"), load("test"); dev = "cuda"
    model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, pretrained_cfg_overlay=dict(file=str(MODELS.parent / "pretrained/convnext_tiny_in22k.safetensors")), num_classes=len(LABELS), drop_path_rate=0.1).to(dev).to(memory_format=torch.channels_last)
    head = [p for n, p in model.named_parameters() if n.startswith("head.fc")]; body = [p for n, p in model.named_parameters() if not n.startswith("head.fc")]
    opt = torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
    steps = EP * (len(tr) // BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.1, anneal_strategy="cos")
    cw = torch.tensor(class_weights(tr), dtype=torch.float32, device=dev); crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    sw = [0.5 if r["synthetic"] else 1.0 for r in tr]; sampler = WeightedRandomSampler(sw, num_samples=len(tr), replacement=True)
    dl = DataLoader(DS_(tr, True), batch_size=BS, sampler=sampler, num_workers=10, drop_last=True, persistent_workers=True, pin_memory=True)
    best, bad, t0 = -1, 0, time.time(); log = []
    for ep in range(EP):
        model.train(); tl = 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item()
        lv = predict(model, va, dev); f = real_f1(va, lv); acc = float(np.mean(lv.argmax(1) == np.array([r["y"] for r in va])))
        log.append(dict(epoch=ep + 1, train_loss=round(tl / len(dl), 4), val_acc=round(acc, 4), val_real_macro_f1=round(f, 4), sec=round(time.time() - t0)))
        print(log[-1], flush=True)
        if f > best: best, bad = f, 0; torch.save(model.state_dict(), MODELS / "convnext_tiny.pt")
        else:
            bad += 1
            if bad >= 4: print("early stop"); break
    model.load_state_dict(torch.load(MODELS / "convnext_tiny.pt")); lv, lt = predict(model, va, dev), predict(model, te, dev)
    np.savez(MODELS / "logits_image.npz", val=lv, test=lt)
    rep = {"val": evaluate(va, softmax(lv)), "test": evaluate(te, softmax(lt)), "log": log, "config": dict(size=SZ, epochs=EP, batch=BS, backbone="convnext_tiny.fb_in22k_ft_in1k")}
    save_json(REPORTS / "image_convnext.json", rep)
    for k in ("val", "test"): print(k, {s: (rep[k][s]["accuracy"], rep[k][s]["macro_f1"]) for s in rep[k]})
