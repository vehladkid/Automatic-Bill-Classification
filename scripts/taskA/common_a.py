"""Task A (expense category) shared code: data loading, metrics, calibration helpers. No torch import here (usable by the sklearn baseline)."""
import sys, json, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, DS, META, PROCESSED, CATEGORIES, SEED

LABELS = sorted(c for c in CATEGORIES)            # 16 classes incl. rent and non_expense
L2I = {l: i for i, l in enumerate(LABELS)}
MODELS, REPORTS = ROOT / "models/taskA", ROOT / "reports/taskA"
CACHE = DS / "interim/taskA_cache"                # resized copies of every image (real files on disk), max side 768
SPLITS = {"train": "train", "val": "validation", "test": "test"}

def load(split):
    """Rows of one split joined with manifest fields (group_id, size, country, merchant, source)."""
    man = {}
    for l in open(META / "manifest.jsonl"):
        r = json.loads(l); man[r["document_id"]] = r
    out = []
    for l in open(PROCESSED / "classification" / f"{SPLITS[split]}.jsonl"):
        r = json.loads(l); m = man[r["document_id"]]
        out.append(dict(id=r["document_id"], image=str(DS / r["image"]), text=r["ocr_text"], label=r["label"], y=L2I[r["label"]], synthetic=r["synthetic"],
                        source=r["source"], ann=(DS / r["ann_path"]) if r["ann_path"] else None, group=m["group_id"], w=m["width"], h=m["height"],
                        country=m["country"], merchant=m.get("merchant"), cache=str(CACHE / f"{r['document_id']}.jpg")))
    return out

def words(row):
    """OCR words [(text, [x0,y0,x1,y1])] in reading order (line-bucketed), original-image pixel coordinates."""
    if not row["ann"] or not Path(row["ann"]).exists(): return []
    ws = json.loads(Path(row["ann"]).read_text()).get("words", [])
    ws = [w for w in ws if w.get("text", "").strip() and len(w.get("bbox", [])) == 4]
    hh = max(row["h"] or 1, 1); bucket = max(hh // 120, 8)
    return sorted(((w["text"], w["bbox"]) for w in ws), key=lambda t: (t[1][1] // bucket, t[1][0]))

# ---------- metrics ----------
def softmax(z, axis=-1): z = z - z.max(axis=axis, keepdims=True); e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)

def macro_f1(y, p, labels=None):
    labels = labels if labels is not None else sorted(set(y) | set(p)); f = []
    for c in labels:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c); fp = sum(1 for a, b in zip(y, p) if a != c and b == c); fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else 0.0

def per_class(y, p):
    rows = {}
    for c in sorted(set(y)):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c); fp = sum(1 for a, b in zip(y, p) if a != c and b == c); fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        rows[LABELS[c]] = dict(n=tp + fn, precision=round(pr, 3), recall=round(rc, 3), f1=round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0)
    return rows

def ece(probs, y, bins=10):
    conf = probs.max(1); pred = probs.argmax(1); ok = (pred == np.asarray(y)); e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)

def nll(probs, y): return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1))))

def bootstrap_ci(rows, probs, fn, n=300, seed=SEED):
    """95% CI of fn(y, pred) resampling GROUPS (documents of one template/merchant are not independent)."""
    rng = np.random.default_rng(seed); gs = {}
    for i, r in enumerate(rows): gs.setdefault(r["group"], []).append(i)
    keys = list(gs); y = np.array([r["y"] for r in rows]); pred = probs.argmax(1); vals = []
    for _ in range(n):
        idx = [i for k in rng.choice(len(keys), len(keys)) for i in gs[keys[k]]]
        vals.append(fn(list(y[idx]), list(pred[idx])))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]

def evaluate(rows, probs, name=""):
    """Full report dict: overall / real-only / synthetic-only / Indian-real-only, with group-bootstrap CI for macro-F1."""
    y = [r["y"] for r in rows]; pred = list(probs.argmax(1)); rep = {}
    subsets = {"all": np.ones(len(rows), bool), "real": np.array([not r["synthetic"] for r in rows]), "synthetic": np.array([r["synthetic"] for r in rows]),
               "real_indian": np.array([(not r["synthetic"]) and r["country"] == "IN" for r in rows])}
    for k, m in subsets.items():
        if m.sum() == 0: continue
        idx = np.where(m)[0]; ys = [y[i] for i in idx]; ps = [pred[i] for i in idx]; sub = [rows[i] for i in idx]
        rep[k] = dict(n=int(m.sum()), accuracy=round(float(np.mean([a == b for a, b in zip(ys, ps)])), 4), macro_f1=round(macro_f1(ys, ps, sorted(set(ys))), 4),
                      macro_f1_ci95=bootstrap_ci(sub, probs[idx], lambda a, b: macro_f1(a, b, sorted(set(a)))), ece=round(ece(probs[idx], ys), 4), nll=round(nll(probs[idx], ys), 4),
                      per_class=per_class(ys, ps))
    return rep

def confusion(y, pred):
    n = len(LABELS); m = np.zeros((n, n), int)
    for a, b in zip(y, pred): m[a, b] += 1
    return m

def class_weights(rows, power=0.5):
    cnt = np.bincount([r["y"] for r in rows], minlength=len(LABELS)).astype(float); cnt[cnt == 0] = 1
    w = (cnt.sum() / (len(cnt) * cnt)) ** power; return w / w.mean()

def save_json(p, obj): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))
