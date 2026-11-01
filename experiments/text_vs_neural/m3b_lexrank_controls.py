"""Control for Model 3: is LexRank's choice of lines useful, or does any equally short text do as well?  Same units (OCR lines), same classifier (TF-IDF + LR, C from the project), same splits.
Controls at 50% and 25% of the lines: RANDOM lines (seeded; mean of 3 seeds) and the FIRST lines (head).  Results are appended to outputs/m3_lexrank.json under 'controls'."""
import sys, json, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "taskA"))
from exp_common import *
import t1_text_baseline as T1
from common_a import words as row_words
tr, va, te = splits(); C = json.loads((ROOT / "reports/taskA/text_baseline.json").read_text())["C"]
def lines_of(row):
    ws = row_words(row)
    if ws:
        hh = max(row["h"] or 1, 1); bucket = max(hh // 120, 8); out = {}
        for t, b in ws: out.setdefault(b[1] // bucket, []).append(t)
        L = [" ".join(v) for _, v in sorted(out.items())]
        if len(L) >= 2: return L
    w = row["text"].split(); return [" ".join(w[i:i + 8]) for i in range(0, len(w), 8)] or [row["text"]]
for rows in (tr, va, te):
    for r in rows: r["_lines"] = lines_of(r)
def pick(row, frac, mode, rng):
    L = row["_lines"]; k = max(1, int(round(frac * len(L))))
    if k >= len(L): return " ".join(L)
    idx = sorted(rng.choice(len(L), k, replace=False)) if mode == "random" else list(range(k)); return " ".join(L[i] for i in idx)
lg = lambda p: np.log(np.clip(p, 1e-6, 1)); full = float(np.mean([len(r["text"].split()) for r in te])); out = []
for frac in (0.5, 0.25):
    for mode, seeds in (("random", (0, 1, 2)), ("first lines", (0,))):
        accs, f1s, words = [], [], []
        for s in seeds:
            rng = np.random.default_rng(s); b = lambda rows: [dict(r, text=pick(r, frac, "random" if mode == "random" else "head", rng)) for r in rows]
            trs, vas, tes = b(tr), b(va), b(te); vm = T1.fit(trs, C); pv, pt = T1.proba(vm, vas), T1.proba(vm, tes); res = evaluate_model(f"{mode} {int(frac * 100)}%", lg(pv), lg(pt), va, te)
            accs.append(res["calibrated"]["real"]["accuracy"]); f1s.append(res["calibrated"]["real"]["macro_f1"]); words.append(float(np.mean([len(r["text"].split()) for r in tes])))
        out.append(dict(config=f"{mode} {int(frac * 100)}% of lines", frac=frac, mode=mode, n_seeds=len(seeds), words_per_doc=float(np.mean(words)), reduction_pct=100 * (1 - float(np.mean(words)) / full), accuracy_real=float(np.mean(accs)), accuracy_sd=float(np.std(accs)),
                        macro_f1_real=float(np.mean(f1s)), macro_f1_sd=float(np.std(f1s))))
        print(out[-1], flush=True)
p = OUT / "m3_lexrank.json"; d = json.loads(p.read_text()); d["controls"] = out; p.write_text(json.dumps(d, indent=1, default=float))
