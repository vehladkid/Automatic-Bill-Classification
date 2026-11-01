"""Label audit from group-OOF text-model probabilities (train+val, real docs): list documents where the model confidently disagrees with the label,
excluding classes whose real data is tiny. Writes reports/taskA/label_review_candidates.csv and contact sheets in reports/taskA/audit/ for visual review."""
import csv
import numpy as np
from PIL import Image, ImageDraw
from common_a import *
if __name__ == "__main__":
    z = np.load(MODELS / "oof_text.npz"); P = dict(zip(z["ids"], z["probs"])); rows = load("train") + load("val"); out = []
    for r in rows:
        if r["synthetic"]: continue
        p = P[r["id"]]; k = int(p.argmax())
        if k != r["y"] and p[k] >= 0.6: out.append((float(p[k] - p[r["y"]]), r, k, float(p[k])))
    out.sort(key=lambda t: -t[0]); (REPORTS / "audit").mkdir(exist_ok=True)
    with open(REPORTS / "label_review_candidates.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank", "document_id", "source", "label", "model_says", "model_conf", "image"])
        for i, (_, r, k, c) in enumerate(out): w.writerow([i, r["id"], r["source"], r["label"], LABELS[k], round(c, 3), r["cache"]])
    print("candidates:", len(out), "by source:", {s: sum(1 for _, r, *_ in out if r["source"] == s) for s in sorted({r["source"] for _, r, *_ in out})})
    for s in range(0, min(len(out), 96), 12):
        sheet = Image.new("RGB", (6 * 330, 2 * 440), "white"); d = ImageDraw.Draw(sheet)
        for j, (_, r, k, c) in enumerate(out[s:s + 12]):
            try: im = Image.open(r["cache"]).convert("RGB"); im.thumbnail((320, 400)); sheet.paste(im, ((j % 6) * 330, (j // 6) * 440 + 30))
            except Exception: pass
            d.text(((j % 6) * 330 + 4, (j // 6) * 440 + 2), f"#{s+j} {r['source'][:10]} L:{r['label']} M:{LABELS[k]}", fill=(200, 0, 0))
        sheet.save(REPORTS / "audit" / f"sheet_{s // 12:02d}.jpg", quality=70)
