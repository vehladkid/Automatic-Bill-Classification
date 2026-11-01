"""Train-time augmentation for photographed/scanned bills. NOTHING is written to disk by the pipeline: import `augment`
in the training Dataset.__getitem__ and apply it ONLY to train-split samples (this is the leakage guarantee).
Deliberately excluded: horizontal/vertical flips (mirror text never occurs), hue shifts (logo colour is a class cue),
large rotations (>12 deg; use a separate orientation head), cutout over the whole total/amount region.
`python augment.py [N]` writes datasets/interim/aug_preview.jpg for eyeballing."""
import io, sys, random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from common import INTERIM, DS

def _persp(img, r, s=0.04):
    w, h = img.size; d = lambda m: r.uniform(-s, s) * m
    q = [(d(w), d(h)), (w + d(w), d(h)), (w + d(w), h + d(h)), (d(w), h + d(h))]
    return img.transform(img.size, Image.PERSPECTIVE, _inv([(0, 0), (w, 0), (w, h), (0, h)], q), Image.BICUBIC, fillcolor=(235, 235, 235))

def _inv(src, dst):  # PIL wants coeffs mapping dst(output) -> src(input)
    A = []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
    return np.linalg.solve(np.array(A, float), np.array(src, float).reshape(8))

def augment(img, rng=None, p=0.5, geometric=True):
    """geometric=False keeps pixel positions (rotate/perspective/crop off) so OCR boxes stay valid for layout models."""
    r = rng or random.Random(); img = img.convert("RGB")
    if geometric and r.random() < p: img = img.rotate(r.uniform(-8, 8), Image.BICUBIC, expand=True, fillcolor=(235, 235, 235))
    if geometric and r.random() < p * .6: img = _persp(img, r)
    if geometric and r.random() < p * .5:  # crop up to 6% per side
        w, h = img.size; l, t = int(w * r.uniform(0, .06)), int(h * r.uniform(0, .06)); img = img.crop((l, t, w - int(w * r.uniform(0, .06)), h - int(h * r.uniform(0, .06))))
    if r.random() < p: img = ImageEnhance.Brightness(img).enhance(r.uniform(.7, 1.25))
    if r.random() < p: img = ImageEnhance.Contrast(img).enhance(r.uniform(.7, 1.3))
    if r.random() < p * .5:  # soft shadow: horizontal gradient multiply
        a = np.asarray(img, np.float32); w = a.shape[1]; g = np.linspace(r.uniform(.6, 1), 1, w) if r.random() < .5 else np.linspace(1, r.uniform(.6, 1), w)
        img = Image.fromarray((a * g[None, :, None]).clip(0, 255).astype(np.uint8))
    if r.random() < p * .5: img = img.filter(ImageFilter.GaussianBlur(r.uniform(.4, 1.6)))
    if r.random() < p * .4:  # low-res simulation
        w, h = img.size; f = r.uniform(.4, .7); img = img.resize((max(32, int(w * f)), max(32, int(h * f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if r.random() < p * .4:
        a = np.asarray(img, np.float32); img = Image.fromarray((a + np.random.default_rng(r.randint(0, 10**9)).normal(0, r.uniform(3, 12), a.shape)).clip(0, 255).astype(np.uint8))
    if r.random() < p:  # JPEG artefacts
        b = io.BytesIO(); img.save(b, "JPEG", quality=r.randint(35, 85)); img = Image.open(b).convert("RGB")
    return img

if __name__ == "__main__":
    import json
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = [json.loads(l) for l in open(DS / "processed/classification/train.jsonl")]
    r = random.Random(0); pick = r.sample([x for x in rows if not x["synthetic"]], n)
    sheet = Image.new("RGB", (n * 260, 2 * 360), "white")
    for i, x in enumerate(pick):
        o = Image.open(DS / x["image"]).convert("RGB"); o.thumbnail((250, 350)); a = augment(o, r, p=1.0); a.thumbnail((250, 350))
        sheet.paste(o, (i * 260, 0)); sheet.paste(a, (i * 260, 360))
    sheet.save(INTERIM / "aug_preview.jpg", quality=85); print("wrote", INTERIM / "aug_preview.jpg")
