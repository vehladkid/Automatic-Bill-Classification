"""Annotation helper: renders evidence crops (full page, header, footer/totals, HSN column) of one pool page into a scratch dir and prints the SUGGESTION_ONLY candidates + OCR-derived HSN column codes.
Used by the annotator to read the real image; it does not decide anything."""
import sys, json, re
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_pool_analysis import lines_of
ROOT = Path(__file__).resolve().parents[2]; POOL = ROOT / "datasets/interim/v2_pool"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/claude-1000/-home-tejvir-singh-Desktop-ML-project/7de79d50-1400-4e56-8937-5c33f57d2810/scratchpad/view")

def hsn_column(o):
    ws = o["words"]; H = o["height"]
    hs = [w for w in ws if re.fullmatch(r"(HSN|SAC|HSN/SAC|HSNCODE|HSN/?SACCODE|HSNSAC)\W*", w["text"].upper().replace(" ", "")) and w["bbox"][1] < 0.6 * H]
    if not hs: return None, []
    h = sorted(hs, key=lambda w: w["bbox"][1])[0]; cx = (h["bbox"][0] + h["bbox"][2]) / 2
    col = [w for w in ws if abs((w["bbox"][0] + w["bbox"][2]) / 2 - cx) < max(60, (h["bbox"][2] - h["bbox"][0])) and w["bbox"][1] > h["bbox"][3] and re.fullmatch(r"\d{4,8}", re.sub(r"\D", "", w["text"]) or "x")]
    return h, col

def main(d):
    o = json.loads((POOL / "ocr" / f"{d}.json").read_text()); im = Image.open(POOL / "images" / f"{d}.jpg"); W, H = im.size; OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.jpg"): f.unlink()
    def save(name, box, maxw):
        c = im.crop(box); r = min(maxw / c.width, 2.0); c = c.resize((int(c.width * r), int(c.height * r))); c.save(OUT / name, quality=88)
    full = im.copy(); full.thumbnail((1000, 1300)); full.save(OUT / "1_full.jpg", quality=85)
    xs0 = min(w["bbox"][0] for w in o["words"]); xs1 = max(w["bbox"][2] for w in o["words"]); ys0 = min(w["bbox"][1] for w in o["words"]); ys1 = max(w["bbox"][3] for w in o["words"]); ext = ys1 - ys0
    X0, X1 = max(xs0 - 30, 0), min(xs1 + 30, W)
    save("2_header.jpg", (X0, max(ys0 - 30, 0), X1, int(ys0 + 0.27 * ext)), 1800); save("3_footer.jpg", (X0, int(ys0 + 0.58 * ext), X1, min(ys1 + 30, H)), 1800)
    h, col = hsn_column(o)
    if h:
        x0 = max(int(h["bbox"][0] - 60), 0); x1 = min(int(h["bbox"][2] + 160), W); y0 = int(h["bbox"][1] - 10)
        yl = max([w["bbox"][3] for w in col] or [y0 + int(0.2 * H)]); save("4_hsn_col.jpg", (x0, y0, x1, min(H, int(yl + 60))), 420)
    if h and col:
        yl = max(w["bbox"][3] for w in col); save("5_tabletotal.jpg", (X0, int(yl - 30), X1, int(yl + 0.075 * ext)), 1800)
    codes = [re.sub(r"\D", "", w["text"]) for w in col]
    print(d, "size", W, H, "| HSN header", bool(h), "| OCR column codes:", codes)
    rev = {r["document_id"]: r for r in json.load(open(POOL / "review_candidates.json"))}[d]
    print("supplier cand:", rev["supplier_candidate"])
    for k, v in rev["candidates"].items():
        if v and k != "hsn_sac": print(" ", k, [(x["value"], x["bbox"]) for x in v[:3]])
    print(" ocr header:", " | ".join(" ".join(w["text"] for w in l) for l in lines_of(o["words"])[:10])[:700])
if __name__ == "__main__": main(sys.argv[1])
