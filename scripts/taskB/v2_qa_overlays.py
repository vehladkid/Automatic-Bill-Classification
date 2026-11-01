"""Draw annotation boxes (field-name labels only, no values) on the QA pages -> qa/bbox_pages/pXX/overlay_*.jpg"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
POOL = Path(__file__).resolve().parents[2] / "datasets/interim/v2_pool"; Q = POOL / "qa"
mp = json.load(open(Q / "page_map_PRIVATE.json")); A = {a["document_id"]: a for a in json.load(open(POOL / "annot/annotations_v2_tierA.json"))}
COL = {'merchant_name':(255,0,255),'gstin':(0,200,255),'invoice_number':(230,200,0),'invoice_date':(255,140,0),'taxable_value':(0,200,0),'discount':(120,60,200),'round_off':(150,150,150),'tax_total':(160,0,200),'cgst':(0,140,0),'sgst':(0,100,200),'hsn_sac':(255,0,0),'total_amount':(255,0,0)}
for pid, m in mp.items():
    a = A[m["document_id"]]; im = Image.open(POOL / "images" / f"{m['document_id']}.jpg").convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
    try: ft = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)
    except Exception: ft = None
    boxes = {}
    for f, r in a["fields"].items():
        bb = r.get("bbox")
        if not bb: continue
        bl = bb if isinstance(bb[0], list) else [bb]
        for k, b in enumerate(bl):
            c = COL.get(f, (0, 0, 255)); d.rectangle(b, outline=c, width=6); d.text((b[0], max(b[1] - 40, 0)), f + (str(k + 1) if len(bl) > 1 else ""), fill=c, font=ft)
        boxes[f] = len(bl)
    pd = Q / "bbox_pages" / pid; pd.mkdir(parents=True, exist_ok=True); im.save(pd / "overlay_full_res.jpg", quality=90)
    t = im.copy(); t.thumbnail((1200, 1500)); t.save(pd / "overlay_overview.jpg", quality=88)
    for name, (x, y) in {"top": (0, .34), "middle": (.28, .68), "bottom": (.6, 1.0)}.items():
        c = im.crop((0, int(x * H), W, int(y * H))); r = min(2000 / c.width, 1.6); c.resize((int(c.width * r), int(c.height * r))).save(pd / f"overlay_{name}.jpg", quality=88)
    json.dump({"fields_with_box": boxes, "fields_without_box": [f for f, r in a["fields"].items() if f not in boxes and r.get("value")]}, open(pd / "boxes_present.json", "w"))
