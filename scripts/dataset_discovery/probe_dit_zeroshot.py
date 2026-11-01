"""ZERO-SHOT probe of microsoft/dit-base-finetuned-rvlcdip (MIT, 16 RVL-CDIP classes) on the project's own REAL documents. No training, no tuning, TEST split never used.
For up to N documents per document_type (train+val splits, fixed seed, images only) it records what the model says. This answers: can an off-the-shelf RVL-CDIP classifier recognise OUR document types?
(Its label space has only 'invoice' in common with ours; everything else would need fine-tuning.) Also records latency and peak GPU memory. Output: reports/doc_type/dit_zeroshot_v1.json."""
import sys, json, random, time, collections
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
ROOT = Path(__file__).resolve().parents[2]; DS = ROOT / "datasets"; M = ROOT / "models/pretrained/dit-base-finetuned-rvlcdip"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rows = [json.loads(l) for l in open(DS / "metadata/doc_type_manifest_v1.jsonl")]; rows = [r for r in rows if not r["synthetic"] and r["split"] in ("train", "val") and not r["file_path"].lower().endswith(".pdf")]
by = collections.defaultdict(list)
for r in rows: by[r["doc_type"]].append(r)
rnd = random.Random(0); sample = {t: rnd.sample(v, min(N, len(v))) for t, v in sorted(by.items())}
proc = AutoImageProcessor.from_pretrained(M); dev = "cuda" if torch.cuda.is_available() else "cpu"; model = AutoModelForImageClassification.from_pretrained(M, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval(); id2 = model.config.id2label
torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None; res = {}; times = []
for t, rs in sample.items():
    c = collections.Counter(); inv = 0; n = 0
    for r in rs:
        try: im = Image.open(DS / r["file_path"]).convert("RGB")
        except Exception: continue
        x = proc(im, return_tensors="pt")["pixel_values"].to(dev).to(model.dtype); t0 = time.time()
        with torch.no_grad(): lg = model(pixel_values=x).logits[0]
        times.append(time.time() - t0); c[id2[int(lg.argmax())]] += 1; n += 1
    res[t] = dict(n=n, predicted_rvlcdip_class=dict(c.most_common()), share_invoice=round(c.get("invoice", 0) / max(n, 1), 3))
out = dict(model="microsoft/dit-base-finetuned-rvlcdip", device=dev, docs_per_class_max=N, per_doc_type=res, latency_ms_mean=round(1000 * sum(times) / len(times), 1), peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 1e6) if dev == "cuda" else None, n_images=len(times),
           caveat="zero-shot: RVL-CDIP classes (letter, form, email, handwritten, advertisement, scientific report/publication, specification, file folder, news article, budget, invoice, presentation, questionnaire, resume, memo) are not our label space")
(ROOT / "reports/doc_type/dit_zeroshot_v1.json").write_text(json.dumps(out, indent=1)); print(f"images {len(times)} | latency {out['latency_ms_mean']} ms | peak GPU {out['peak_gpu_mb']} MB")
for t, v in res.items(): print(f"  {t:18s} n={v['n']:3d} invoice-share={v['share_invoice']:.2f}  top: {list(v['predicted_rvlcdip_class'].items())[:3]}")
