# Audit probe (2026-10-27): measures sequential latency, BillService.LOCK serialisation, and UNGUARDED component-level concurrency (1/2/4 threads) with HRBC_OUT in a temp dir. Run: cd app && ../.venv/bin/python ../reports/email_ingestion/smoke/concurrency_probe.py  (stop the Streamlit app first; 4-thread step is EXPECTED to hit CUDA OOM on a 6 GB GPU)
import os, sys, time, json, threading, subprocess, statistics, tempfile
SP = os.path.dirname(os.path.abspath(__file__)); os.environ["HRBC_OUT"] = tempfile.mkdtemp(prefix="benchout_")
os.environ["HRBC_LLM_MODE"] = "off"
R = "/home/tejvir-singh/Desktop/ML-project"; sys.path.insert(0, R + "/app")
import service as SV
from pathlib import Path
docs = [p for p in sorted(Path(R, "demo/sample_bills").iterdir()) if p.suffix.lower() in (".png", ".jpg", ".pdf") and "blank" not in p.name]
def gpu(): return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip())
peak = {"v": 0}; stop = {"v": False}
def poll():
    while not stop["v"]: peak["v"] = max(peak["v"], gpu()); time.sleep(0.2)
base = gpu(); threading.Thread(target=poll, daemon=True).start()
t0 = time.time(); svc = SV.BillService().load(); print("load_s", round(time.time() - t0, 1), "errors", svc.errors, "gpu_after_load_MiB", gpu(), "baseline_MiB", base, flush=True)
# stage timers (wrappers only inside this benchmark process)
st = {"ocr": [], "A": [], "B": []}; tl = threading.local()
def wrap(obj, name, key):
    f = getattr(obj, name)
    def g(*a, **k):
        t = time.time(); r = f(*a, **k); st[key].append(time.time() - t); return r
    setattr(obj, name, g)
wrap(svc, "_ocr", "ocr"); wrap(svc.A, "predict", "A"); wrap(svc.B, "run", "B")
data = [(p.name, p.read_bytes()) for p in docs]
svc.process(data[0][1], data[0][0], owner="E1"); [v.clear() for v in st.values()]   # warm-up
lat = []
t0 = time.time()
for n, b in data:
    t = time.time(); r = svc.process(b, n, owner="E1"); lat.append(time.time() - t)
wall = time.time() - t0
print("SEQ docs", len(data), "wall_s", round(wall, 1), "docs/min", round(60 * len(data) / wall, 1), "lat_mean", round(statistics.mean(lat), 2), "median", round(statistics.median(lat), 2), "max", round(max(lat), 2))
print("stage means s:", {k: (round(statistics.mean(v), 2), len(v)) for k, v in st.items()}, flush=True)
# 2 threads calling the REAL process(): does BillService.LOCK serialise?
[v.clear() for v in st.values()]; res = []
def run(chunk):
    for n, b in chunk: t = time.time(); svc.process(b, n, owner="E2"); res.append(time.time() - t)
t0 = time.time(); th = [threading.Thread(target=run, args=(data[i::2],)) for i in range(2)]; [x.start() for x in th]; [x.join() for x in th]; w2 = time.time() - t0
print("2 THREADS via process() wall_s", round(w2, 1), "docs/min", round(60 * len(data) / w2, 1), "(LOCK =", type(SV.LOCK).__name__, ")", flush=True)
# unguarded component-level experiment (scratch only): OCR + A + B per doc without the service LOCK, 1/2/4 threads
from PIL import Image
import numpy as np
def one(b, n):
    d = Path(tempfile.mkdtemp()); p = d / n; p.write_bytes(b)
    if p.suffix == ".pdf": return
    segs = svc._ocr(p); txt, ws = svc._text_from_segments(segs, 0)
    if not ws: return
    svc.A.predict(str(p), words=[(s["text"], s["bbox"]) for s in segs])
    W, H = Image.open(p).size
    doc = dict(document_id="x", words=[w["text"] for w in ws], boxes=[w["bbox"] for w in ws], width=W, height=H, country="?", supported_fields=list(svc.mB.FIELDS), fields={}, labels=[])
    svc.B.run(doc, Image.open(p).convert("RGB"))
imgs = [(n, b) for n, b in data if not n.endswith(".pdf")]
for T in (1, 2, 4):
    err = []; peak["v"] = 0
    def w(chunk):
        for n, b in chunk:
            try: one(b, n)
            except Exception as e: err.append(f"{type(e).__name__}: {str(e)[:100]}")
    t0 = time.time(); th = [threading.Thread(target=w, args=(imgs[i::T],)) for i in range(T)]; [x.start() for x in th]; [x.join() for x in th]; ww = time.time() - t0
    print(f"UNGUARDED threads={T} docs={len(imgs)} wall_s={ww:.1f} docs/min={60*len(imgs)/ww:.1f} peak_gpu_MiB={peak['v']} errors={len(err)} {set(err)}", flush=True)
stop["v"] = True
