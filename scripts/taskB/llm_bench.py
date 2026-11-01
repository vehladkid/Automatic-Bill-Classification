"""Controlled LLM benchmark on the frozen Task B REAL test documents (read-only on datasets/ and models/; writes reports/taskB/llm/ only).
Same documents, same OCR words, same scorer (evaluate.py) for every system. Systems:
  v1                 frozen pipeline (hybrid + gating) = reports/taskB/all_predictions.pkl
  text_*             v1 + text-only LLM (OCR lines + boxes, no image)       [separate cache]
  vlm_*              v1 + vision-language LLM (OCR lines + boxes + page image)
  *_raw              LLM output used as-is (no validation)      *_validated  grounded values only      *_fused  evidence-based fusion with v1 (what the app would use)
Slices: real / real_indian / source dataset / low-resolution / multi-page(if any).  Paired bootstrap (docs resampled) for fused - v1 micro-F1.
Fusion weights are fixed in app/llm/fuse.py and are NOT tuned on this split. Run on the validation split first for plumbing only.
Resource sampling: nvidia-smi VRAM (max over run, minus idle baseline) and the RSS of any 'ollama' process; token counts from the runtime; energy estimate = GPU-seconds x ASSUMED_WATTS (an assumption, printed)."""
import sys, json, argparse, time, pickle, collections, threading, subprocess, os, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "app", ROOT / "scripts", ROOT / "scripts" / "taskB"): sys.path.insert(0, str(p))
from PIL import Image
from schema import *
from data import load
from evaluate import evaluate, micro, score_field
import llm_eval as LE
from llm import validate as V, enhance as EN, fuse as FU
from llm.client import OllamaClient, LLMUnavailable, LLMBadOutput
from llm.config import settings
OUT = ROOT / "reports/taskB/llm"; F = LE.FIELDS_EVAL; ASSUMED_WATTS = 60; INR_PER_KWH = 8.0     # ASSUMPTIONS for the cost estimate, not measurements

class Mon(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.stop = False; self.vram = []; self.rss = []
    def run(self):
        while not self.stop:
            try: self.vram.append(int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.split()[0]))
            except Exception: pass
            tot = 0
            for pid in os.listdir("/proc"):
                if pid.isdigit():
                    try:
                        if open(f"/proc/{pid}/comm").read().strip().startswith("ollama"): tot += int([l for l in open(f"/proc/{pid}/status") if l.startswith("VmRSS")][0].split()[1])
                    except Exception: pass
            self.rss.append(tot / 1024); time.sleep(1)

def slices(d):
    s = ["real"]; s.append("real_indian") if d["country"] == "IN" else None; s.append("src:" + d["source_dataset"])
    if d["width"] * d["height"] < 700 * 900: s.append("low_res(<0.63MP)")
    return s

def run_mode(cl, docs, cfg, use_image, cache_f):
    cache = {json.loads(l)["id"]: json.loads(l) for l in open(cache_f)} if cache_f.exists() else {}; mon = Mon(); mon.start(); base_v = None; t0 = time.time(); fails = collections.Counter(); stats = []
    time.sleep(1.5); base_v = min(mon.vram) if mon.vram else 0; streak = 0
    for i, d in enumerate(docs):
        did = d["document_id"]
        if did in cache: continue
        segs = LE.segs_of(d)
        try:
            js, raw, secs = EN.run_llm(cl, segs, (d["width"], d["height"]), Image.open(ROOT / "datasets" / d["image"]).convert("RGB") if use_image else None); st = dict(cl.last_stats); st["fallback_text"] = bool(getattr(cl, "last_fallback", False))
        except (LLMUnavailable, LLMBadOutput) as e: js, secs, st = None, None, {}; fails[type(e).__name__] += 1; streak = streak + 1
        else: streak = 0
        if streak >= 10: raise SystemExit("aborting: 10 consecutive LLM failures (server/runner problem, not a model result) - nothing further recorded")
        if js is not None or streak < 10:
            with open(cache_f, "a") as f: f.write(json.dumps(dict(id=did, js=js, secs=secs, stats=st)) + "\n")
        cache[did] = dict(id=did, js=js, secs=secs, stats=st)
        if (i + 1) % 10 == 0: print(f"  {i + 1}/{len(docs)} {time.time() - t0:.0f}s", flush=True)
    mon.stop = True; time.sleep(1.1)
    res = dict(peak_vram_mb=max(mon.vram) if mon.vram else None, idle_vram_mb=base_v, rss_mb_peak=round(max(mon.rss), 0) if mon.rss else None, wall_s_this_run=round(time.time() - t0, 1), failures=dict(fails))
    return cache, res

def score_systems(docs, P, cache):
    sysp = {k: {} for k in ("v1", "raw", "validated", "fused", "fused_suggest")}; rej = 0; props_n = 0; auto_n = 0; rows = {}
    for d in docs:
        did = d["document_id"]; segs = LE.segs_of(d); ocr = V.OCR(segs); v1 = LE.v1_fields(P.get(did, {})); sysp["v1"][did] = {f: P[did][f] for f in F if f in P.get(did, {})}
        js = (cache.get(did) or {}).get("js")
        if js is None: 
            for k in ("raw", "validated", "fused", "fused_suggest"): sysp[k][did] = sysp["v1"][did]
            continue
        props = (js.get("fields") if isinstance(js, dict) else None) or {}; raw_p, val_p, chk = {}, {}, {}
        for f in F:
            p = props.get(f); p = p if isinstance(p, dict) else ({"value": p} if p is not None else None); v = (p or {}).get("value")
            if v in (None, "", []): continue
            props_n += 1; rp = LE.raw_pred(f, v)
            if rp: raw_p[f] = rp
            g = V.ground(f, v, ocr)
            if g["status"] in ("grounded", "corrected"): chk[f] = dict(value=v, ground=g); val_p[f] = dict(value=str(v), norm=g["norm"], score=None, confidence=None)
            else: rej += 1
        fused, _ = FU.fuse(dict(v1), chk, ocr)
        sysp["raw"][did] = raw_p; sysp["validated"][did] = val_p; sysp["fused"][did] = {f: LE.as_pred(f, v) for f, v in fused.items() if f in F}
        sysp["fused_suggest"][did] = {f: LE.as_pred(f, v) for f, v in fused.items() if f in F and v["origin"] != "llm"}      # DEFAULT policy: LLM-only values become suggestions, not fields
        rows[did] = {f: (v["needs_review"], v["level"]) for f, v in fused.items() if f in F}; auto_n += bool(EN.needs_llm(v1, ocr))
    return sysp, dict(proposals=props_n, rejected_ungrounded=rej, auto_triggered=auto_n), rows

def perdoc(docs, preds):
    out = {}
    for d in docs:
        t = [0, 0, 0]
        for f in F:
            r = score_field([d], preds, f); t[0] += r["tp"]; t[1] += r["fp"]; t[2] += r["fn"]
        out[d["document_id"]] = t
    return out
def f1_of(ids, pd):
    tp = sum(pd[i][0] for i in ids); fp = sum(pd[i][1] for i in ids); fn = sum(pd[i][2] for i in ids); return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int); ap.add_argument("--modes", default="text,vlm"); ap.add_argument("--boot", type=int, default=1000); a = ap.parse_args()
    cfg = settings() | {"mode": "always", "keep_alive": "30m"}; sp = "test" if a.split == "test" else "val"
    docs = [d for d in load(sp) if not d["synthetic"]]; docs = docs[: a.limit] if a.limit else docs
    P = pickle.load(open(ROOT / "reports/taskB/all_predictions.pkl", "rb"))["preds"][sp]; OUT.mkdir(parents=True, exist_ok=True); report = dict(split=sp, n_docs=len(docs), model=cfg["model"], fields=F, modes={}, assumptions=dict(gpu_watts=ASSUMED_WATTS, inr_per_kwh=INR_PER_KWH))
    ids = [d["document_id"] for d in docs]
    for mode in a.modes.split(","):
        use_img = mode == "vlm"; c2 = dict(cfg, use_image=use_img); cl = OllamaClient(c2); okm, names = cl.available()
        if not okm: raise SystemExit(f"model {cfg['model']} not available: {names}")
        tag = f"{sp}_{cfg['model'].replace(':', '-')}_{mode}"; print("mode", mode, flush=True)
        cache, res = run_mode(cl, docs, c2, use_img, OUT / f"bench_cache_{tag}.jsonl")
        sysp, vstats, flags = score_systems(docs, P, cache)
        secs = [cache[i]["secs"] for i in ids if cache.get(i, {}).get("secs")]; tok_in = [cache[i]["stats"].get("prompt_eval_count") or 0 for i in ids if cache.get(i)]; tok_out = [cache[i]["stats"].get("eval_count") or 0 for i in ids if cache.get(i)]
        res.update(latency_s=dict(mean=round(sum(secs) / len(secs), 2), median=round(sorted(secs)[len(secs) // 2], 2), p90=round(sorted(secs)[int(len(secs) * .9)], 2), n=len(secs)) if secs else None,
                   tokens=dict(prompt_mean=round(sum(tok_in) / max(len(tok_in), 1)), output_mean=round(sum(tok_out) / max(len(tok_out), 1))),
                   energy_estimate=dict(gpu_seconds=round(sum(secs), 0), kwh=round(sum(secs) * ASSUMED_WATTS / 3.6e6, 4), inr=round(sum(secs) * ASSUMED_WATTS / 3.6e6 * INR_PER_KWH, 2), api_cost_usd=0.0, note="local inference: no API cost; energy = ASSUMED watts x measured seconds"))
        res["validation"] = vstats; sc = {}
        for name, preds in sysp.items():
            ev = evaluate(docs, preds, F); sc[name] = dict(real=dict(micro=micro(ev["real"], F), per_field={f: dict(f1=r["f1"], precision=r["precision"], recall=r["recall"], n_gt=r["n_gt"], halluc_rate=r["halluc_rate"], fp=r["fp"], fn=r["fn"], null_rate=r["null_rate"]) for f, r in ev["real"].items()}), real_indian=micro(ev["real_indian"], F))
        res["scores"] = sc
        # slices
        pdv = {n: perdoc(docs, p) for n, p in sysp.items()}; sl = collections.defaultdict(list)
        for d in docs:
            for s in slices(d): sl[s].append(d["document_id"])
        res["slices_micro_f1"] = {s: {"n_docs": len(v), **{n: round(f1_of(v, pdv[n]), 4) for n in sysp}} for s, v in sl.items() if len(v) >= 5}
        # paired bootstrap on documents
        rnd = random.Random(0); diffs = {n: [] for n in ("raw", "validated", "fused", "fused_suggest")}
        for _ in range(a.boot):
            s = [rnd.choice(ids) for _ in ids]; b = f1_of(s, pdv["v1"])
            for n in diffs: diffs[n].append(f1_of(s, pdv[n]) - b)
        res["bootstrap_f1_diff_vs_v1"] = {n: dict(mean=round(sum(v) / len(v), 4), ci95=[round(sorted(v)[int(.025 * len(v))], 4), round(sorted(v)[int(.975 * len(v))], 4)]) for n, v in diffs.items()}
        # completeness / hallucination
        comp = {}
        for n, preds in sysp.items():
            have = tot = absent = fp_abs = 0
            for d in docs:
                for f in F:
                    if f not in d["supported_fields"]: continue
                    g = d["fields"].get(f); p = (preds.get(d["document_id"]) or {}).get(f)
                    if g is not None: tot += 1; have += p is not None
                    else: absent += 1; fp_abs += p is not None
            comp[n] = dict(completeness=round(have / tot, 4) if tot else None, filled_when_gt_absent=round(fp_abs / absent, 4) if absent else None, n_gt=tot, n_absent=absent)
        res["completeness_and_hallucination"] = comp
        lv = collections.defaultdict(lambda: [0, 0])
        for d in docs:
            for f, p in sysp["fused"][d["document_id"]].items():
                if f not in d["supported_fields"] or d["document_id"] not in flags: continue
                g = d["fields"].get(f); ok = g is not None and values_match(f, p["norm"], g["norm"]); lvl = flags[d["document_id"]].get(f)
                if lvl: lv[lvl[1]][0] += 1; lv[lvl[1]][1] += ok
        res["fused_precision_by_evidence_level"] = {k: dict(n=v[0], precision=round(v[1] / v[0], 4)) for k, v in lv.items() if v[0]}
        # failure cases: fused wrong where v1 right, and fixed
        worse, better = [], []
        for d in docs:
            for f in F:
                if f not in d["supported_fields"]: continue
                g = d["fields"].get(f); gv = g["norm"] if g else None
                a1 = (sysp["v1"][d["document_id"]].get(f) or {}).get("norm"); a2 = (sysp["fused"][d["document_id"]].get(f) or {}).get("norm")
                ok1 = gv is not None and a1 is not None and values_match(f, a1, gv); ok2 = gv is not None and a2 is not None and values_match(f, a2, gv)
                if ok1 and not ok2: worse.append(dict(doc=d["document_id"], field=f, gt=str(gv)[:40], v1=str(a1)[:40], fused=str(a2)[:40]))
                if ok2 and not ok1: better.append(dict(doc=d["document_id"], field=f, gt=str(gv)[:40], v1=str(a1)[:40], fused=str(a2)[:40]))
        res["fused_fixed_vs_broke"] = dict(fixed=len(better), broke=len(worse), examples_broke=worse[:12], examples_fixed=better[:12])
        report["modes"][mode] = res
        print(json.dumps({n: sc[n]["real"]["micro"] for n in sc}, indent=0)[:900], res["latency_s"], "VRAM", res["peak_vram_mb"], flush=True)
    (OUT / f"bench_{sp}_{cfg['model'].replace(':', '-')}.json").write_text(json.dumps(report, indent=1, default=float)); print("written")
if __name__ == "__main__": main()
