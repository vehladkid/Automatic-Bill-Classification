"""Combines the measured resource benchmarks (outputs/resources/*.json), training times (runs/*/config.json) and historical values into one cost table + scenario estimates.
Estimates (docs x measured median latency) are labelled as such. Historical (not re-measured) values are labelled 'historical'. Output: outputs/cost_analysis.json"""
import sys, json, glob
import numpy as np
from pathlib import Path
FIN = Path(__file__).resolve().parent; ROOT = FIN.parents[1]; RES = FIN / "outputs" / "resources"; T = ROOT / "experiments/text_vs_neural/outputs"
J = lambda p: json.loads(Path(p).read_text())
res = {p.stem: J(p) for p in RES.glob("*.json")}
def pick(name): return res.get(name) or next((v for k, v in res.items() if k.startswith(name)), None)
models = {"tfidf": ("TF-IDF + LR", pick("tfidf")), "xlmr": ("XLM-RoBERTa-base (fine-tuned, 256 tokens)", pick("xlmr_primary256_s0")), "bilstm": ("BiLSTM", pick("bilstm")), "bilstm_attn": ("BiLSTM + attention", pick("bilstm_attn")), "layoutlmv3": ("LayoutLMv3-base", pick("layoutlmv3"))}
extra = {k: v for k, v in res.items() if k.startswith("xlmr_") and k != "xlmr_primary256_s0"}
rows = {}
for k, (nm, r) in models.items():
    if r is None: continue
    cpu, gpu = r.get("cpu"), r.get("cuda"); size = sum(r["artifact_files"].values()) / 2**20
    rows[k] = dict(name=nm, artifact_mb=size, artifact_files=r["artifact_files"], load_seconds=r["load_seconds"], rss_after_load_mb=r["rss_after_load_mb"], rss_peak_mb=r["rss_peak_mb"], rss_increment_mb=r["rss_peak_increment_over_before_model_mb"],
                   cpu_model_only_ms=cpu["model_only"]["median_ms"] if cpu else None, cpu_end_to_end_ms=cpu["end_to_end"]["median_ms"] if cpu else None, cpu_end_to_end_p95_ms=cpu["end_to_end"]["p95_ms"] if cpu else None,
                   gpu_model_only_ms=gpu["model_only"]["median_ms"] if gpu else None, gpu_end_to_end_ms=gpu["end_to_end"]["median_ms"] if gpu else None, gpu_peak_mb_b1=gpu["gpu_memory"]["peak_allocated_mb"] if gpu else None, gpu_peak_mb_b32=gpu["gpu_memory_batch32"]["peak_allocated_mb"] if gpu else None,
                   throughput_cpu=cpu["throughput_docs_per_s_end_to_end"] if cpu else None, throughput_gpu=gpu["throughput_docs_per_s_end_to_end"] if gpu else None, dependencies=r.get("deployment_dependencies"))
# training: measured here (XLM-R seeds) vs historical (others)
tt = {}
for v in sorted({p.parent.name.rsplit("_s", 1)[0] for p in (FIN / "runs").glob("*/config.json")}):
    c = [J(p) for p in (FIN / "runs").glob(f"{v}_s*/config.json") if J(p)["status"] == "completed"]
    if c: tt[v] = dict(n_runs=len(c), seconds_mean=float(np.mean([x["train_seconds"] for x in c])), seconds_sd=float(np.std([x["train_seconds"] for x in c], ddof=1)) if len(c) > 1 else None, seconds_min=float(min(x["train_seconds"] for x in c)), seconds_max=float(max(x["train_seconds"] for x in c)),
                       gpu_peak_mb_mean=float(np.mean([x["peak_gpu_allocated_mb"] for x in c])), source="measured in this study (GPU shared with idle demo servers; other jobs ran at low priority - see README)")
tt["tfidf"] = dict(n_runs=1, seconds_mean=J(FIN / "outputs/tfidf/config.json")["train_seconds"], source="measured in this study, 1 CPU thread (earlier 16-thread refit: 55 s)")
tt["bilstm"] = dict(n_runs=3, seconds_mean=J(T / "m_bilstm.json")["train_seconds"], source="HISTORICAL (earlier study; selected seed), GPU"); tt["bilstm_attn"] = dict(n_runs=3, seconds_mean=J(T / "m_bilstm_attn.json")["train_seconds"], source="HISTORICAL (earlier study; selected seed), GPU")
tt["layoutlmv3"] = dict(n_runs=1, seconds_mean=J(T / "m6_layoutlmv3.json")["train_seconds"], source="HISTORICAL (original Task A training log), GPU")
out = dict(per_model=rows, training=tt, extra_xlmr_variants={k: dict(maxlen=v.get("maxlen"), cpu_end_to_end_ms=v["cpu"]["end_to_end"]["median_ms"], gpu_end_to_end_ms=v["cuda"]["end_to_end"]["median_ms"], gpu_peak_b32=v["cuda"]["gpu_memory_batch32"]["peak_allocated_mb"]) for k, v in extra.items() if v.get("cpu") and v.get("cuda")})
if "tfidf" in rows and "xlmr" in rows:
    a, b = rows["xlmr"], rows["tfidf"]; r = lambda x, y: (x / y) if x and y else None
    out["ratios_xlmr_over_tfidf"] = dict(artifact_size=r(a["artifact_mb"], b["artifact_mb"]), size_note="XLM-R artifact = pretrained base weights + fine-tuned trainable weights (fp16 delta) as stored; TF-IDF = production joblib", cpu_model_only=r(a["cpu_model_only_ms"], b["cpu_model_only_ms"]), cpu_end_to_end=r(a["cpu_end_to_end_ms"], b["cpu_end_to_end_ms"]),
                                         gpu_end_to_end_vs_tfidf_cpu=r(a["gpu_end_to_end_ms"], b["cpu_end_to_end_ms"]), rss_peak=r(a["rss_peak_mb"], b["rss_peak_mb"]), throughput_cpu_bs32=r(b["throughput_cpu"]["32"], a["throughput_cpu"]["32"]), training_seconds=r(tt.get("primary256", {}).get("seconds_mean"), tt["tfidf"]["seconds_mean"]))
    out["xlmr_merged_fp32_checkpoint_mb"] = 278055952 * 4 / 2**20; out["xlmr_merged_vs_tfidf_ratio"] = out["xlmr_merged_fp32_checkpoint_mb"] / b["artifact_mb"]
    out["scenarios_seconds"] = {}
    for n in (1000, 10000, 100000):
        d = {}
        for k, rw in rows.items():
            d[k] = dict(cpu_end_to_end_batch1_s=n * rw["cpu_end_to_end_ms"] / 1000 if rw["cpu_end_to_end_ms"] else None, gpu_end_to_end_batch1_s=n * rw["gpu_end_to_end_ms"] / 1000 if rw["gpu_end_to_end_ms"] else None,
                        cpu_throughput_bs32_s=n / rw["throughput_cpu"]["32"] if rw["throughput_cpu"] else None, gpu_throughput_bs32_s=(n / rw["throughput_gpu"]["32"]) if rw["throughput_gpu"] and not isinstance(rw["throughput_gpu"]["32"], str) else None)
        out["scenarios_seconds"][str(n)] = d
    out["scenario_note"] = "ESTIMATES = number of documents / measured median or measured throughput; they exclude OCR (about 0.2-0.5 s per page for every model), model loading and queueing"
(FIN / "outputs" / "cost_analysis.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps({k: out[k] for k in ("ratios_xlmr_over_tfidf", "training") if k in out}, indent=1, default=float)[:2500])
