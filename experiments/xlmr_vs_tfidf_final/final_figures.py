"""Figures for the final study (all read from outputs/). Needs matplotlib (not in the project venv): PYTHONPATH=<dir with matplotlib> python final_figures.py"""
import sys, json
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIN = Path(__file__).resolve().parent; FIG = FIN / "figures"; FIG.mkdir(exist_ok=True); O = FIN / "outputs"; J = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
S = J(O / "stats/primary256_stats.json"); tr_ = J(O / "truncation_audit.json"); cost = J(O / "cost_analysis.json"); ph1 = J(O / "phase1_reconciliation.json"); sel = J(FIN / "configs/selected_variant.json")
BL, RD, GR = "#1f77b4", "#d62728", "#7f7f7f"
if S:   # 1. per-seed macro-F1 vs TF-IDF
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); ps = S["per_seed"]; x = [p["seed"] for p in ps]; ax.axhline(S["tfidf"]["macro_f1"], color=BL, ls="--", label=f"TF-IDF + LR ({S['tfidf']['macro_f1']:.3f})"); ax.scatter(x, [p["macro_f1"] for p in ps], color=RD, zorder=3, label="XLM-R, one run per seed")
    for p in ps: ax.annotate(f"{p['macro_f1']:.3f}", (p["seed"], p["macro_f1"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    if ph1: ax.scatter([-0.8], [ph1["recomputed"]["xlmr_original"]["macro_f1"]], marker="D", color=GR, zorder=3, label="original single run (checkpoint not saved)")
    ax.axhline(S["stability"]["macro_f1_mean"], color=RD, lw=0.8, alpha=0.5); ax.set_xlabel("training seed"); ax.set_ylabel("macro-F1, real test documents (n=386)"); ax.set_title("XLM-R macro-F1 across training seeds vs TF-IDF", fontsize=10); ax.legend(fontsize=7, loc="lower right"); ax.set_xticks([-0.8] + x); ax.set_xticklabels(["orig."] + [str(i) for i in x]); plt.tight_layout(); plt.savefig(FIG / "final_fig1_seed_macro_f1.png", dpi=170); plt.close()
    # 2. bootstrap distribution
    d = np.load(O / "stats/primary256_hier_delta_macro_f1.npy"); P = S["primary"]; fig, ax = plt.subplots(figsize=(6.4, 3.6)); ax.hist(d, bins=60, color="#9ecae1"); ax.axvline(0, color="k", lw=1); ax.axvline(0.02, color="orange", ls="--", label="practical threshold 0.02"); ax.axvline(P["delta_macro_f1"], color=RD, label=f"point estimate {P['delta_macro_f1']:+.3f}")
    for v in P["ci95_hierarchical"]: ax.axvline(v, color=GR, ls=":")
    ax.set_title(f"Macro-F1 difference XLM-R - TF-IDF (hierarchical bootstrap, 95% interval {P['ci95_hierarchical'][0]:+.3f} to {P['ci95_hierarchical'][1]:+.3f})", fontsize=8); ax.set_xlabel("macro-F1 difference"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / "final_fig2_delta_distribution.png", dpi=170); plt.close()
    # 3. per-class
    pc = S["per_class"]; cl = list(pc); fig, ax = plt.subplots(figsize=(8, 3.9)); xx = np.arange(len(cl)); ax.bar(xx - 0.2, [pc[c]["tfidf_f1"] for c in cl], 0.4, color=BL, label="TF-IDF + LR"); ax.bar(xx + 0.2, [pc[c]["xlmr_f1_mean"] for c in cl], 0.4, color=RD, label="XLM-R (mean of seeds; whiskers min-max)", yerr=[[pc[c]["xlmr_f1_mean"] - pc[c]["xlmr_f1_min"] for c in cl], [pc[c]["xlmr_f1_max"] - pc[c]["xlmr_f1_mean"] for c in cl]], capsize=2)
    ax.set_xticks(xx); ax.set_xticklabels([f"{c}\n(n={pc[c]['support']})" for c in cl], fontsize=7, rotation=0); ax.set_ylabel("F1"); ax.set_title("Per-class F1, real test documents", fontsize=10); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / "final_fig3_per_class.png", dpi=170); plt.close()
    # 4. training curves
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for p in S["seeds"]:
        h = J(FIN / "runs" / f"primary256_s{p}" / "history.json"); ax.plot([e["epoch"] for e in h], [e["val_real_macro_f1"] for e in h], marker="o", ms=3, label=f"seed {p}")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation real-only macro-F1"); ax.set_title("XLM-R validation curves (checkpoint chosen on these, not on test)", fontsize=9); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / "final_fig4_validation_curves.png", dpi=170); plt.close()
if tr_:   # 5. truncation per class
    pc = tr_["per_class_test_real"]; cl = list(pc); fig, ax = plt.subplots(figsize=(8, 3.6)); xx = np.arange(len(cl)); ax.bar(xx - 0.2, [pc[c]["over_256_pct"] for c in cl], 0.4, color=RD, label="> 256 tokens (truncated by XLM-R-256)"); ax.bar(xx + 0.2, [pc[c]["over_512_pct"] for c in cl], 0.4, color="#fdae6b", label="> 512 tokens")
    ax.set_xticks(xx); ax.set_xticklabels([f"{c}\n(n={pc[c]['n']})" for c in cl], fontsize=7); ax.set_ylabel("% of real test documents"); ax.set_title("Share of documents longer than the XLM-R input limit, by class (token count includes the 2 special tokens)", fontsize=9); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / "final_fig5_truncation_by_class.png", dpi=170); plt.close()
if cost and S:   # 6. performance vs cost
    m = {"tfidf": S["tfidf"]["macro_f1"], "xlmr": S["stability"]["macro_f1_mean"]}; ext = {"bilstm": 0.598, "bilstm_attn": 0.683, "layoutlmv3": 0.704}
    old = J(FIN.parents[0] / "text_vs_neural/outputs/results.json");
    if old: ext = {b["key"]: b["macro_f1"] for b in old["table_B"] if b["key"] in ext}
    m.update(ext); fig, ax = plt.subplots(figsize=(6.6, 4.1)); col = {"tfidf": BL, "xlmr": RD, "bilstm": "#ff7f0e", "bilstm_attn": "#2ca02c", "layoutlmv3": "#8c564b"}
    for k, r in cost["per_model"].items():
        if r["cpu_end_to_end_ms"] and k in m: ax.scatter(r["cpu_end_to_end_ms"], m[k], s=40 + r["artifact_mb"] / 3, color=col[k], alpha=0.8); ax.annotate(f"{r['name'].split(' (')[0]}\n{r['artifact_mb']:.0f} MB", (r["cpu_end_to_end_ms"], m[k]), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_ylim(0.55, 0.97); ax.set_xlim(0.8, 3000); ax.set_xscale("log"); ax.set_xlabel("CPU end-to-end latency per document (ms, log scale; OCR excluded)"); ax.set_ylabel("macro-F1 (real test; XLM-R = mean of seeds)"); ax.set_title("Performance versus cost (bubble area grows with model size)", fontsize=10, pad=10); plt.tight_layout(); plt.savefig(FIG / "final_fig6_performance_vs_cost.png", dpi=170); plt.close()
    sc = cost.get("scenarios_seconds", {}); keys = [k for k in ("tfidf", "xlmr") if k in cost["per_model"]]
    if sc and len(keys) == 2:
        fig, ax = plt.subplots(figsize=(7, 3.8)); ns = list(sc); xx = np.arange(len(ns)); w = 0.2
        for j, (k, lab, fld, c) in enumerate((("tfidf", "TF-IDF, CPU", "cpu_end_to_end_batch1_s", BL), ("xlmr", "XLM-R, CPU (batch 1)", "cpu_end_to_end_batch1_s", RD), ("xlmr", "XLM-R, GPU (batch 1)", "gpu_end_to_end_batch1_s", "#fc9272"), ("xlmr", "XLM-R, GPU (batch 32)", "gpu_throughput_bs32_s", "#fdd0a2"))):
            v = [sc[n][k][fld] for n in ns]; ax.bar(xx + (j - 1.5) * w, v, w, color=c, label=lab)
        ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels([f"{int(n):,} documents" for n in ns]); ax.set_ylabel("estimated time (s, log scale)"); ax.set_title("Estimated classification time (documents x measured latency; OCR excluded)", fontsize=9); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / "final_fig7_scenarios.png", dpi=170); plt.close()
us, ur = J(O / "unseen_synthetic/stats.json"), J(O / "unseen_real/stats.json")
if us and ur:   # 8. unseen sets
    A = us["analyses"]; sets = [("synthetic\nclean", A["clean_seeds0to4"]), ("synthetic\nnoisy photo", A["noisy_seeds0to4"]), ("synthetic\nlong preamble", A["long_seeds0to4"]), ("real Portuguese\n(2 classes)", ur["analyses"]["primary_classes_ge10_seeds0to4"]), ("real test set\n(main, n=386)", dict(tfidf=dict(macro_f1=S["tfidf"]["macro_f1"]), xlmr=dict(macro_f1_mean=S["stability"]["macro_f1_mean"]), ci95_hierarchical=S["primary"]["ci95_hierarchical"], delta_macro_f1=S["primary"]["delta_macro_f1"]))]
    fig, ax = plt.subplots(figsize=(7.4, 3.9)); xx = np.arange(len(sets)); ax.bar(xx - 0.2, [a["tfidf"]["macro_f1"] for _, a in sets], 0.4, color=BL, label="TF-IDF + LR"); ax.bar(xx + 0.2, [a["xlmr"]["macro_f1_mean"] for _, a in sets], 0.4, color=RD, label="XLM-R (mean of 5 pre-registered seeds)")
    for i, (_, a) in enumerate(sets): ax.annotate(f"{a['delta_macro_f1']:+.3f}\n[{a['ci95_hierarchical'][0]:+.2f}, {a['ci95_hierarchical'][1]:+.2f}]", (i, max(a["tfidf"]["macro_f1"], a["xlmr"]["macro_f1_mean"]) + 0.02), ha="center", fontsize=7)
    ax.set_xticks(xx); ax.set_xticklabels([n for n, _ in sets], fontsize=7); ax.set_ylim(0, 1.12); ax.set_ylabel("macro-F1"); ax.set_title("Macro-F1 on the main test set and on the unseen sets (label: difference and 95% hierarchical interval)", fontsize=8); ax.legend(fontsize=7, loc="upper right"); plt.tight_layout(); plt.savefig(FIG / "final_fig8_unseen_sets.png", dpi=170); plt.close()
hy = J(O / "hybrid/hybrid_stats.json")
if hy:   # 9. hybrid routing curve (descriptive)
    cv = hy["analyses"]["main_real_test_seeds0to4"]["routing_curve_descriptive"]; a = hy["analyses"]["main_real_test_seeds0to4"]; fig, ax = plt.subplots(figsize=(6.4, 3.8)); ax.plot([100 * c["routed_share"] for c in cv], [c["macro_f1"] for c in cv], marker="o", ms=3, color="#2ca02c", label="hybrid (mean of 5 seeds), by threshold")
    ax.axhline(a["tfidf_macro_f1"], color=BL, ls="--", label=f"TF-IDF alone ({a['tfidf_macro_f1']:.3f})"); ax.axhline(a["xlmr_macro_f1_mean"], color=RD, ls="--", label=f"XLM-R alone ({a['xlmr_macro_f1_mean']:.3f})"); ax.scatter([100 * a["routed_share_mean"]], [a["hybrid_macro_f1_mean"]], s=70, color="k", zorder=4, label=f"validation-chosen thresholds ({100 * a['routed_share_mean']:.0f}% routed)")
    ax.set_xlabel("share of documents sent to XLM-R (%)"); ax.set_ylabel("macro-F1, real test (n=386)"); ax.set_title("Hybrid routing curve on the real test set (descriptive; thresholds were NOT chosen on it)", fontsize=8); ax.legend(fontsize=7, loc="lower right"); plt.tight_layout(); plt.savefig(FIG / "final_fig9_hybrid_curve.png", dpi=170); plt.close()
print("figures:", sorted(p.name for p in FIG.glob("final_fig*.png")))
