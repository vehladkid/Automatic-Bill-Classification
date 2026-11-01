"""READ-ONLY analysis of the recorded frozen test-split benchmark (reports/taskB/llm/bench_cache_test_*_text.jsonl + all_predictions.pkl). Changes no pipeline code, prompt, threshold or policy.
Adds what bench_test_*.json does not contain: macro-F1 (evaluate.macro rule: fields with >=5 GT), per-expense-category micro-F1, paired bootstrap for the DEFAULT policy, fixed-vs-broke lists. Writes a NEW file only."""
import sys, json, pickle, collections, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in ("app", "scripts", "scripts/taskB"): sys.path.insert(0, str(ROOT / p))
from data import load
from evaluate import evaluate, micro, macro, by_category, score_field
from schema import values_match
import llm_bench as B
docs = [d for d in load("test") if not d["synthetic"]]
P = pickle.load(open(ROOT / "reports/taskB/all_predictions.pkl", "rb"))["preds"]["test"]
cache = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "reports/taskB/llm/bench_cache_test_qwen2.5vl-3b_text.jsonl")}
assert set(cache) == {d["document_id"] for d in docs}, "cache and test documents differ"
sysp, vstats, flags = B.score_systems(docs, P, cache); F = B.F; out = {"n_docs": len(docs), "same_documents_all_systems": True}
ev = {n: evaluate(docs, p, F)["real"] for n, p in sysp.items()}
out["micro"] = {n: micro(ev[n], F) for n in sysp}; out["macro_f1_over_fields_with_ge5_gt"] = {n: macro(ev[n], F) for n in sysp}
out["fields_in_macro"] = [f for f in F if f in ev["v1"] and ev["v1"][f]["n_gt"] >= 5]
cats = collections.defaultdict(lambda: {"n_docs": 0})
man = {json.loads(l)["document_id"]: json.loads(l) for l in open(ROOT / "datasets/metadata/manifest.jsonl")}
for n in ("v1", "fused_suggest", "fused"):
    for c, dd in {c: [d for d in docs if man[d["document_id"]]["expense_category"] == c] for c in sorted({man[d["document_id"]]["expense_category"] for d in docs})}.items():
        r = evaluate(dd, sysp[n], F)["real"]; m = micro(r, F); cats[c]["n_docs"] = len(dd); cats[c][n] = dict(f1=m["f1"], tp=m["tp"], fp=m["fp"], fn=m["fn"])
out["by_expense_category_micro_f1"] = dict(cats)
pdv = {n: B.perdoc(docs, sysp[n]) for n in sysp}; ids = [d["document_id"] for d in docs]; rnd = random.Random(0); ds = []; dm = []
for _ in range(2000):
    s = [rnd.choice(ids) for _ in ids]; ds.append(B.f1_of(s, pdv["fused_suggest"]) - B.f1_of(s, pdv["v1"]))
out["paired_bootstrap_2000_default_minus_v1"] = dict(mean=round(sum(ds) / len(ds), 4), ci95=[round(sorted(ds)[50], 4), round(sorted(ds)[1949], 4)], share_resamples_default_better=round(sum(x > 0 for x in ds) / len(ds), 3))
fixed, broke = [], []
for d in docs:
    for f in F:
        if f not in d["supported_fields"]: continue
        g = d["fields"].get(f); gv = g["norm"] if g else None
        a1 = (sysp["v1"][d["document_id"]].get(f) or {}).get("norm"); a2 = (sysp["fused_suggest"][d["document_id"]].get(f) or {}).get("norm")
        ok1 = gv is not None and a1 is not None and values_match(f, a1, gv); ok2 = gv is not None and a2 is not None and values_match(f, a2, gv)
        row = dict(doc=d["document_id"], src=d["source_dataset"], field=f, gt=str(gv)[:30], v1=str(a1)[:30], default=str(a2)[:30])
        if ok1 and not ok2: broke.append(row)
        if ok2 and not ok1: fixed.append(row)
out["default_policy_vs_v1"] = dict(fixed=len(fixed), broke=len(broke), broke_by_field=dict(collections.Counter(r["field"] for r in broke)), broke_by_source=dict(collections.Counter(r["src"] for r in broke)), fixed_by_field=dict(collections.Counter(r["field"] for r in fixed)), broke_examples=broke[:15], fixed_examples=fixed[:15])
bad = [i for i in ids if cache[i]["js"] is None]; out["llm_failed_docs"] = [dict(doc=i, src=next(d["source_dataset"] for d in docs if d["document_id"] == i)) for i in bad]
out["auto_mode_trigger_rate"] = dict(triggered=vstats["auto_triggered"], of=len(docs))
(ROOT / "reports/taskB/llm/test_analysis_qwen2.5vl-3b_text.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps({k: out[k] for k in ("micro", "macro_f1_over_fields_with_ge5_gt", "paired_bootstrap_2000_default_minus_v1", "llm_failed_docs", "auto_mode_trigger_rate")}, indent=1)); print("default vs v1:", {k: v for k, v in out["default_policy_vs_v1"].items() if not k.endswith("examples")})
print("categories:"); [print(f"  {c:14s} n={v['n_docs']:3d} v1={v['v1']['f1']:.3f} default={v['fused_suggest']['f1']:.3f} fused_add={v['fused']['f1']:.3f}") for c, v in out["by_expense_category_micro_f1"].items() if v["n_docs"] >= 3]
