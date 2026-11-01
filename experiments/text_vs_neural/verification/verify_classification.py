"""INDEPENDENT re-computation of every classification number in outputs/results.json, the model JSONs and the CSV tables.
Independent = does not import exp_common's metric functions: labels are read straight from the project's jsonl split files, metrics come from scikit-learn / scipy, ECE uses a different binning code
(np.searchsorted), temperatures are re-fitted with scipy, and the paired bootstrap is re-implemented with a different seed and 10,000 resamples. Reads saved logits only - no model is run or trained."""
import json, csv, sys
import numpy as np
from pathlib import Path
from scipy.special import log_softmax, softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from scipy.stats import binomtest
ROOT = Path(__file__).resolve().parents[3]; EXP = ROOT / "experiments/text_vs_neural"; OUT = EXP / "outputs"
def rows(split): return [json.loads(l) for l in open(ROOT / f"datasets/processed/classification/{split}.jsonl")]
tr, va, te = rows("train"), rows("validation"), rows("test")
labels = sorted({r["label"] for r in tr + va + te}); l2i = {l: i for i, l in enumerate(labels)}
man = {}
for l in open(ROOT / "datasets/metadata/manifest.jsonl"):
    m = json.loads(l); man[m["document_id"]] = m
yv = np.array([l2i[r["label"]] for r in va]); yt = np.array([l2i[r["label"]] for r in te]); real_t = np.array([not r["synthetic"] for r in te]); group_t = np.array([man[r["document_id"]]["group_id"] for r in te])
report = dict(checks=[], mismatches=[]); TOL = 2e-6
def check(name, expected, got, tol=TOL):
    ok = (expected is None and got is None) or (expected is not None and got is not None and abs(expected - got) <= tol); report["checks"].append(dict(name=name, stored=expected, recomputed=got, ok=bool(ok)))
    if not ok: report["mismatches"].append(dict(name=name, stored=expected, recomputed=got))
def ece(p, y, nb=10):
    conf = p.max(1); ok = p.argmax(1) == y; edges = np.linspace(0, 1, nb + 1); b = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, nb - 1); e = 0.0
    for k in range(nb):
        m = b == k
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e
def metrics(lg, y):
    p = softmax(lg, 1); pred = p.argmax(1); labs = sorted(set(y.tolist())); pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=labs, zero_division=0)
    nll = -np.mean(log_softmax(lg, 1)[np.arange(len(y)), y]); brier = np.mean(((p - np.eye(len(labels))[y]) ** 2).sum(1))
    return dict(accuracy=accuracy_score(y, pred), macro_precision=pr.mean(), macro_recall=rc.mean(), macro_f1=f1.mean(), weighted_f1=(f1 * sup).sum() / sup.sum(), ece=ece(p, y), nll=nll, brier=brier, n=len(y), per_class=dict(zip([labels[i] for i in labs], zip(pr, rc, f1, sup))))
def refit_T(lg, y):
    return minimize_scalar(lambda t: -np.mean(log_softmax(lg / t, 1)[np.arange(len(y)), y]), bounds=(0.3, 6), method="bounded").x
print("split sizes: val", len(va), "test", len(te), "real test", int(real_t.sum()), "| classes", len(labels), "| test groups (real)", len(set(group_t[real_t])))
check("n_val", 634, len(va), 0); check("n_test", 629, len(te), 0); check("n_real_test", 386, int(real_t.sum()), 0)
report["split"] = dict(n_train=len(tr), n_val=len(va), n_test=len(te), n_real_test=int(real_t.sum()), labels=labels)
# leakage / consistency of the split files themselves
ids = [set(r["document_id"] for r in s) for s in (tr, va, te)]; gs = [set(man[i]["group_id"] for i in s) for s in ids]
report["leakage"] = dict(id_overlap_train_test=len(ids[0] & ids[2]), id_overlap_val_test=len(ids[1] & ids[2]), group_overlap_train_test=len(gs[0] & gs[2]), group_overlap_val_test=len(gs[1] & gs[2]), group_overlap_train_val=len(gs[0] & gs[1]))
print("split leakage:", report["leakage"])
M = {"tfidf": ("m1_tfidf.json", "tfidf_lr"), "bilstm": ("m_bilstm.json", "bilstm"), "bilstm_attn": ("m_bilstm_attn.json", "bilstm_attn"), "xlmr": ("m5_xlmr.json", "xlmr"), "layoutlmv3": ("m6_layoutlmv3.json", "layoutlmv3"), "fusion": ("m7_fusion.json", "fusion"),
     "lexrank": ("m3_lexrank.json", "lexrank_50")}
LG = {}
for k, (jf, lf) in M.items():
    J = json.loads((OUT / jf).read_text()); J = J["best"] if k == "lexrank" else J; z = np.load(OUT / f"{lf}_logits.npz"); lv, lt = z["val"].astype(np.float64), z["test"].astype(np.float64); LG[k] = lt
    check(f"{k}: val logits rows", len(va), len(lv), 0); check(f"{k}: test logits rows", len(te), len(lt), 0); check(f"{k}: logits width", len(labels), lt.shape[1], 0)
    T = refit_T(lv, yv); check(f"{k}: temperature (refit on validation)", J["temperature"], T, 2e-3)
    for tag, g in (("raw", lt), ("calibrated", lt / J["temperature"])):
        for sub, mask in (("real", real_t), ("all", np.ones_like(real_t))):
            m = metrics(g[mask], yt[mask]); s = J[tag][sub]
            for q in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "ece", "nll", "brier"): check(f"{k}: {tag}/{sub}/{q}", s[q], m[q])
            for cl, (p_, r_, f_, n_) in m["per_class"].items(): check(f"{k}: {tag}/{sub}/per-class F1 {cl}", s["per_class"][cl]["f1"], f_); check(f"{k}: {tag}/{sub}/support {cl}", s["per_class"][cl]["support"], n_, 0)
    cm = np.zeros((len(labels), len(labels)), int)
    for a, b in zip(yt[real_t], lt.argmax(1)[real_t]): cm[a, b] += 1
    check(f"{k}: confusion matrix (real test)", 0, int(np.abs(cm - np.array(J["confusion_real"])).sum()), 0); check(f"{k}: confusion matrix diagonal == accuracy*n", int(round(J["calibrated"]["real"]["accuracy"] * 386)), int(np.trace(cm)), 0)
# CSV tables vs recomputation
tb = {r["key"]: r for r in json.loads((OUT / "results.json").read_text())["table_B"]}
with open(EXP / "table_B_classification.csv") as fh:
    for row in csv.DictReader(fh):
        k = row["key"]; m = metrics(LG[k] / tb[k]["temperature"], yt) if False else metrics((LG[k] / tb[k]["temperature"])[real_t], yt[real_t])
        for q, col in (("accuracy", "accuracy"), ("macro_f1", "macro_f1"), ("macro_precision", "macro_precision"), ("macro_recall", "macro_recall"), ("weighted_f1", "weighted_f1"), ("ece", "ece_calibrated"), ("nll", "nll_calibrated"), ("brier", "brier_calibrated")):
            check(f"CSV table_B {k}: {col}", float(row[col]), m[q], 1e-6)
# ---------------- paired bootstrap: independent re-implementation, different seed, 10,000 resamples ----------------
idx_real = np.where(real_t)[0]; g_real = group_t[idx_real]; ug = np.unique(g_real); members = [np.where(g_real == g)[0] for g in ug]; rng = np.random.default_rng(987654); NB = 10000
pred = {k: LG[k].argmax(1)[idx_real] for k in LG}; y = yt[idx_real]; accs = {k: np.empty(NB) for k in pred}; f1s = {k: np.empty(NB) for k in pred}
for b in range(NB):
    sel = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))]); ys = y[sel]; labs = sorted(set(ys.tolist()))
    for k in pred: accs[k][b] = (pred[k][sel] == ys).mean(); f1s[k][b] = f1_score(ys, pred[k][sel], labels=labs, average="macro", zero_division=0)
stored = json.loads((OUT / "results.json").read_text())["paired_bootstrap"]; boot = {}
for k in pred:
    d_f1 = f1s[k] - f1s["tfidf"]; d_acc = accs[k] - accs["tfidf"]; boot[k] = dict(macro_f1_ci95=[*np.percentile(f1s[k], [2.5, 97.5])], accuracy_ci95=[*np.percentile(accs[k], [2.5, 97.5])], d_f1=[d_f1.mean(), *np.percentile(d_f1, [2.5, 97.5])], d_acc=[d_acc.mean(), *np.percentile(d_acc, [2.5, 97.5])])
    if k in stored:
        for q, sk in (("macro_f1_ci95", "macro_f1_ci95"), ("accuracy_ci95", "accuracy_ci95")):
            for j, e in enumerate(("lo", "hi")): check(f"bootstrap {k}: {q} {e} (stored 2,000 resamples vs independent 10,000; tol 0.012)", stored[k][sk][j], boot[k][q][j], 0.012)
        for q, sk in (("d_f1", "delta_macro_f1"), ("d_acc", "delta_accuracy")):
            for j, e in enumerate(("mean", "lo", "hi")): check(f"bootstrap {k}: delta {q} {e} vs TF-IDF (tol 0.012)", stored[k][sk][j], boot[k][q][j], 0.012)
        # the conclusion that matters: does the interval exclude zero? (must agree)
        sv = "better" if stored[k]["delta_macro_f1"][1] > 0 else "worse" if stored[k]["delta_macro_f1"][2] < 0 else "unresolved"; iv = "better" if boot[k]["d_f1"][1] > 0 else "worse" if boot[k]["d_f1"][2] < 0 else "unresolved"
        check(f"bootstrap {k}: macro-F1 verdict identical (stored '{sv}' vs independent '{iv}')", 0, 0 if sv == iv else 1, 0)
# attention vs bilstm
d = f1s["bilstm_attn"] - f1s["bilstm"]; ci = np.percentile(d, [2.5, 97.5]); sa = json.loads((OUT / "results.json").read_text())["paired_bootstrap_attention_vs_bilstm"]["bilstm_attn"]["delta_macro_f1"]
check("attention vs BiLSTM: delta macro-F1 mean (tol 0.012)", sa[0], d.mean(), 0.012); check("attention vs BiLSTM: CI low (tol 0.012)", sa[1], ci[0], 0.012); check("attention vs BiLSTM: CI high (tol 0.012)", sa[2], ci[1], 0.012)
report["bootstrap_independent"] = {k: {a: [float(x) for x in np.atleast_1d(v)] for a, v in b.items()} for k, b in boot.items()}; report["bootstrap_attention_vs_bilstm"] = [float(d.mean()), float(ci[0]), float(ci[1])]
# ---------------- sensitivity checks (extra, not in the report): document-level bootstrap and exact McNemar for XLM-R and fusion vs TF-IDF ----------------
rng2 = np.random.default_rng(55); n = len(idx_real); sens = {}
for k in ("xlmr", "fusion", "lexrank"):
    dd = []; da = []
    for b in range(4000):
        s = rng2.integers(0, n, n); ys = y[s]; labs = sorted(set(ys.tolist())); dd.append(f1_score(ys, pred[k][s], labels=labs, average="macro", zero_division=0) - f1_score(ys, pred["tfidf"][s], labels=labs, average="macro", zero_division=0)); da.append((pred[k][s] == ys).mean() - (pred["tfidf"][s] == ys).mean())
    b01 = int(((pred[k] == y) & (pred["tfidf"] != y)).sum()); b10 = int(((pred[k] != y) & (pred["tfidf"] == y)).sum()); p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    sens[k] = dict(doc_level_delta_macro_f1=[float(np.mean(dd)), *map(float, np.percentile(dd, [2.5, 97.5]))], doc_level_delta_accuracy=[float(np.mean(da)), *map(float, np.percentile(da, [2.5, 97.5]))], mcnemar_only_model_correct=b01, mcnemar_only_tfidf_correct=b10, mcnemar_exact_p=float(p))
report["sensitivity"] = sens; report["n_real_test_groups"] = len(ug)
print("sensitivity (vs TF-IDF):", json.dumps({k: {a: (np.round(v, 3).tolist() if isinstance(v, list) else (round(v, 4) if isinstance(v, float) else v)) for a, v in s.items()} for k, s in sens.items()}))
n_ok = sum(c["ok"] for c in report["checks"]); print(f"\n{n_ok}/{len(report['checks'])} checks passed; mismatches: {len(report['mismatches'])}")
for m in report["mismatches"][:40]: print("MISMATCH", m)
(EXP / "verification" / "classification_check.json").write_text(json.dumps(report, indent=1, default=float))
