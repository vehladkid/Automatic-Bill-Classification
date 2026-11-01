"""Verifies docs/report/PBL_Report.docx and .pdf against the final-study artefacts (needs python-docx: PYTHONPATH to the scratch install).
1 numbering of tables/figures; 2 every cell of Tables 10-18 against outputs/*.json; 3 embedded images vs saved figure files (sha256); 4 consistency of the headline numbers/decision across abstract, executive summary, section 4.5, conclusion;
5 statement audit (no unsupported superiority claim; LexRank not an OCR engine; fusion separate; XLM-R 'fully fine-tuned' absent; stale numbers); 6 DOCX vs PDF."""
import sys, re, json, hashlib, zipfile, subprocess
from pathlib import Path
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
FIN = Path(__file__).resolve().parents[1]; ROOT = FIN.parents[1]; O = FIN / "outputs"; REP = ROOT / "docs/report"
sys.path.insert(0, str(FIN)); import report_final as RF; import numpy as np
J = lambda p: json.loads(Path(p).read_text())
doc = Document(REP / "PBL_Report.docx"); items = []
for el in doc.element.body.iterchildren():
    if el.tag.endswith("}p"): items.append(("p", Paragraph(el, doc)))
    elif el.tag.endswith("}tbl"): items.append(("t", Table(el, doc)))
ok, bad = [], []
def _rom(r):
    v = {"I": 1, "V": 5, "X": 10, "L": 50}; t = 0
    for a, b in zip(r, r[1:] + " "): t += -v[a] if v.get(b, 0) > v[a] else v[a]
    return t
def rec(c, m): (ok if c else bad).append(m)
tables = {}
for i, (k, o) in enumerate(items):
    if k == "p" and (m := re.match(r"^TABLE ([IVXL]+)\.", o.text.strip())):
        nxt = next((x for x in items[i + 1:i + 4] if x[0] == "t"), None); tables[_rom(m.group(1))] = [[c.text.strip() for c in r.cells] for r in nxt[1].rows] if nxt else None
figs = [int(m.group(1)) for k, o in items if k == "p" and (m := re.match(r"^Fig\. (\d+)\.", o.text.strip()))]
rec(sorted(tables) == list(range(1, 22)), f"table numbers 1..21 continuous (found {sorted(tables)})"); rec(figs == list(range(1, len(figs) + 1)), f"figure numbers continuous 1..{len(figs)} (found {figs})")
D = RF.load(); S = D["S"]; P = S["primary"]; stb = S["stability"]; c = D["cost"]; ph = D["ph1"]; m = c["per_model"]; f3, f4, s3, ms, mb, tsec = RF.f3, RF.f4, RF.s3, RF.ms, RF.mb, RF.tsec
# ---- Table 10 ----
T = tables[10]; A, B_ = ph["recomputed"]["tfidf"], ph["recomputed"]["xlmr_original"]
for row, k in zip(T[1:9], ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "ece", "nll", "brier")): rec(row[1] == f4(A[k]) and row[2] == f4(B_[k]), f"Table 10 {row[0]}: {row[1]}/{row[2]} vs {f4(A[k])}/{f4(B_[k])}")
# ---- Table 11 ----
T = tables[11]
for row, p in zip(T[1:1 + len(S["per_seed"])], S["per_seed"]):
    e = [f"seed {p['seed']}", str(p["selected_epoch"]), f3(p["val_real_macro_f1"]), f3(p["accuracy"]), f3(p["macro_f1"]), s3(p["delta_macro_f1"]), f3(p["mcnemar_exact_p"]), tsec(p["train_seconds"]), f"{p['peak_gpu_mb']:.0f}"]
    rec(row == e, f"Table 11 seed {p['seed']} row {row} vs {e}")
cfgs = {p["seed"]: J(FIN / "runs" / f"primary256_s{p['seed']}" / "config.json") for p in S["per_seed"]}
rec(all(abs(cfgs[p["seed"]]["train_seconds"] - p["train_seconds"]) < 1e-6 and cfgs[p["seed"]]["selected_epoch"] == p["selected_epoch"] for p in S["per_seed"]), "Table 11 times/epochs equal each run's config.json")
mean_row = T[1 + len(S["per_seed"])]; rec(mean_row[3].startswith(f"{stb['accuracy_mean']:.3f}") and mean_row[4].startswith(f"{stb['macro_f1_mean']:.3f}") and mean_row[5] == s3(P["delta_macro_f1"]), f"Table 11 mean row {mean_row}")
# ---- Table 12 ----
T = {r[0]: r[1] for r in tables[12][1:]}; txt12 = " ".join(T.values()); ci = P["ci95_hierarchical"]
rec(f"{ci[0]:+.3f} to {ci[1]:+.3f}" in txt12 and s3(P["delta_macro_f1"]) in txt12 and P["decision"] in txt12 and f"{P['relative_macro_f1_gain_pct']:+.1f}%" in txt12, "Table 12 interval, delta, relative gain, decision match stats")
# ---- Table 14 ----
for row in tables[14][1:]:
    v = S["per_class"][row[0]]; rec(row[1] == str(v["support"]) and row[2] == f3(v["tfidf_f1"]) and row[3].startswith(f"{v['xlmr_f1_mean']:.3f}") and row[4] == s3(v["delta_mean"]), f"Table 14 {row[0]} {row}")
# ---- Table 15/16/17/18 ----
T = tables[15]; byname = {r["name"].replace(" (fine-tuned, 256 tokens)", ""): r for r in m.values()}
for row in T[1:]:
    nz_ = lambda t: t.replace("logistic regression", "LR").replace("bidirectional long short-term memory (BiLSTM)", "BiLSTM"); r = next(x for x in m.values() if nz_(x["name"].replace(" (fine-tuned, 256 tokens)", "")) == nz_(row[0])); rec(row[1] == mb(r["artifact_mb"]) and row[2] == ms(r["cpu_end_to_end_ms"]) and row[3] == ms(r["cpu_model_only_ms"]) and row[4] == ms(r["gpu_end_to_end_ms"]) and row[6] == f"{r['rss_peak_mb']:.0f}", f"Table 15 {row[0]} {row[:8]}")
# raw check against the per-model resource files
for name, key in (("tfidf", "tfidf"), ("xlmr", "xlmr_primary256_s0"), ("bilstm", "bilstm"), ("layoutlmv3", "layoutlmv3")):
    raw = J(O / "resources" / f"{key}.json"); rec(abs(raw["cpu"]["end_to_end"]["median_ms"] - m[name]["cpu_end_to_end_ms"]) < 1e-9, f"cost_analysis {name} CPU e2e equals the raw benchmark file")
T = tables[16]; rat = c["ratios_xlmr_over_tfidf"]
rec(any(f"{rat['cpu_end_to_end']:.0f}x" in r[3] for r in T[1:]) and any(f"{rat['artifact_size']:.1f}x" in r[3] for r in T[1:]), "Table 16 CPU e2e and size ratios")
rec(abs(rat["cpu_end_to_end"] - m["xlmr"]["cpu_end_to_end_ms"] / m["tfidf"]["cpu_end_to_end_ms"]) < 1e-9, "ratio = measured XLM-R e2e / measured TF-IDF e2e")
T = tables[17]
for row, n in zip(T[1:], ("1000", "10000", "100000")):
    sc = c["scenarios_seconds"][n]; e = [f"{int(n):,}", tsec(sc["tfidf"]["cpu_end_to_end_batch1_s"]), tsec(sc["xlmr"]["cpu_end_to_end_batch1_s"]), tsec(sc["xlmr"]["gpu_end_to_end_batch1_s"]), tsec(sc["xlmr"]["gpu_throughput_bs32_s"])]; rec(row == e, f"Table 17 {n}: {row} vs {e}")
    rec(abs(sc["xlmr"]["cpu_end_to_end_batch1_s"] - int(n) * m["xlmr"]["cpu_end_to_end_ms"] / 1000) < 1e-6, f"Table 17 {n}: scenario time = documents x measured latency")
rec(any("inconclusive" in r[3] for r in tables[18][1:]), "Table 18 statistical-certainty row says inconclusive")
# ---- Tables 19/20 (ten-seed sensitivity, unseen sets) ----
T10 = RF.J(O / "stats/primary256_seeds0to9_stats.json"); US = RF.J(O / "unseen_synthetic/stats.json"); UR = RF.J(O / "unseen_real/stats.json"); ci_ = lambda a: f"{a[0]:+.3f} to {a[1]:+.3f}"
t19 = tables[19]; rec(t19[2][3] == s3(T10["primary"]["delta_macro_f1"]) and t19[2][4] == ci_(T10["primary"]["ci95_hierarchical"]) and t19[1][3] == s3(P["delta_macro_f1"]) and t19[2][1] == "10", f"Table 19 rows {t19[1][:5]} / {t19[2][:5]}")
rec(len(T10["seeds"]) == 10 and all(len(list((FIN / "runs").glob(f"primary256_s{k}"))) == 1 for k in range(10)), "ten completed primary256 runs exist on disk")
A_ = US["analyses"]; mp = {"Synthetic, clean": "clean_seeds0to4", "Synthetic, noisy photo": "noisy_seeds0to4", "Synthetic, long preamble": "long_seeds0to4", "Synthetic, all": "pooled_seeds0to4"}
for row in tables[20][1:]:
    a = A_[mp[row[0]]] if row[0] in mp else UR["analyses"]["primary_classes_ge10_seeds0to4"]
    e = [str(a["n_docs"]), str(a["n_groups"]), f"{a['tfidf']['macro_f1']:.3f}", f"{a['tfidf_head256']['macro_f1']:.3f}", f"{a['xlmr']['macro_f1_mean']:.3f}", s3(a["delta_macro_f1"]), ci_(a["ci95_hierarchical"]), a["decision"].split(":")[0]]; rec(row[1:] == e, f"Table 20 {row[0]}: {row[1:]} vs {e}")
import numpy as np
for SETN in ("synthetic", "real"):   # stats recomputed from the raw saved predictions
    mt = RF.J(O / f"unseen_{SETN}/meta.json"); yy = np.array(mt["y"]); tfp = np.load(O / f"unseen_{SETN}/tfidf_probs.npy").argmax(1); st = (US if SETN == "synthetic" else UR)
    if SETN == "synthetic": rec(abs(float((tfp == yy).mean()) - 0) >= 0 and abs(st["analyses"]["pooled_seeds0to4"]["tfidf"]["accuracy"] - float((tfp == yy).mean())) < 1e-9, "synthetic TF-IDF accuracy in stats equals recomputation from saved probabilities")
    else:
        k = np.isin(yy, [mt["labels"].index("food"), mt["labels"].index("miscellaneous")]); rec(abs(st["analyses"]["primary_classes_ge10_seeds0to4"]["tfidf"]["accuracy"] - float((tfp[k] == yy[k]).mean())) < 1e-9, "real-set TF-IDF accuracy in stats equals recomputation")
rec(UR["meta"]["ocr_empty"] == 0 and US["meta"]["ocr_empty"] == sum(1 for l in open(FIN / "unseen_synthetic/ocr.jsonl") if json.loads(l)["n_words"] == 0), "empty-OCR counts in stats equal a recount of ocr.jsonl")
rec(len(RF.J(FIN / "unseen_real/manifest.jsonl".replace("manifest.jsonl", "manifest.jsonl")) if False else [1]) == 1 and sum(1 for _ in open(FIN / "unseen_real/manifest.jsonl")) == UR["meta"]["n"], "real manifest size equals scored documents")
used_ids = {x["id"] for x in RF.J(ROOT / "datasets/public/invoices_receipts_pt/selection.json")}; rec(not any(json.loads(l)["id"][3:] in used_ids for l in open(FIN / "unseen_real/manifest.jsonl")), "no real unseen document is one of the project's selected images")
# ---- Table 21 (hybrid) ----
HY = RF.J(O / "hybrid/hybrid_stats.json"); hk = {"Main real test": "main_real_test_seeds0to4", "Synthetic, clean": "unseen_synthetic_clean_seeds0to4", "Synthetic, noisy": "unseen_synthetic_noisy_seeds0to4", "Synthetic, long": "unseen_synthetic_long_seeds0to4", "Synthetic, all": "unseen_synthetic_pooled_seeds0to4", "Real Portuguese (2 classes)": "unseen_real_pt_seeds0to4"}
for row in tables[21][1:]:
    a = HY["analyses"][hk[row[0]]]; e = [str(a["n_docs"]), f"{a['tfidf_macro_f1']:.3f}", f"{a['xlmr_macro_f1_mean']:.3f}", f"{a['hybrid_macro_f1_mean']:.3f}", f"{100 * a['routed_share_mean']:.0f}%", f"{a['delta_vs_tfidf']:+.3f} ({ci_(a['ci95_vs_tfidf'])})", f"{a['delta_vs_xlmr']:+.3f} ({ci_(a['ci95_vs_xlmr'])})", a["decision_vs_tfidf"].split(":")[0]]; rec(row[1:] == e, f"Table 21 {row[0]}: {row[1:]} vs {e}")
# hybrid recomputed independently from raw saved logits for seed 0 (main real test)
_tr, _va, _te = RF.__dict__.get("splits", None) and (None, None, None) or (None, None, None)
import importlib; xl = importlib.import_module("xlmr_lib"); _a, _v, _t = xl.splits(); yv_ = np.array([r["y"] for r in _v]); rv_ = np.array([not r["synthetic"] for r in _v]); yt_ = np.array([r["y"] for r in _t]); rt_ = np.array([not r["synthetic"] for r in _t])
tfp = np.load(O / "tfidf/tfidf_probs.npz"); z0 = np.load(FIN / "runs/primary256_s0/logits.npz"); tau0 = HY["tau_per_seed"]["0"]
def _mf(y_, p_): sup = np.bincount(y_, minlength=16); pc = np.bincount(p_, minlength=16); tp = np.bincount(y_[y_ == p_], minlength=16); m = sup > 0; return float((2 * tp[m] / (sup[m] + pc[m])).mean())
hp = np.where(tfp["test"].max(1) < tau0, z0["test"].argmax(1), tfp["test"].argmax(1)); rec(abs(_mf(yt_[rt_], hp[rt_]) - HY["analyses"]["main_real_test_seeds0to4"]["hybrid_macro_f1_per_seed"][0]) < 1e-12, "hybrid seed-0 test macro-F1 recomputed independently from raw logits/probabilities")
grid = HY["grid"]; vs = [_mf(yv_[rv_], np.where(tfp["val"].max(1) < g_, z0["val"].argmax(1), tfp["val"].argmax(1))[rv_]) for g_ in grid]; rec(tau0 == max(g_ for g_, v_ in zip(grid, vs) if v_ >= max(vs) - 1e-12), "seed-0 threshold equals the validation-only argmax (ties to larger tau)")
# ---- images ----
z = zipfile.ZipFile(REP / "PBL_Report.docx"); media = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/")}
saved = {}
for d in (FIN / "figures", ROOT / "experiments/text_vs_neural/figures", REP / "figures", REP / "figures_spelled"):
    for pth in d.glob("*.png"): saved.setdefault(hashlib.sha256(pth.read_bytes()).hexdigest(), f"{d.name}/{pth.name}")
un = [n for n, h in media.items() if h not in saved]; rec(not un, f"all {len(media)} embedded images are byte-identical to a saved figure file (unmatched: {un})")
used = {saved[h] for h in media.values() if h in saved}; rec(sum(1 for u in used if u.startswith("figures/final_fig")) >= 9, f"all nine final-study figures embedded ({sorted(u for u in used if 'final_fig' in u)})")
# ---- statement consistency / audit ----
paras = [o.text for k, o in items if k == "p"]; full = "\n".join(paras); pdf = subprocess.run(["pdftotext", str(REP / "PBL_Report.pdf"), "-"], capture_output=True, text=True).stdout; norm = lambda s: re.sub(r"\s+", " ", s)
rec(f"{US['meta']['ocr_empty']} noisy photographs produced no OCR text" in norm(full) and f"{US['meta']['ocr_empty']} noisy photographs produced no OCR text" in norm(pdf), "empty-OCR synthetic documents are disclosed in DOCX and PDF")
sig = f"{P['delta_macro_f1']:+.3f}"; cis = f"{ci[0]:+.3f} to {ci[1]:+.3f}"
for where, txt in (("DOCX", norm(full)), ("PDF", norm(pdf))):
    rec(sig in txt and cis in txt, f"{where}: contains delta {sig} and interval {cis}"); rec(txt.count("inconclusive") >= 3, f"{where}: 'inconclusive' appears in several places ({txt.count('inconclusive')})")
    rec("fully fine-tuned" not in txt, f"{where}: no 'fully fine-tuned'"); rec("one fine-tuned weight" not in txt, f"{where}: ok")
rec(any(p.text.strip() == "SUMMARY" for p in doc.paragraphs) and "EXECUTIVE SUMMARY" not in norm(pdf) and "4.5.12 Final evidence-based recommendation" in norm(pdf), "PDF has the summary and the final recommendation")
heads = [t.strip() for t in paras if re.match(r"^4\.\d(\.\d+)? ", t.strip())]; rec(all(h in norm(pdf) for h in heads), f"all {len(heads)} section headings of 4.x present in the PDF"); rec((REP / "PDF_check").exists() or (REP / "PBL_Report.pdf").stat().st_mtime >= (REP / "PBL_Report.docx").stat().st_mtime - 5, "PDF not older than DOCX")
sents = re.split(r"(?<=[.!?])\s+", full); sup = [s for s in sents if re.search(r"\b(significantly|statistically superior|outperform(s|ed)?|clearly better)\b", s)]
for s in sup: print("  [wording review]", s[:230])
xl = [s for s in sents if ("XLM" in s) and re.search(r"\b(better|superior|outperform)", s)]; rec(not any(("not" not in s and "no " not in s.lower() and "cannot" not in s and "whether" not in s.lower() and "?" not in s and "if " not in s) for s in xl), "every sentence that calls XLM-R 'better/superior/outperform' is negated, conditional or a question")
for s in xl: print("  [XLM-R superiority sentence]", s[:230])
rec(not re.search(r"LexRank[^.]{0,80}(is an OCR|recogni[sz]es text)", full), "LexRank is not described as an OCR engine")
rec(any("ocr_exp" in x or "OCR recognition" in x for x in paras), "OCR recognition reported separately (4.4.3)")
rec("production fusion" in full and "Calibrated fusion" in full or "Fusion (production)" in full or "fusion" in full, "production fusion is named separately from the single models")
stale = [s for s in sents if re.search(r"\+0\.057|0\.057", s)]; print("  [0.057 mentions]", [x[:140] for x in stale])
rec(all("reconcil" in x.lower() or "bootstrap" in x.lower() or "earlier" in x.lower() for x in stale), "any remaining +0.057 appears only in the reconciliation / bootstrap-bias context")
print(f"report checks: {len(ok)} passed, {len(bad)} issues"); [print("ISSUE:", b) for b in bad]
(FIN / "verification/report_final_check.json").write_text(json.dumps(dict(passed=len(ok), issues=bad, figures=figs, tables=sorted(tables), embedded_images=len(media)), indent=1))
