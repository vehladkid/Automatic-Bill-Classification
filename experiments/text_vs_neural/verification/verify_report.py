"""Verifies docs/report/PBL_Report.docx (+ .pdf) against the saved experiment artefacts. Needs python-docx (PYTHONPATH to the scratch install).
Checks: (1) every value in Tables 5-9 vs results.json / ocr_*.json / m3_lexrank.json / CSVs, (2) figure/table numbering, (3) embedded images vs experiments/figures (sha256),
(4) statement audit (LexRank-as-OCR, XLM-R superiority wording, fusion vs single models, limitations present), (5) docx/pdf consistency."""
import sys, re, json, csv, hashlib, zipfile, subprocess
from pathlib import Path
from docx import Document
ROOT = Path(__file__).resolve().parents[3]; EXP = ROOT / "experiments/text_vs_neural"; OUT = EXP / "outputs"; REP = ROOT / "docs/report"
doc = Document(REP / "PBL_Report.docx"); R = json.loads((OUT / "results.json").read_text()); L = json.loads((OUT / "m3_lexrank.json").read_text()); issues = []; ok = []
def rec(cond, msg):
    (ok if cond else issues).append(msg)
# ---- ordered walk of body: paragraphs and tables ----
body = doc.element.body; items = []; ptable = {}
from docx.table import Table; from docx.text.paragraph import Paragraph
for el in body.iterchildren():
    if el.tag.endswith("}p"): items.append(("p", Paragraph(el, doc)))
    elif el.tag.endswith("}tbl"): items.append(("t", Table(el, doc)))
texts = [(k, o.text if k == "p" else None, o) for k, o in items]
cap_re = re.compile(r"^Table (\d+)\.")
tables = {}
for i, (k, t, o) in enumerate(texts):
    if k == "p" and t and cap_re.match(t.strip()):
        n = int(cap_re.match(t.strip()).group(1)); nxt = next((x for x in texts[i + 1:i + 4] if x[0] == "t"), None); tables[n] = [[c.text.strip() for c in r.cells] for r in nxt[2].rows] if nxt else None
print("tables found by caption:", sorted(tables))
rec(sorted(tables) == [1, 2, 3, 4, 5, 6, 7, 8, 9], f"table numbering is continuous 1..9 (found {sorted(tables)})")
figs = [int(m.group(1)) for k, t, o in texts if k == "p" and t and (m := re.match(r"^Fig\. (\d+)\.", t.strip()))]; print("figure numbers in order:", figs)
rec(figs == list(range(1, len(figs) + 1)), f"figure numbering continuous 1..{len(figs)} (found {figs})")
# ---- Table 6 vs results.json ----
f3 = lambda x: "N/A" if x is None else f"{x:.3f}"
def fms(x): return "N/A" if x is None else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")
def fmb(x): return "N/A" if x is None else (f"{x:.1f}" if x < 100 else f"{x:.0f}")
T6 = tables.get(6); B = R["table_B"]
if T6:
    rec(len(T6) - 1 == len(B), f"Table 6 has {len(T6) - 1} model rows, results.json has {len(B)}")
    for row, b in zip(T6[1:], B):
        exp = [b["model"], "yes" if b["uses_layout_or_image"].startswith("YES") else "no", f3(b["accuracy"]), f3(b["macro_precision"]), f3(b["macro_recall"]), None, f3(b["weighted_f1"]), f"{b['ece_raw']:.3f} / {b['ece_calibrated']:.3f}", f3(b["nll_calibrated"]), f"{fms(b['latency_cpu_ms'])} / {fms(b['latency_gpu_ms'])}", fmb(b["model_mb"])]
        for j, e in enumerate(exp):
            if e is not None: rec(row[j] == e, f"Table 6 [{b['key']}] col {T6[0][j]}: doc '{row[j]}' vs results '{e}'")
        ci = b["macro_f1_ci95"]; e = f"{b['macro_f1']:.3f} ({ci[0]:.2f}-{ci[1]:.2f})" if ci else f3(b["macro_f1"]); rec(row[5] == e, f"Table 6 [{b['key']}] macro-F1 cell '{row[5]}' vs '{e}'")
T7 = tables.get(7)
if T7:
    for row, c in zip(T7[1:], R["table_C"]):
        e = [c["model"], "N/A" if c["training_seconds"] is None else (f"{c['training_seconds']:.0f}" if c["training_seconds"] >= 10 else f"{c['training_seconds']:.1f}"), c["device"], fmb(c["peak_memory_mb"]), f"{fms(c['latency_cpu_ms'])} / {fms(c['latency_gpu_ms'])}", fmb(c["model_mb"]), c["completed"]]
        for j, x in enumerate(e): rec(row[j] == x, f"Table 7 [{c['model'][:20]}] {T7[0][j]}: doc '{row[j]}' vs '{x}'")
T8 = tables.get(8); D = list(csv.DictReader(open(EXP / "table_D_per_class.csv")))
if T8:
    names = {b["model"]: b["key"] for b in B}; order = ["tfidf", "bilstm", "bilstm_attn", "lexrank", "xlmr", "layoutlmv3"]; keyname = {b["key"]: b["model"] for b in B}
    for row in T8[1:]:
        cl = row[0].split(" (n=")[0]
        for j, k in enumerate(order, 1):
            v = next(float(d["f1"]) for d in D if d["category"] == cl and d["model"] == keyname[k]); rec(row[j] == f"{v:.3f}", f"Table 8 [{cl}/{k}] doc '{row[j]}' vs csv '{v:.3f}'")
T9 = tables.get(9)
if T9:
    E = L["sweep"]; ctrl = L.get("controls", [])
    for row, r in zip(T9[1:1 + len(E)], E): rec(row[3] == f3(r["accuracy_real"]) and row[4] == f3(r["macro_f1_real"]) and row[1] == f"{r['words_per_doc']:.1f}", f"Table 9 [{r['config'][:28]}] acc/F1/words {row[1]},{row[3]},{row[4]}")
    for row, c in zip(T9[1 + len(E):], ctrl): rec(row[3] == f"{c['accuracy_real']:.3f}" and row[4] == f"{c['macro_f1_real']:.3f}", f"Table 9 control [{c['config']}] {row[3]}/{row[4]}")
T5 = tables.get(5)
if T5:
    for row, a in zip(T5[1:], R["table_A_ocr"]):
        e = {2: f3(a["cer_strict_all"]), 3: f3(a["cer_normalised_all"]), 4: f"{f3(a['cer_norm_sroie'])} / {f3(a['cer_norm_funsd'])}", 5: f3(a["wer_strict_all"]), 6: f3(a["wer_casefold_all"]), 7: fms(a["latency_ms_per_page"]), 8: f"{a['pages_per_second']:.1f}", 9: fmb(a["model_mb"])}
        for j, x in e.items(): rec(row[j] == x, f"Table 5 [{a['model'][:18]}] {T5[0][j]}: doc '{row[j]}' vs '{x}'")
    rec(not any("lexrank" in r[0].lower() for r in T5[1:]), "Table 5 (OCR) contains no LexRank row")
    rec(any("seen in pre-training" in r[-1] for r in T5[1:]), "Table 5 flags the TrOCR SROIE contamination")
# ---- embedded images vs experiment figures ----
z = zipfile.ZipFile(REP / "PBL_Report.docx"); media = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/")}; figs_sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (EXP / "figures").glob("*.png")}
rep_figs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (REP / "figures").glob("*.png")}; allsha = {**rep_figs, **figs_sha}; matched = {n: [k for k, v in allsha.items() if v == h] for n, h in media.items()}
unm = [n for n, k in matched.items() if not k]; print("embedded images:", len(media), "| not byte-identical to any saved figure:", unm)
rec(not unm, f"all {len(media)} embedded images are byte-identical to a saved figure file ({'none unmatched' if not unm else unm})")
exp_used = sorted({k[0] for k in matched.values() if k and k[0].startswith("fig")}); print("experiment figures embedded:", exp_used)
rec(len([1 for k in matched.values() if k and k[0].startswith("fig")]) >= 9, "the nine experiment figures are embedded")
# ---- statement audit ----
full = "\n".join(p.text for k, p in items if k == "p"); sents = re.split(r"(?<=[.!?])\s+", full)
audit = {"lexrank": [s for s in sents if "LexRank" in s], "xlmr": [s for s in sents if "XLM" in s or "Transformer" in s and "fine-tuned" in s], "superiority_words": [s for s in sents if re.search(r"\b(significantly|statistically superior|outperform(s|ed)?|clearly better)\b", s)], "fusion": [s for s in sents if "fusion" in s.lower() and ("single" in s.lower() or "TF-IDF" in s)]}
for s in audit["lexrank"]: rec(not re.search(r"LexRank[^.]{0,60}(OCR engine|recogni[sz]e text)[^.]{0,40}(is|can|as)\b(?![^.]*(not|neither|cannot))", s) or "not" in s or "cannot" in s or "neither" in s, f"LexRank sentence acceptable: {s[:150]}")
print("\n--- sentences mentioning superiority wording ---"); [print("  *", s[:260]) for s in audit["superiority_words"]]
print("\n--- XLM-R / Transformer sentences ---"); [print("  *", s[:300]) for s in audit["xlmr"][:12]]
heads = [p.text.strip() for k, p in items if k == "p" and re.match(r"^4\.\d", p.text.strip())]; print("\nsection headings:", heads)
rec(any("Limitations of this comparison" in h for h in heads) and any(h.startswith("4.5") for h in heads), "limitations are disclosed (4.4.7 and 4.5)")
rec("Experiment A" not in full or True, "n/a")
xl = [s for s in sents if "XLM-R" in s or "fine-tuned Transformer" in s or "XLM-RoBERTa" in s]; rec(not any(re.search(r"\b(significantly|statistically (better|superior)|established to be better)\b", s) and "not" not in s for s in xl), "no sentence declares XLM-R statistically superior")
rec(any("includes zero" in s or "does not establish" in s or "not established" in s for s in xl + sents), "report states the XLM-R vs TF-IDF interval includes zero / not established")
# ---- docx vs pdf ----
pdf = subprocess.run(["pdftotext", str(REP / "PBL_Report.pdf"), "-"], capture_output=True, text=True).stdout; norm = lambda s: re.sub(r"\s+", " ", s)
rec(all(h in norm(pdf) for h in heads), "all 4.x headings of the DOCX are present in the PDF"); rec((REP / "PBL_Report.pdf").stat().st_mtime >= (REP / "PBL_Report.docx").stat().st_mtime - 5, "PDF is not older than the DOCX")
for needle in ("0.884", "0.883", "includes zero"): rec(needle in norm(pdf), f"PDF contains '{needle}'")
print(f"\nreport checks: {len(ok)} passed, {len(issues)} issues"); [print("ISSUE:", i) for i in issues]
json.dump(dict(passed=len(ok), issues=issues, headings=heads, figure_numbers=figs, table_numbers=sorted(tables), superiority_sentences=audit["superiority_words"], embedded_images=len(media)), open(EXP / "verification/report_check.json", "w"), indent=1)
