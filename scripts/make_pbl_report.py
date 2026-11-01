"""Builds docs/report/PBL_Report.docx (layout of 'PBL REPORT FORMAT.docx') + figures. Needs python-docx and matplotlib (not in requirements.txt):
PYTHONPATH=<dir with python-docx, matplotlib> .venv/bin/python scripts/make_pbl_report.py
All numbers are copied from docs/TASK_A_REPORT.md, docs/TASK_B_REPORT.md, reports/doc_type/new_classes/metrics.json and the session tests."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "xlmr_vs_tfidf_final")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "text_vs_neural"))
try:
    import report_final as RF; RF_D = RF.load(); RF_OK = all(RF_D.get(k) for k in ("S", "cost", "tr", "om", "pw", "ph1", "tf"))
except Exception as e_: RF, RF_D, RF_OK = None, None, False; print("final study not available:", type(e_).__name__, e_)
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "docs/report"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)

# ---------------- figures ----------------
def box(ax, x, y, w, h, text, fc="#dbeafe", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="#1e3a8a", lw=1)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def arrow(ax, x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#1e3a8a", lw=1.3))
fig, ax = plt.subplots(figsize=(10, 6.4)); ax.set_xlim(0, 11); ax.set_ylim(-0.3, 6.2); ax.axis("off")
box(ax, 0.1, 5.0, 1.9, 0.9, "Bill upload\n(image, PDF or\ne-mail attachment)", "#fef9c3", 8); box(ax, 2.35, 5.0, 1.7, 0.9, "Validation and\npage rendering", "#e5e7eb", 8); box(ax, 4.4, 5.0, 1.9, 0.9, "Text recognition\n(RapidOCR) giving\nwords and boxes", "#e5e7eb", 8)
box(ax, 6.65, 5.0, 2.0, 0.9, "Stage 1 (Task A):\nexpense category,\n16 classes, fusion", "#dbeafe", 8); box(ax, 9.0, 5.0, 1.9, 0.9, "Task A label\n(not non_expense):\nfile under it", "#dcfce7", 8)
for x1, x2 in ((2.0, 2.35), (4.05, 4.4), (6.3, 6.65), (8.65, 9.0)): arrow(ax, x1, 5.45, x2, 5.45)
box(ax, 2.35, 3.5, 3.95, 0.9, "Task B: 34-field extraction\n(rules, gradient-boosted ranker,\nLayoutLMv3, optional local language model)", "#dbeafe", 8); arrow(ax, 5.2, 5.0, 4.6, 4.4)
box(ax, 6.65, 3.5, 2.0, 0.9, "label = non_expense,\nTask A failed or\nlow confidence", "#fee2e2", 8); arrow(ax, 7.65, 5.0, 7.65, 4.4)
box(ax, 6.65, 1.8, 2.0, 1.2, "Stage 2: document-type\nclassifier (12 types),\nterm-frequency model +\nlogistic regression,\ntrained on synthetic data", "#ede9fe", 7.5); arrow(ax, 7.65, 3.5, 7.65, 3.0)
box(ax, 9.0, 2.35, 1.9, 0.8, "p >= 0.60: rent,\nsalary slip, Form 16,\n... filed under type", "#dcfce7", 7.5); arrow(ax, 8.65, 2.75, 9.0, 2.75)
box(ax, 9.0, 1.15, 1.9, 0.8, "p < 0.60: stays\nnon_expense", "#fee2e2", 7.5); arrow(ax, 8.65, 2.05, 9.0, 1.6)
box(ax, 2.35, 1.8, 3.95, 1.2, "Document profile and tax-section\nsuggestions (house rent allowance,\n80C, 80D, 24(b), 80EEB, meal vouchers):\nsuggestion only, HR verifies", "#fef3c7", 7.5); arrow(ax, 4.3, 3.5, 4.3, 3.0)
box(ax, 2.35, -0.1, 8.55, 0.8, "Streamlit interface  |  e-mail ingestion worker (Gmail, read-only)\nresults folder  |  classified/<type>/ folders  |  HR claims database", "#f3f4f6", 8); arrow(ax, 4.3, 1.8, 4.3, 0.7)
plt.tight_layout(); plt.savefig(FIG / "architecture.png", dpi=200); plt.close()

def bars(path, labels, vals, title, ylabel, colors, ylim=(0, 1)):
    fig, ax = plt.subplots(figsize=(6.2, 3.3)); b = ax.bar(labels, vals, color=colors); ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    for r, v in zip(b, vals): ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.xticks(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
bars(FIG / "taskA_models.png", ["TF-IDF+LR\n(text)", "ConvNeXt-T\n(image)", "LayoutLMv3\n(text+layout)", "Calibrated\nfusion"], [0.831, 0.627, 0.704, 0.839], "Task A: real-document macro-F1 on the held-out test set", "macro-F1", ["#93c5fd", "#93c5fd", "#93c5fd", "#1d4ed8"])
bars(FIG / "taskB_systems.png", ["Rule baseline", "GBDT ranker", "LayoutLMv3", "Hybrid (final)"], [0.444, 0.546, 0.646, 0.663], "Task B: real-document micro-F1 on the held-out test set", "micro-F1", ["#fca5a5", "#fca5a5", "#fca5a5", "#b91c1c"])
fig, ax = plt.subplots(figsize=(6.2, 3.3)); th = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; cov = [0.99, 0.972, 0.956, 0.933, 0.909, 0.873, 0.824]; pr = [0.914, 0.925, 0.935, 0.944, 0.954, 0.964, 0.981]
ax.plot(th, cov, "o-", label="coverage (auto-processed share)"); ax.plot(th, pr, "s-", label="precision of auto-processed"); ax.axvline(0.61, ls="--", c="gray"); ax.text(0.615, 0.84, "tau = 0.61", fontsize=8)
ax.set_xlabel("confidence threshold"); ax.set_ylim(0.8, 1.0); ax.legend(fontsize=8); ax.set_title("Task A risk-coverage on real test documents", fontsize=10); plt.tight_layout(); plt.savefig(FIG / "risk_coverage.png", dpi=200); plt.close()

# ---------------- document ----------------
doc = Document(); sec = doc.sections[0]; sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"): setattr(sec, m, Inches(1))
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10.5); st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
import re
def roman(n):
    out = ""
    for sym, val in (("L", 50), ("XL", 40), ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1)):
        while n >= val: out += sym; n -= val
    return out
# Abbreviations: metric abbreviations are always spelled out; others are spelled out at first use (IEEE convention) and may be used afterwards.
ALWAYS = [(r"\s*\((?:CIs?|CER|WER|ECE|NLL|LR)\)", ""), (r"\bCIs\b", "confidence intervals"), (r"\bCI\b", "confidence interval"), (r"\bCER\b", "character error rate"), (r"\bWER\b", "word error rate"), (r"\bECE\b", "expected calibration error"), (r"\bNLL\b", "negative log-likelihood"),
          (r"\bLogReg\b", "logistic regression"), (r"\bLR\b", "logistic regression"), (r"\bRSS\b", "resident memory"), (r"\bp95\b", "95th-percentile"), (r"\be2e\b", "end-to-end"), (r"\bGT\b", "ground-truth")]
FIRST = [("OCR", "optical character recognition"), ("TF-IDF", "term frequency-inverse document frequency"), ("GBDT", "gradient-boosted decision tree"), ("LLM", "large language model"), ("HR", "human resources"), ("GPU", "graphics processing unit"), ("CPU", "central processing unit"),
         ("BiLSTM", "bidirectional long short-term memory"), ("HRA", "house rent allowance"), ("GSTIN", "goods and services tax identification number"), ("CGST", "central goods and services tax"), ("SGST", "state goods and services tax"), ("IGST", "integrated goods and services tax"), ("CRNN", "convolutional recurrent neural network"),
         ("CTC", "connectionist temporal classification"), ("UI", "user interface"), ("API", "application programming interface"), ("GCN", "graph convolutional network"), ("CRF", "conditional random field"), ("NLP", "natural language processing"), ("RAM", "random-access memory"), ("HSN", "harmonised system of nomenclature"), ("SAC", "services accounting code"), ("EV", "electric vehicle"), ("OOV", "out-of-vocabulary"), ("CNN", "convolutional neural network"), ("RNNs", "recurrent neural networks"), ("VQA", "visual question answering")]
_seen = set()
def expand(t, first=True):
    if not t: return t
    for rx, rep in ALWAYS: t = re.sub(rx, rep, t)
    t = re.sub(r"\bTable (\d+)\b", lambda m: f"Table {roman(int(m.group(1)))}", t)
    if first:
        for ab, full in FIRST:
            if ab in _seen: continue
            m = re.search(rf"(?<![A-Za-z0-9_-]){re.escape(ab)}(?![A-Za-z0-9_])", t)
            if m:
                _seen.add(ab); pre = t[max(0, m.start() - len(full) - 2):m.start()].lower()
                if not pre.startswith(full.lower()[:12]) and f"{full.lower()} ({ab.lower()})" not in t.lower(): t = t[:m.start()] + f"{full} ({ab})" + t[m.end():]
    return t
_buf = []
def flush():
    if _buf:
        txt = " ".join(x if x.rstrip()[-1:] in ".!?:;)" else x.rstrip() + "." for x in _buf); _buf.clear(); _para(txt, align="j")
def _para(text="", bold=False, size=None, align=None, italic=False, space_after=6, style=None, raw=False):
    text = text if raw else expand(text); p = doc.add_paragraph(style=style); r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if align == "c": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "j": p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after); return p
def para(text="", bold=False, size=None, align=None, italic=False, space_after=6, style=None, raw=False):
    flush(); m = re.match(r"^Table (\d+)\. (.*)$", text, re.S)
    if m and italic:   # IEEE table caption: TABLE V. Caption (centred, above the table)
        p = _para(f"TABLE {roman(int(m.group(1)))}. {m.group(2)}", size=9, align="c", space_after=3); p.paragraph_format.keep_with_next = True; return p
    return _para(text, bold, size, align, italic, space_after, style, raw)
def head(t): flush(); p = _para(t, bold=True, size=11, align=None, space_after=4, raw=True); p.paragraph_format.space_before = Pt(8); p.paragraph_format.keep_with_next = True; return p
def bullet(t): _buf.append(expand(t))                       # no bullet points: consecutive items are merged into one paragraph
def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color); tcPr.append(sh)
def table(rows, widths=None, fs=9, header=True):
    flush(); rows = [[expand(str(v)) for v in row] for row in rows]
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    if widths:
        for j, w in enumerate(widths): t.columns[j].width = Inches(w)
    for row_ in t.rows:      # keep each row on one page
        trPr = row_._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ""; r = c.paragraphs[0].add_run(str(v)); r.font.size = Pt(fs); r.bold = (header and i == 0)
            if header and i == 0: shade(c, "D9E2F3")
            if widths: c.width = Inches(widths[j])
    doc.add_paragraph().paragraph_format.space_after = Pt(2); return t
def figure(path, w, cap):
    flush(); doc.add_picture(str(path), width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; _para(cap, italic=False, size=9, align="c", space_after=8)

_t = para("ML-Based Automated HR Expense Claim Intelligence: Two-Stage Bill Classification, OCR Field Extraction and Tax-Document Recognition for Indian Employee Claims", bold=True, size=15, align="c", raw=True)
for _a in ("Tejvir Singh (24BAI1040)", "Chitwan Singh (24BAI1629)", "Dasari Durga Naga Sri Surya Vignesh (24BAI1589)"): para(_a, size=11, align="c", space_after=0, raw=True)
para("[Guide name, department, institute]", size=10.5, align="c", italic=True, raw=True)
para("")
_ab = ("Employee expense claims reach human resources (HR) teams as photographs, scans and PDF files of very different bills, and the staff sort and key them in by hand. "
       "This work proposes a two-stage pipeline that reads each bill with optical character recognition (OCR), classifies it into one of 16 expense categories with a calibrated fusion of a term frequency-inverse document frequency (TF-IDF) text model, an image model and a layout-aware transformer, extracts up to 34 fields, and passes any bill the first model cannot place to a second document-type classifier for rent receipts, salary slips, insurance premiums, school fees and other tax-evidence documents that were previously rejected as non-expense. "
       "The approach matters because tax-saving documents would otherwise be lost, and because a confidence-gated hand-off to HR keeps automation safe. "
       "On held-out real documents the first stage reaches 0.907 accuracy and 0.839 macro-F1, with 93.0% of bills processed automatically at 94.4% precision, and the field extractor reaches 0.663 micro-F1; the second stage scores 1.000 on synthetic held-out layouts, a saturated result that reflects the synthetic generator and shows only that the integration works, not that it will be accurate on real documents. "
       "Against the project's own baselines, the fusion is only slightly better than the best single model (the difference is not statistically established), the hybrid extractor improves micro-F1 over the rule baseline by 0.219, and a controlled comparison with recurrent, attention, summarisation-based and transformer text classifiers is reported on identical OCR text. "
       + (RF.abstract_line(RF_D) if RF_OK else "") + " Published scores are not compared directly because datasets differ.")
_p = para("", align="j", space_after=6); _r = _p.add_run("Abstract\u2014"); _r.bold = True; _r.italic = True; _p.add_run(expand(_ab)).bold = True
_p = para("", align="j", space_after=6); _r = _p.add_run("Index Terms\u2014"); _r.bold = True; _r.italic = True; _p.add_run(expand("expense claim automation, document classification, OCR, LayoutLMv3, information extraction, confidence calibration, two-stage classifier, Indian tax documents.")).italic = True
if RF_OK:
    head("SUMMARY")
    RF.exec_summary(para, bullet, RF_D)

head("1. INTRODUCTION")
for t in (
 "Reimbursement and tax-declaration workflows in Indian organisations depend on employees uploading bills: electricity and gas statements, restaurant and fuel receipts, train and flight tickets, hotel folios, and, at year end, rent receipts, insurance premium receipts, school-fee receipts and loan-interest certificates. HR staff must decide what each document is, read its amount, date and vendor, and file it under the right head. The work is repetitive, error-prone and slow, and the documents are visually diverse: phone photographs with perspective and noise, screenshots, scanned PDFs, regional-language headings and many vendor layouts.",
 "Automating it needs three capabilities: reading the page (OCR), deciding what kind of document it is, and extracting structured fields. Recent document-AI models such as LayoutLMv3 jointly use text, layout and image and set strong results on form and receipt understanding benchmarks [1], but benchmark scores are usually reported on clean, English, largely non-Indian data, and a deployed claim system also needs calibrated confidence so that uncertain bills are routed to a person instead of being silently mis-filed.",
 "This project builds such a system as a working prototype. Task A classifies a bill into one of 16 expense categories using a calibrated fusion of a TF-IDF text model, a ConvNeXt-Tiny image model and a LayoutLMv3 model. Task B extracts up to 34 fields with a per-field hybrid of rules, a gradient-boosted candidate ranker and LayoutLMv3. A rule layer proposes income-tax sections as suggestions only; eligibility is left to HR. A Streamlit application and an e-mail ingestion worker expose the pipeline.",
 "A limitation of the first version motivates the present work: its 16 labels describe what money was spent on, so tax-evidence documents (rent receipt, Form 16, salary slip, insurance premium, school fee, loan certificate, investment proof, meal-voucher statement, EV document, bank statement) fell into the 'non_expense' class. We therefore add a second-stage document-type classifier that is consulted whenever the first model finds no expense category, and the application no longer reports 'non_expense' unless the second model also cannot place the document. No public labelled dataset exists for these Indian document types, so the second stage is trained on synthetic data only; this report states plainly what that does and does not prove."):
    para(t, align="j")

head("2. LITERATURE SURVEY")
para("Table 1. Related work reviewed: existing method, advantage, disadvantage and how this work addresses it.", italic=True, size=10)
para("Table 1 reviews recent work on document understanding, information extraction, document classification and confidence calibration. Journal papers published in 2021-2025 are [4], [5], [8], [16]-[27]; conference papers and preprints ([1]-[3], [6], [7], [9]-[15]) are included because they describe models and datasets used in this work. For each paper the table gives the existing work, its advantage, its disadvantage and how this project addresses it.", align="j")
LIT = [
 ["Ref", "Existing work", "Advantage", "Disadvantage", "How this work overcomes it"],
 ["[1] Huang et al., 2022", "LayoutLMv3: unified text and image masking, word-patch alignment pre-training for Document AI.", "State of the art on form, receipt, VQA, classification and layout tasks with one architecture.", "Needs fine-tuning data; weights are CC-BY-NC-SA (non-commercial); benchmarks are not Indian bills.", "Fine-tuned on our bills and used inside a calibrated fusion and a hybrid extractor, with real-only evaluation."],
 ["[2] Xu et al., 2021", "LayoutLMv2: multi-modal pre-training with spatial-aware attention for visually-rich documents.", "Adds visual and 2-D relative-position cues to text; strong on FUNSD, CORD, SROIE.", "Heavier visual backbone; still needs OCR and in-domain labels.", "We use OCR once and share it among all models; cheaper TF-IDF model is part of the fusion."],
 ["[3] Li et al., 2022", "DiT: self-supervised pre-training of a document image transformer.", "Strong image-only document classification (RVL-CDIP) without OCR.", "Document-type labels are generic (letter, memo...), not tax-evidence types; zero-shot DiT could not separate our types.", "We trained an OCR-text document-type model for the missing Indian types."],
 ["[4] Yeghiazaryan et al., 2022 (IEEE Access)", "TokenGrid: GCN over text segments plus BiLSTM-CRF for key information extraction from unstructured documents.", "Uses text, visual and spatial cues together; efficient.", "Needs annotated field data per layout; limited public Indian data.", "Hybrid extractor picks the best system per field on validation; fields without evidence are left empty, not guessed."],
 ["[5] Yan et al., 2025 (Information Processing & Management)", "DocExtractNet: LayoutLMv3-based framework for receipts with an ImageEnhance step for low-quality images and a PrecisionHints strategy that supplements missing key-value pairs.", "Reports F1 of 97.38% on CORD (96.56% for the LayoutLMv3 baseline).", "Evaluated on CORD, FUNSD and a finance-receipt set, not on Indian bills; extraction accuracy is the focus, not routing of uncertain cases.", "We evaluate on real Indian bills and add calibrated abstention and an HR-review route."],
 ["[6] Larson, Lim and Leach, 2023 (EACL)", "Critical evaluation of RVL-CDIP: estimated 8.1% label noise, ambiguous documents, train-test overlap and personal data.", "Shows reported accuracies may be inflated.", "Diagnoses the problem but supplies no new data.", "Group-aware splits, near-duplicate audit, real-only headline metrics; synthetic scores reported separately."],
 ["[7] Cui et al., 2021", "Survey 'Document AI: Benchmarks, Models and Applications'.", "Maps tasks, datasets and pre-training approaches.", "Survey only; no deployment guidance.", "Used to select the model families (text, image, layout) combined here."],
 ["[8] Rombach and Fettke, 2025 (ACM Computing Surveys)", "Systematic literature review of deep-learning key information extraction from business documents, covering 130 approaches published 2017-2024.", "Compares approaches and datasets across the field.", "A survey: it does not supply labelled Indian tax-document data.", "Dataset audit with provenance and licence register; synthetic data clearly separated from real."],
 ["[9] Wang et al., 2025 (arXiv 2510.13366)", "Survey 'Document Intelligence in the Era of Large Language Models'.", "Covers LLM-based document understanding.", "LLM output can be wrong or ungrounded.", "Optional local LLM may only confirm values found in the OCR text and never overwrites the extractor."],
 ["[10] Wang et al., 2022 (LiLT)", "Language-independent layout transformer for structured document understanding.", "Transfers across languages by decoupling layout from text.", "Needs per-language fine-tuning for best results.", "Indian bills with Hindi headings are handled by the TF-IDF and image branches as well."],
 ["[11] Appalaraju et al., 2021 (DocFormer)", "Multi-modal transformer with shared spatial features for document understanding.", "Strong on RVL-CDIP and entity tasks.", "Large pre-training cost.", "Smaller fine-tuned models on a 6 GB GPU."],
 ["[12] Kim et al., 2022 (Donut)", "OCR-free document understanding transformer.", "Avoids OCR error propagation.", "Heavy; weaker on small noisy photos with little data.", "We keep OCR but fall back to the image model when OCR is empty."],
 ["[13] Liu et al., 2022 (ConvNeXt)", "Modernised pure-ConvNet image backbone.", "Efficient, accurate image features.", "Image-only; weak at reading amounts or titles.", "Used as the image branch of the fusion."],
 ["[14] Guo et al., 2017", "Temperature scaling for calibrating neural network confidence.", "Simple, effective, one parameter.", "Calibration alone does not decide when to defer.", "Per-model temperatures plus a validation-chosen routing threshold tau = 0.61."],
 ["[15] Huang et al., 2019", "ICDAR 2019 SROIE: scanned receipt OCR and information extraction competition and dataset of 1,000 receipt images.", "Standard receipt benchmark.", "Not Indian bills and limited to a few key fields.", "Used as one real source among several, never as the only evidence."],
 ["[16] Mifsud et al., 2025 (Machine Learning and Knowledge Extraction)", "Receipt information extraction with a joint multi-modal transformer and a rule-based model, on a dataset of real and synthetic receipts (DocTR for OCR).", "Reports overall F1 of 0.98 on classifying receipt text, including line items; shows that a rule layer can add to the transformer.", "Receipt extraction only; no document-type or tax-evidence classification.", "We also combine learned models with a rule layer (document profile and tax suggestions) and add a second-stage document-type model."],
 ["[17] Liu et al., 2021 (Neurocomputing)", "Survey 'Document image classification: Progress over two decades': textual, structural, visual and hybrid methods on public benchmarks.", "Organises classification methods and open issues.", "Survey; benchmark document types are generic, not tax-evidence types.", "Hybrid text + image + layout fusion for Task A and a text model for the new tax-document types."],
 ["[18] Gawlikowski et al., 2023 (Artificial Intelligence Review)", "A survey of uncertainty in deep neural networks.", "Reviews ways to estimate and use model uncertainty.", "General; not specific to bills or documents.", "Calibrated confidence plus a validation-chosen routing threshold sends uncertain bills to HR."],
 ["[19] Ha and Horak, 2022 (Signal Processing: Image Communication)", "OCRMiner: extracts indexing metadata from scanned invoices by combining text analysis with layout features on top of standard OCR output.", "Works from possibly erroneous OCR output; the authors report recovering invoice data in 90% of English and 88% of Czech invoices with an open-source OCR.", "Targets (semi-)structured invoices and metadata extraction; it does not decide an expense category or hand uncertain documents to a person.", "We also read OCR output with layout cues (LayoutLMv3 branch) but add expense classification, a second document-type stage and a confidence-gated HR route."],
 ["[20] Baviskar et al., 2021 (IEEE Access)", "A high-quality multi-layout unstructured invoice dataset, assessed with a statistical data-validation technique and evaluated with several feature-extraction approaches for key-field extraction.", "Diverse invoice layouts and a documented validation process address the shortage of annotated invoice data.", "Covers invoice documents for field extraction; it does not provide expense-category or tax-evidence labels.", "We keep a provenance and licence register, split by template group and report real and synthetic scores separately."],
 ["[21] Saifullah et al., 2024 (IEEE Access)", "Studies nine explainability methods on ten document-classification models using RVL-CDIP and Tobacco3482; most methods give poor explanations, and only Occlusion and DeepSHAP are relatively faithful.", "Shows that high benchmark accuracy does not mean a model relies on meaningful evidence.", "An analysis study: it does not offer a deployable classifier or a routing rule.", "Our main text classifier is an inspectable linear model, and every model is judged on real-only metrics with calibrated confidence."],
 ["[22] Omurca et al., 2023 (Applied Intelligence)", "Classifies twelve types of university student documents uploaded as PDF using (i) OCR with traditional machine learning, (ii) deep learning and (iii) an entropy-based fusion of deep models; the best F-score is 94.45% (EfficientNetB3 with ExtraTree).", "Shows that combining text-based and image-based models improves document classification.", "One institution and twelve document types; no calibrated abstention or tax-evidence types.", "Calibrated fusion of text, image and layout models with a validation-chosen threshold, plus a second stage for tax documents."],
 ["[23] Gbada et al., 2024 (Neurocomputing)", "Models a visually rich document as a weighted graph whose nodes carry visual, textual and spatial features and classifies nodes with a graph convolutional network; evaluated on invoices and receipts.", "Uses relations between neighbouring text regions; reported to match or exceed strong baselines.", "Needs node-level field annotations; benchmark documents are not Indian bills.", "A per-field choice between rules, a candidate ranker and LayoutLMv3 is made on validation data; fields without evidence are left empty."],
 ["[24] Gbada et al., 2024 (International Journal on Document Analysis and Recognition)", "Survey of deep-learning approaches for information extraction from visually rich documents: datasets, challenges and methods.", "Organises methods, datasets and open problems in one place.", "A survey; it supplies no labelled Indian tax-document data.", "Used to choose the model families and to document dataset gaps."],
 ["[25] Balanya et al., 2024 (Neural Computing and Applications)", "Studies post-hoc confidence scaling that generalises temperature scaling; shows that complex calibrators fail when data are limited and proposes entropy-based temperature scaling that is robust to data scarcity.", "Guidance on calibrating when only a small validation set exists.", "Evaluated on standard classification benchmarks, not on documents.", "We fit one temperature per model on the validation set only, which suits our small validation sets."],
 ["[26] Fernando and Tsokos, 2022 (IEEE Transactions on Neural Networks and Learning Systems)", "A dynamically weighted balanced loss for class-imbalanced learning that is classification-calibrated, tested on cyber-intrusion and medical-imaging data.", "Addresses minority-class errors and confidence calibration together.", "Not tested on document classification.", "Our models use balanced class weights and we report macro-F1 and per-class scores so that minority-class errors are visible."],
 ["[27] Luxemburk and Cejka, 2023 (Computer Networks)", "Releases a dataset with almost 200 fine-grained service labels and a neural network with a reject option; it reports 97.04% accuracy and detection of 91.94% of unknown services at a 5% false-positive rate.", "Demonstrates that a classifier can refuse inputs of classes it has not seen.", "Applied to encrypted network traffic, not to documents.", "The second stage accepts a label only at probability 0.60 or higher and otherwise leaves the document as non_expense for HR review."],
]
table(LIT, widths=[0.9, 1.6, 1.4, 1.4, 1.6], fs=8)
para("Summary: published work shows that multimodal transformers are accurate on benchmark documents, that benchmark labels can be noisy [6], and that confidence calibration is needed for deployment [14]. None of the surveyed work provides a classifier for Indian tax-evidence document types or an end-to-end claim workflow with a human-review route; this project fills that gap with the limits stated in Section 4.", align="j")

head("3. PROPOSED WORK")
for t in (
 "The system accepts a bill image or PDF (or an e-mail attachment), validates it, renders pages and runs RapidOCR on the GPU once. Stage 1 (Task A) fuses three classifiers - TF-IDF + logistic regression on OCR text, ConvNeXt-Tiny on the image and LayoutLMv3 on text, layout and image - after per-model temperature scaling, with weights 0.5 / 0.3 / 0.2 selected on validation only. A bill whose fused confidence is below tau = 0.61 is flagged for HR review. In parallel, Task B extracts 34 fields (merchant, invoice number and date, totals, GSTIN, CGST/SGST/IGST, travel and hotel fields and so on) with a per-field choice between a rule baseline, a gradient-boosted candidate ranker and a LayoutLMv3 token classifier.",
 "Stage 2 (new) is a document-type classifier for the categories Stage 1 has no label for: rent_receipt, rental_agreement, insurance_premium_receipt, school_fee_receipt, loan_interest_certificate, loan_statement, investment_certificate, meal_voucher_statement, ev_document, form_16, salary_slip and bank_statement (cheque has no approved data and is not covered). It is a TF-IDF (character 2-5 grams + word 1-2 grams) logistic-regression model on OCR text, trained only on synthetic documents generated by the project (11 types, 110 documents each) plus a synthetic bank-statement set, with whole layouts held out for validation and test. It is invoked only when Stage 1 returns 'non_expense', fails, or is below the routing threshold, and its answer is accepted only at probability >= 0.60. When accepted for a 'non_expense' bill, the new type replaces the label everywhere in the application and the bill is filed under it; otherwise the bill stays non_expense. The result is always marked for HR review because the model has never seen a real document of these types."):
    para(t, align="j")
figure(FIG / "architecture.png", 6.3, "Fig. 1. Architecture and flow of the two-stage bill classification and extraction pipeline.")
para("Implementation notes: Python, PyTorch, scikit-learn, RapidOCR (ONNX), Streamlit; the e-mail worker polls a Gmail mailbox read-only and processes only registered senders. The second stage lives in app/second_classifier.py and is wired in app/service.py; the training script is scripts/doc_type_new/train_new_classes.py.", align="j")

head("4. PERFORMANCE ANALYSIS")
para("4.1 Parameters, data and testing", bold=True)
for t in (
 "Data. The expense dataset has 7,924 documents (5,494 real, 2,430 synthetic Indian). Splits are by document group (template, merchant, near-duplicate family) with an independent audit finding no validation or test image within perceptual-hash distance 4 of training. Task A: train 4,848 (2,904 real), validation 634 (391 real), test 629 (386 real). Task B: 6,442 documents (2,932 real, 3,510 synthetic), split 5,160 / 646 / 636. All hyper-parameters, fusion weights, temperatures and thresholds were chosen on validation data; the test set was evaluated once per model.",
 "Metrics. Real-document accuracy and macro-F1 are the headline numbers because templated synthetic bills score about 1.0 for every model. Confidence intervals are bootstrap over document groups. Expected calibration error (ECE) measures how well confidence matches correctness; auto-coverage and auto-precision describe the HR routing rule. Task B uses per-field precision, recall and F1 and the micro-F1 over fields.",
 "Second-stage test design. The 12 new classes were trained on two layouts, validated on a third and tested on an unseen layout (layout D; held-out templates for bank statements): 357 validation and 274 test documents. To choose the acceptance threshold, we measured the maximum second-stage probability on 600 synthetic test bills of the existing types: median 0.18, 99th percentile 0.47, while correct new-type documents scored at least 0.93 at the 1st percentile; the threshold was therefore set to 0.60."):
    para(t, align="j")
para("4.2 Results obtained", bold=True)
para("Table 2. Task A (expense category) on the held-out test set, real documents.", italic=True, size=10)
table([["Model", "Real acc.", "Real macro-F1 (95% CI)", "Real-Indian acc.", "ECE"],
       ["TF-IDF + LogReg (OCR text)", "0.883", "0.831 (0.75-0.87)", "0.936", "0.026"], ["ConvNeXt-Tiny (image)", "0.800", "0.627 (0.54-0.67)", "0.836", "0.068"],
       ["LayoutLMv3-base", "0.863", "0.704 (0.62-0.77)", "0.909", "0.045"], ["Calibrated fusion (final)", "0.907", "0.839 (0.75-0.89)", "0.955", "0.020"]], widths=[2.3, 0.9, 1.7, 1.2, 0.6])
para("Routing: with tau = 0.61 the test result is 93.0% auto-coverage at 0.944 auto-precision (27 of 386 real documents sent to HR). Group 5-fold cross-validation of the text model gives 0.857 real accuracy / 0.838 macro-F1.", align="j")
para("Table 3. Task B (field extraction), real test documents.", italic=True, size=10)
table([["System", "Real micro-F1 (95% CI)", "Real macro-F1", "Indian-real micro-F1"], ["Rule baseline", "0.444 (0.414-0.480)", "0.358", "0.467"], ["GBDT candidate ranker", "0.546 (0.501-0.588)", "0.451", "0.449"],
       ["LayoutLMv3", "0.646 (0.606-0.685)", "0.549", "0.534"], ["Hybrid (final)", "0.663 (0.621-0.697)", "0.578", "0.599"]], widths=[2.2, 1.9, 1.3, 1.5])
para("Per-field real F1 of the hybrid: total_amount 0.820, tax_total 0.821, merchant_address 0.829, subtotal 0.765, invoice_date 0.750, merchant_name 0.543, invoice_number 0.422. Merchant name and invoice number remain weak and should be verified by a person.", align="j")
para("Table 4. Second-stage classifier (synthetic data only; 12 classes). The perfect scores are a property of the synthetic generator, not an estimate of real-world accuracy.", italic=True, size=10)
_sat = json.loads((ROOT / "reports/doc_type/new_classes/saturation_check.json").read_text()) if (ROOT / "reports/doc_type/new_classes/saturation_check.json").exists() else None
table([["Evaluation", "Documents", "Accuracy", "Macro-F1", "Notes"], ["Validation (held-out layout)", "357", "1.000", "1.000", "no confusions"], ["Test (unseen layout / templates)", "274", "1.000", "1.000", "refit on train+val for the saved model"]]
      + ([["Test, class-name keywords removed", str(_sat["n_test"]), f"{_sat['accuracy_keywords_removed']:.3f}", "-", "diagnostic: model unchanged"], ["Test, half of the words dropped at random", str(_sat["n_test"]), f"{_sat['accuracy_half_words_dropped']:.3f}", "-", "mean of 5 random draws"], ["Test, keywords removed and half of the words dropped", str(_sat["n_test"]), f"{_sat['accuracy_keywords_removed_and_half_words_dropped']:.3f}", "-", "mean of 5 random draws"]] if _sat else []), widths=[2.3, 0.9, 0.8, 0.8, 1.9])
para("Why the accuracy is 1.000. The held-out 'layout' differs in page design, but every document of a type is produced by the same generator, so documents of one type share the same field labels, number formats and phrasing, and the types differ from one another in many words, not only in their titles. The classifier is a linear model on word and character n-grams, which separates such classes trivially. Removing the class-name keywords, or dropping half of the words at random, leaves the accuracy at 1.000 (Table IV), which shows that the score does not depend on a few title words and that the task is saturated. Bank statements, taken from a different source, are also 100 of the 274 test documents. A real rent receipt, salary slip or insurance receipt varies in wording, language, layout and print quality far more than this generator does, and the accuracy on real documents is unknown and expected to be much lower. For this reason the second stage keeps the 0.60 acceptance threshold, marks every accepted result for HR review, and no real-world accuracy is claimed for it.", align="j")
para("End-to-end checks in the application: all 12 new document types were uploaded through the Streamlit interface; the second stage accepted and filed each under its type (confidence 0.93-0.99). A confident expense bill (food) never reached the second stage, and text the model cannot place (a generic thank-you line, empty text) stays non_expense. The application's self-test passes 32 of 32 checks, including the second-stage cases. Thirteen real bills from the held-out test split (one per expense category, e.g. electricity, gas, water, parking, train, hotel) were re-run through the full pipeline and all were classified correctly with confidence 0.989-1.000. Runtime: models load in about 7-8 s; OCR runs at roughly 170 images per minute on an RTX 4050 (6 GB); a bill takes on the order of ten seconds from upload to result in the interface.", align="j")
figure(FIG / "taskA_models.png", 4.6, "Fig. 2. Task A real macro-F1 per model: the calibrated fusion is the best.")
figure(FIG / "taskB_systems.png", 4.6, "Fig. 3. Task B real micro-F1 per system: the hybrid is the best.")
figure(FIG / "risk_coverage.png", 4.6, "Fig. 4. Task A risk-coverage on real test documents; the dashed line is the chosen threshold.")
para("4.3 Comparative analysis", bold=True)
para("The comparison above is against this project's own baselines (single models; rule, ranker and LayoutLMv3 extractors), all evaluated on the same held-out real documents: fusion scores slightly higher than the best single model (macro-F1 0.839 vs 0.831; accuracy 0.907 vs 0.883), but a paired bootstrap (Section 4.4) finds the difference from TF-IDF alone not statistically established, and its ECE (0.020) is only modestly lower than TF-IDF's (0.026) without an interval; the hybrid extractor the hybrid extractor beats the rule baseline by 0.219 micro-F1 with a paired bootstrap interval of [0.192, 0.247]. A numeric comparison with figures published in the surveyed papers (for example RVL-CDIP accuracies in the mid-to-high 90s, or SROIE/CORD field F1) would be misleading: those use different datasets, label sets and splits, and [6] shows that even the benchmark labels carry noise. Qualitatively, our system adds what the surveyed models do not: calibrated abstention, an HR route, Indian document types and a complete claim workflow.", align="j")
try:                                           # 4.4 (earlier comparison, single XLM-R run) is generated from experiments/text_vs_neural/outputs; skipped if those results do not exist
    import report_section_4_4 as RS
    if (RS.OUT / "results.json").exists(): ctx_ = RS.build(para, bullet, table, figure); RS.interpret(ctx_, para, bullet)
except Exception as e_: print("comparison section skipped:", type(e_).__name__, e_)
if RF_OK:                                      # 4.5 final multi-seed study + cost analysis (experiments/xlmr_vs_tfidf_final/outputs)
    RF.build_main(para, bullet, table, figure, RF_D); RF.worth_it(para, bullet, table, figure, RF_D); RF.others_and_end(para, bullet, table, figure, RF_D)
para("4.6 Limitations", bold=True)
for t in ("The second-stage model has seen only synthetic documents. Its 1.000 scores are a property of the generator (consistent titles and wording), not a measure of real-world accuracy; an earlier baseline trained on synthetic documents scored about 0 on real documents of other classes. Real documents of the 12 new types must be collected, with consent, before any accuracy claim is made.",
          "The second stage has no explicit reject class; the 0.60 threshold is the only guard against forcing a document into one of 12 labels. Cheque documents are not covered.",
          "Real-test support is small for several Task A classes (medical, water, internet, gas) and for Task B fields such as GSTIN, CGST, SGST; HSN/SAC has no real annotation. Task B is a research-grade baseline.",
          "LayoutLMv3 weights are non-commercial (CC-BY-NC-SA-4.0); some real documents have unverified redistribution rights, so models and data should stay internal."):
    bullet(t)

head("5. CONCLUSION")
for t in (
 "We built a working two-stage pipeline that reads Indian employee bills, classifies them, extracts fields, hands documents the first model cannot place to a second document-type classifier, and routes uncertain cases to HR.",
 "Stage 1 reaches 0.907 real accuracy and 0.839 real macro-F1 with calibrated confidence (93.0% auto-coverage at 0.944 precision), and the hybrid extractor reaches 0.663 real micro-F1, significantly above the rule baseline.",
 (RF.conclusion_line(RF_D) if RF_OK else "A measured comparison on identical OCR text (Section 4.4) is reported; see that section for its limits."),
 "The new second stage recognises 12 tax-evidence and financial document types that were previously shown as non_expense; the application now shows non_expense only when the second model also cannot place the document.",
 "Its perfect synthetic scores prove the integration, not real-world accuracy; the strongest next step is a consented collection of real rent receipts, insurance, school-fee and similar documents, followed by retraining with a reject class.",
 "Strong points of the work are group-aware, real-only evaluation, calibrated abstention, full provenance of data and an honest account of limits."):
    bullet(t)

head("6. REFERENCES")
REFS = [
 "Y. Huang, T. Lv, L. Cui, Y. Lu, F. Wei, \"LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking,\" Proc. 30th ACM International Conference on Multimedia (MM '22), pp. 4083-4091, 2022, doi:10.1145/3503161.3548112.",
 "Y. Xu, Y. Xu, T. Lv, L. Cui, F. Wei, G. Wang, Y. Lu, D. Florencio, C. Zhang, W. Che, M. Zhang, L. Zhou, \"LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding,\" Proc. 59th Annual Meeting of the ACL and 11th IJCNLP (Long Papers), pp. 2579-2591, 2021, doi:10.18653/v1/2021.acl-long.201.",
 "J. Li, Y. Xu, T. Lv, L. Cui, C. Zhang, F. Wei, \"DiT: Self-supervised Pre-training for Document Image Transformer,\" Proc. 30th ACM International Conference on Multimedia (MM '22), 2022, arXiv:2203.02378.",
 "A. Yeghiazaryan, K. Khechoyan, G. Nalbandyan, S. Muradyan, \"TokenGrid: Toward More Efficient Data Extraction From Unstructured Documents,\" IEEE Access, vol. 10, pp. 39261-39268, 2022, doi:10.1109/ACCESS.2022.3164674.",
 "Z. Yan, Z. Ye, J. Ge, J. Qin, J. Liu, Y. Cheng, C. Gurrin, \"DocExtractNet: A novel framework for enhanced information extraction from business documents,\" Information Processing & Management, vol. 62, no. 3, 2025, doi:10.1016/j.ipm.2024.104046.",
 "S. Larson, G. Lim, K. Leach, \"On Evaluation of Document Classification with RVL-CDIP,\" Proc. 17th Conference of the European Chapter of the ACL (EACL), pp. 2665-2678, 2023, doi:10.18653/v1/2023.eacl-main.195.",
 "L. Cui, Y. Xu, T. Lv, F. Wei, \"Document AI: Benchmarks, Models and Applications,\" arXiv:2111.08609, 2021.",
 "A. M. Rombach, P. Fettke, \"Deep Learning based Key Information Extraction from Business Documents: Systematic Literature Review,\" ACM Computing Surveys, vol. 58, no. 2, pp. 1-37, 2025, doi:10.1145/3749369.",
 "W. Wang, H. Hu, Z. Zhang, Z. Li, H. Shao, D. Dahlmeier, \"Document Intelligence in the Era of Large Language Models: A Survey,\" arXiv:2510.13366, 2025.",
 "J. Wang, L. Jin, K. Ding, \"LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding,\" Proc. 60th Annual Meeting of the ACL (Long Papers), pp. 7747-7757, 2022, doi:10.18653/v1/2022.acl-long.534.",
 "S. Appalaraju, B. Jasani, B. U. Kota, Y. Xie, R. Manmatha, \"DocFormer: End-to-End Transformer for Document Understanding,\" Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2021, arXiv:2106.11539.",
 "G. Kim, T. Hong, M. Yim, J. Nam, J. Park, J. Yim, W. Hwang, S. Yun, D. Han, S. Park, \"OCR-free Document Understanding Transformer,\" Proc. European Conference on Computer Vision (ECCV), 2022, arXiv:2111.15664.",
 "Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie, \"A ConvNet for the 2020s,\" Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976-11986, 2022.",
 "C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, \"On Calibration of Modern Neural Networks,\" Proc. 34th International Conference on Machine Learning, PMLR 70, pp. 1321-1330, 2017.",
 "Z. Huang, K. Chen, J. He, X. Bai, D. Karatzas, S. Lu, C. V. Jawahar, \"ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction,\" Proc. International Conference on Document Analysis and Recognition (ICDAR), 2019, arXiv:2103.10213.",
 "X. Mifsud, L. Grech, A. Baldacchino, L. Keller, G. Valentino, A. Muscat, \"Receipt Information Extraction with Joint Multi-Modal Transformer and Rule-Based Model,\" Machine Learning and Knowledge Extraction, vol. 7, no. 4, p. 167, 2025, doi:10.3390/make7040167.",
 "L. Liu, Z. Wang, T. Qiu, Q. Chen, Y. Lu, C. Y. Suen, \"Document image classification: Progress over two decades,\" Neurocomputing, vol. 453, pp. 223-240, 2021, doi:10.1016/j.neucom.2021.04.114.",
 "J. Gawlikowski, C. R. N. Tassi, M. Ali, J. Lee, M. Humt, J. Feng, A. Kruspe, R. Triebel, P. Jung, R. Roscher, M. Shahzad, W. Yang, R. Bamler, X. X. Zhu, \"A survey of uncertainty in deep neural networks,\" Artificial Intelligence Review, vol. 56, suppl. 1, pp. 1513-1589, 2023, doi:10.1007/s10462-023-10562-9.",
 "H. T. Ha, A. Horak, \"Information extraction from scanned invoice images using text analysis and layout features,\" Signal Processing: Image Communication, vol. 102, p. 116601, 2022, doi:10.1016/j.image.2021.116601.",
 "D. Baviskar, S. Ahirrao, K. Kotecha, \"Multi-Layout Unstructured Invoice Documents Dataset: A Dataset for Template-Free Invoice Processing and Its Evaluation Using AI Approaches,\" IEEE Access, vol. 9, pp. 101494-101512, 2021, doi:10.1109/ACCESS.2021.3096739.",
 "Saifullah, S. Agne, A. Dengel, S. Ahmed, \"The Reality of High Performing Deep Learning Models: A Case Study on Document Image Classification,\" IEEE Access, vol. 12, pp. 103537-103564, 2024, doi:10.1109/ACCESS.2024.3425910.",
 "S. I. Omurca, E. Ekinci, S. Sevim, E. B. Edinc, S. Eken, A. Sayar, \"A document image classification system fusing deep and machine learning models,\" Applied Intelligence, vol. 53, no. 12, pp. 15295-15310, 2023, doi:10.1007/s10489-022-04306-5.",
 "H. Gbada, K. Kalti, M. A. Mahjoub, \"Multimodal weighted graph representation for information extraction from visually rich documents,\" Neurocomputing, vol. 573, p. 127223, 2024, doi:10.1016/j.neucom.2023.127223.",
 "H. Gbada, K. Kalti, M. A. Mahjoub, \"Deep learning approaches for information extraction from visually rich documents: datasets, challenges and methods,\" International Journal on Document Analysis and Recognition (IJDAR), 2024 (online first), doi:10.1007/s10032-024-00493-8.",
 "S. A. Balanya, J. Maronas, D. Ramos, \"Adaptive temperature scaling for robust calibration of deep neural networks,\" Neural Computing and Applications, vol. 36, no. 14, pp. 8073-8095, 2024, doi:10.1007/s00521-024-09505-4.",
 "K. R. M. Fernando, C. P. Tsokos, \"Dynamically Weighted Balanced Loss: Class Imbalanced Learning and Confidence Calibration of Deep Neural Networks,\" IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 7, pp. 2940-2951, 2022, doi:10.1109/TNNLS.2020.3047335.",
 "J. Luxemburk, T. Cejka, \"Fine-grained TLS services classification with reject option,\" Computer Networks, vol. 220, p. 109467, 2023, doi:10.1016/j.comnet.2022.109467."]
for i, r in enumerate(REFS, 1): para(f"[{i}] {r}", size=9, align="j", space_after=3, raw=True)
doc.save(OUT / "PBL_Report.docx"); print("saved", OUT / "PBL_Report.docx")
