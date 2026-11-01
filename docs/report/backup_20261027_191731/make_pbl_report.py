"""Builds docs/report/PBL_Report.docx (layout of 'PBL REPORT FORMAT.docx') + figures. Needs python-docx and matplotlib (not in requirements.txt):
PYTHONPATH=<dir with python-docx, matplotlib> .venv/bin/python scripts/make_pbl_report.py
All numbers are copied from docs/TASK_A_REPORT.md, docs/TASK_B_REPORT.md, reports/doc_type/new_classes/metrics.json and the session tests."""
import json
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
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "docs/report"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)

# ---------------- figures ----------------
def box(ax, x, y, w, h, text, fc="#dbeafe", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="#1e3a8a", lw=1)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def arrow(ax, x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#1e3a8a", lw=1.3))
fig, ax = plt.subplots(figsize=(9.2, 5.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
box(ax, 0.1, 4.8, 1.6, 0.8, "Bill upload\n(JPG / PNG / PDF)\nor e-mail attachment", "#fef9c3")
box(ax, 2.1, 4.8, 1.6, 0.8, "Validation +\npage rendering", "#e5e7eb"); box(ax, 4.1, 4.8, 1.6, 0.8, "OCR (RapidOCR,\nGPU) words+boxes", "#e5e7eb")
box(ax, 6.1, 4.8, 1.9, 0.8, "Stage 1: Task A\nexpense category\n(16 classes, fusion)", "#dbeafe")
for a, b in ((1.7, 2.1), (3.7, 4.1), (5.7, 6.1)): arrow(ax, a, 5.2, b, 5.2)
box(ax, 8.3, 4.8, 1.6, 0.8, "Task A label\n(not non_expense)\n-> file under it", "#dcfce7"); arrow(ax, 8.0, 5.2, 8.3, 5.2)
ax.text(8.1, 5.75, "expense bill", fontsize=7.5, color="#166534")
box(ax, 6.1, 3.2, 1.9, 0.9, "label = non_expense\nor Task A failed\nor low confidence", "#fee2e2"); arrow(ax, 7.05, 4.8, 7.05, 4.1)
box(ax, 6.1, 1.7, 1.9, 1.0, "Stage 2: document-type\nclassifier (12 types)\nTF-IDF + LogReg\nsynthetic-trained", "#ede9fe"); arrow(ax, 7.05, 3.2, 7.05, 2.7)
box(ax, 8.3, 2.0, 1.6, 0.8, "p >= 0.60:\nrent, salary slip,\nForm 16, ... filed", "#dcfce7"); arrow(ax, 8.0, 2.2, 8.3, 2.4)
box(ax, 8.3, 0.9, 1.6, 0.8, "p < 0.60: stays\nnon_expense", "#fee2e2"); arrow(ax, 8.0, 1.9, 8.3, 1.3)
box(ax, 2.1, 3.2, 3.5, 0.9, "Task B: 34-field extraction (hybrid of rules,\nGBDT ranker, LayoutLMv3) + optional local LLM", "#dbeafe"); arrow(ax, 4.9, 4.8, 4.0, 4.1)
box(ax, 2.1, 1.7, 3.5, 1.0, "Document profile + tax-section suggestions\n(HRA, 80C, 80D, 24(b), 80EEB, meal vouchers)\nsuggestion only - HR verifies", "#fef3c7"); arrow(ax, 3.85, 3.2, 3.85, 2.7)
box(ax, 2.1, 0.3, 7.8, 0.9, "Streamlit UI  |  Email-ingestion worker (Gmail API, read-only)  |  results/ + classified/<type>/ + HR claims database", "#f3f4f6", 9); arrow(ax, 3.85, 1.7, 3.85, 1.2)
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
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12); st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
def para(text="", bold=False, size=None, align=None, italic=False, space_after=6, style=None):
    p = doc.add_paragraph(style=style); r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if align == "c": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "j": p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after); return p
def head(t): para(t, bold=True, size=13, space_after=4)
def bullet(t): p = doc.add_paragraph(style="List Bullet"); p.add_run(t); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after = Pt(2); return p
def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color); tcPr.append(sh)
def table(rows, widths=None, fs=9, header=True):
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ""; r = c.paragraphs[0].add_run(str(v)); r.font.size = Pt(fs); r.bold = (header and i == 0)
            if header and i == 0: shade(c, "D9E2F3")
            if widths: c.width = Inches(widths[j])
    doc.add_paragraph().paragraph_format.space_after = Pt(2); return t
def figure(path, w, cap):
    doc.add_picture(str(path), width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; para(cap, italic=True, size=10, align="c", space_after=8)

para("ML-Based Automated HR Expense Claim Intelligence: Two-Stage Bill Classification, OCR Field Extraction and Tax-Document Recognition for Indian Employee Claims", bold=True, size=14, align="c")
para("[Student name(s), registration number(s)] - [Guide name, department, institute]", size=12, align="c", italic=True)
para("")
head("ABSTRACT")
for t in (
 "Employee expense claims arrive as photographs, scans and PDFs of very different bills, and HR teams sort and key them in by hand; this project automates that step end to end.",
 "We propose a pipeline that runs OCR, classifies each bill with a calibrated fusion of three models (Task A, 16 expense categories), extracts 34 fields (Task B), and passes anything the first model cannot place to a second document-type classifier covering rent receipts, rental agreements, insurance premiums, school fees, loan certificates and statements, investment proofs, meal vouchers, EV documents, Form 16, salary slips and bank statements.",
 "This matters because tax-saving documents (HRA, 80C, 80D, 24(b), 80EEB) were previously rejected as 'non_expense', and because a confidence-gated hand-off to HR keeps automation safe.",
 "On held-out real documents the Task A fusion reaches 0.907 accuracy / 0.839 macro-F1 (auto-processed share 93.0% at 94.4% precision) and the Task B hybrid reaches 0.663 micro-F1; the second stage separates its 12 classes perfectly on synthetic held-out layouts, which is evidence that the integration works, not of real-world accuracy.",
 "Against the project's own baselines the fusion improves real macro-F1 over the best single model (0.839 vs 0.831) and the hybrid extractor improves real micro-F1 over the rule baseline by 0.219; numbers are not compared with published papers because datasets differ."):
    bullet(t)
para("")
para("Keywords: expense claim automation; document classification; OCR; LayoutLMv3; information extraction; confidence calibration; two-stage classifier; Indian tax documents.", italic=True, align="j")

head("1. INTRODUCTION")
for t in (
 "Reimbursement and tax-declaration workflows in Indian organisations depend on employees uploading bills: electricity and gas statements, restaurant and fuel receipts, train and flight tickets, hotel folios, and, at year end, rent receipts, insurance premium receipts, school-fee receipts and loan-interest certificates. HR staff must decide what each document is, read its amount, date and vendor, and file it under the right head. The work is repetitive, error-prone and slow, and the documents are visually diverse: phone photographs with perspective and noise, screenshots, scanned PDFs, regional-language headings and many vendor layouts.",
 "Automating it needs three capabilities: reading the page (OCR), deciding what kind of document it is, and extracting structured fields. Recent document-AI models such as LayoutLMv3 jointly use text, layout and image and set strong results on form and receipt understanding benchmarks [1], but benchmark scores are usually reported on clean, English, largely non-Indian data, and a deployed claim system also needs calibrated confidence so that uncertain bills are routed to a person instead of being silently mis-filed.",
 "This project builds such a system as a working prototype. Task A classifies a bill into one of 16 expense categories using a calibrated fusion of a TF-IDF text model, a ConvNeXt-Tiny image model and a LayoutLMv3 model. Task B extracts up to 34 fields with a per-field hybrid of rules, a gradient-boosted candidate ranker and LayoutLMv3. A rule layer proposes income-tax sections as suggestions only; eligibility is left to HR. A Streamlit application and an e-mail ingestion worker expose the pipeline.",
 "A limitation of the first version motivates the present work: its 16 labels describe what money was spent on, so tax-evidence documents (rent receipt, Form 16, salary slip, insurance premium, school fee, loan certificate, investment proof, meal-voucher statement, EV document, bank statement) fell into the 'non_expense' class. We therefore add a second-stage document-type classifier that is consulted whenever the first model finds no expense category, and the application no longer reports 'non_expense' unless the second model also cannot place the document. No public labelled dataset exists for these Indian document types, so the second stage is trained on synthetic data only; this report states plainly what that does and does not prove."):
    para(t, align="j")

head("2. LITERATURE SURVEY")
para("Table 1 reviews recent work on document understanding, information extraction and confidence calibration (mostly 2021-2025; two earlier foundational papers are marked). For each: the existing work, its advantage, its disadvantage, and how this project addresses it.", align="j")
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
 ["[14] Guo et al., 2017 (foundational)", "Temperature scaling for calibrating neural network confidence.", "Simple, effective, one parameter.", "Calibration alone does not decide when to defer.", "Per-model temperatures plus a validation-chosen routing threshold tau = 0.61."],
 ["[15] Huang et al., 2019 (foundational)", "ICDAR 2019 SROIE: scanned receipt OCR and information extraction competition and dataset of 1,000 receipt images.", "Standard receipt benchmark.", "Not Indian bills and limited to a few key fields.", "Used as one real source among several, never as the only evidence."],
 ["[16] Mifsud et al., 2025 (Machine Learning and Knowledge Extraction)", "Receipt information extraction with a joint multi-modal transformer and a rule-based model, on a dataset of real and synthetic receipts (DocTR for OCR).", "Reports overall F1 of 0.98 on classifying receipt text, including line items; shows that a rule layer can add to the transformer.", "Receipt extraction only; no document-type or tax-evidence classification.", "We also combine learned models with a rule layer (document profile and tax suggestions) and add a second-stage document-type model."],
 ["[17] Liu et al., 2021 (Neurocomputing)", "Survey 'Document image classification: Progress over two decades': textual, structural, visual and hybrid methods on public benchmarks.", "Organises classification methods and open issues.", "Survey; benchmark document types are generic, not tax-evidence types.", "Hybrid text + image + layout fusion for Task A and a text model for the new tax-document types."],
 ["[18] Gawlikowski et al., 2023 (Artificial Intelligence Review)", "A survey of uncertainty in deep neural networks.", "Reviews ways to estimate and use model uncertainty.", "General; not specific to bills or documents.", "Calibrated confidence plus a validation-chosen routing threshold sends uncertain bills to HR."],
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
para("Table 4. Second-stage classifier (synthetic data only; 12 classes).", italic=True, size=10)
table([["Split", "Documents", "Accuracy", "Macro-F1", "Notes"], ["Validation (held-out layout)", "357", "1.000", "1.000", "no confusions"], ["Test (unseen layout / templates)", "274", "1.000", "1.000", "refit on train+val for the saved model"]], widths=[2.1, 1.0, 0.9, 0.9, 1.8])
para("End-to-end checks in the application: all 12 new document types were uploaded through the Streamlit interface; the second stage accepted and filed each under its type (confidence 0.93-0.99). A confident expense bill (food) never reached the second stage, and text the model cannot place (a generic thank-you line, empty text) stays non_expense. The application's self-test passes 32 of 32 checks, including the second-stage cases. Thirteen real bills from the held-out test split (one per expense category, e.g. electricity, gas, water, parking, train, hotel) were re-run through the full pipeline and all were classified correctly with confidence 0.989-1.000. Runtime: models load in about 7-8 s; OCR runs at roughly 170 images per minute on an RTX 4050 (6 GB); a bill takes on the order of ten seconds from upload to result in the interface.", align="j")
figure(FIG / "taskA_models.png", 4.6, "Fig. 2. Task A real macro-F1 per model: the calibrated fusion is the best.")
figure(FIG / "taskB_systems.png", 4.6, "Fig. 3. Task B real micro-F1 per system: the hybrid is the best.")
figure(FIG / "risk_coverage.png", 4.6, "Fig. 4. Task A risk-coverage on real test documents; the dashed line is the chosen threshold.")
para("4.3 Comparative analysis", bold=True)
para("The comparison above is against this project's own baselines (single models; rule, ranker and LayoutLMv3 extractors), all evaluated on the same held-out real documents: fusion beats the best single model (macro-F1 0.839 vs 0.831; accuracy 0.907 vs 0.883) and, more importantly, is better calibrated (ECE 0.020 vs 0.026-0.068), while the hybrid extractor beats the rule baseline by 0.219 micro-F1 with a paired bootstrap interval of [0.192, 0.247]. A numeric comparison with figures published in the surveyed papers (for example RVL-CDIP accuracies in the mid-to-high 90s, or SROIE/CORD field F1) would be misleading: those use different datasets, label sets and splits, and [6] shows that even the benchmark labels carry noise. Qualitatively, our system adds what the surveyed models do not: calibrated abstention, an HR route, Indian document types and a complete claim workflow.", align="j")
para("4.4 Limitations", bold=True)
for t in ("The second-stage model has seen only synthetic documents. Its 1.000 scores are a property of the generator (consistent titles and wording), not a measure of real-world accuracy; an earlier baseline trained on synthetic documents scored about 0 on real documents of other classes. Real documents of the 12 new types must be collected, with consent, before any accuracy claim is made.",
          "The second stage has no explicit reject class; the 0.60 threshold is the only guard against forcing a document into one of 12 labels. Cheque documents are not covered.",
          "Real-test support is small for several Task A classes (medical, water, internet, gas) and for Task B fields such as GSTIN, CGST, SGST; HSN/SAC has no real annotation. Task B is a research-grade baseline.",
          "LayoutLMv3 weights are non-commercial (CC-BY-NC-SA-4.0); some real documents have unverified redistribution rights, so models and data should stay internal."):
    bullet(t)

head("5. CONCLUSION")
for t in (
 "We built a working two-stage pipeline that reads Indian employee bills, classifies them, extracts fields, hands documents the first model cannot place to a second document-type classifier, and routes uncertain cases to HR.",
 "Stage 1 reaches 0.907 real accuracy and 0.839 real macro-F1 with calibrated confidence (93.0% auto-coverage at 0.944 precision), and the hybrid extractor reaches 0.663 real micro-F1, significantly above the rule baseline.",
 "The new second stage recognises 12 tax-evidence and financial document types that were previously shown as non_expense; the application now shows non_expense only when the second model also cannot place the document.",
 "Its perfect synthetic scores prove the integration, not real-world accuracy; the strongest next step is a consented collection of real rent receipts, insurance, school-fee and similar documents, followed by retraining with a reject class.",
 "Strong points of the work are group-aware, real-only evaluation, calibrated abstention, full provenance of data and an honest account of limits."):
    bullet(t)

head("6. REFERENCES")
REFS = [
 "Y. Huang, T. Lv, L. Cui, Y. Lu, F. Wei, \"LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking,\" Proc. 30th ACM International Conference on Multimedia (MM '22), Lisbon, 2022, doi:10.1145/3503161.3548112.",
 "Y. Xu, Y. Xu, T. Lv, L. Cui, F. Wei, G. Wang, Y. Lu, D. Florencio, C. Zhang, W. Che, M. Zhang, L. Zhou, \"LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding,\" Proc. 59th Annual Meeting of the ACL and 11th IJCNLP (Long Papers), pp. 2579-2591, 2021, doi:10.18653/v1/2021.acl-long.201.",
 "J. Li, Y. Xu, T. Lv, L. Cui, C. Zhang, F. Wei, \"DiT: Self-supervised Pre-training for Document Image Transformer,\" Proc. 30th ACM International Conference on Multimedia (MM '22), 2022, arXiv:2203.02378.",
 "A. Yeghiazaryan, K. Khechoyan, G. Nalbandyan, S. Muradyan, \"TokenGrid: Toward More Efficient Data Extraction From Unstructured Documents,\" IEEE Access, vol. 10, pp. 39261-39268, 2022, doi:10.1109/ACCESS.2022.3164674.",
 "Z. Yan, Z. Ye, J. Ge, J. Qin, J. Liu, Y. Cheng, C. Gurrin, \"DocExtractNet: A novel framework for enhanced information extraction from business documents,\" Information Processing & Management, vol. 62, no. 3, 2025, doi:10.1016/j.ipm.2024.104046.",
 "S. Larson, G. Lim, K. Leach, \"On Evaluation of Document Classification with RVL-CDIP,\" Proc. 17th Conference of the European Chapter of the ACL (EACL), pp. 2665-2678, 2023, doi:10.18653/v1/2023.eacl-main.195.",
 "L. Cui, Y. Xu, T. Lv, F. Wei, \"Document AI: Benchmarks, Models and Applications,\" arXiv:2111.08609, 2021.",
 "A. M. Rombach, P. Fettke, \"Deep Learning based Key Information Extraction from Business Documents: Systematic Literature Review,\" ACM Computing Surveys, vol. 58, no. 2, art. 45, 2025, doi:10.1145/3749369.",
 "W. Wang, H. Hu, Z. Zhang, Z. Li, H. Shao, D. Dahlmeier, \"Document Intelligence in the Era of Large Language Models: A Survey,\" arXiv:2510.13366, 2025.",
 "J. Wang, L. Jin, K. Ding, \"LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding,\" Proc. 60th Annual Meeting of the ACL (Long Papers), pp. 7747-7757, 2022, doi:10.18653/v1/2022.acl-long.534.",
 "S. Appalaraju, B. Jasani, B. U. Kota, Y. Xie, R. Manmatha, \"DocFormer: End-to-End Transformer for Document Understanding,\" Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2021, arXiv:2106.11539.",
 "G. Kim, T. Hong, M. Yim, J. Nam, J. Park, J. Yim, W. Hwang, S. Yun, D. Han, S. Park, \"OCR-free Document Understanding Transformer,\" Proc. European Conference on Computer Vision (ECCV), 2022, arXiv:2111.15664.",
 "Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie, \"A ConvNet for the 2020s,\" Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976-11986, 2022.",
 "C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, \"On Calibration of Modern Neural Networks,\" Proc. 34th International Conference on Machine Learning, PMLR 70, pp. 1321-1330, 2017. (foundational)",
 "Z. Huang, K. Chen, J. He, X. Bai, D. Karatzas, S. Lu, C. V. Jawahar, \"ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction,\" Proc. International Conference on Document Analysis and Recognition (ICDAR), 2019, arXiv:2103.10213. (foundational)",
 "X. Mifsud, L. Grech, A. Baldacchino, L. Keller, G. Valentino, A. Muscat, \"Receipt Information Extraction with Joint Multi-Modal Transformer and Rule-Based Model,\" Machine Learning and Knowledge Extraction, vol. 7, no. 4, art. 167, 2025, doi:10.3390/make7040167.",
 "L. Liu, Z. Wang, T. Qiu, Q. Chen, Y. Lu, C. Y. Suen, \"Document image classification: Progress over two decades,\" Neurocomputing, vol. 453, pp. 223-240, 2021, doi:10.1016/j.neucom.2021.04.114.",
 "J. Gawlikowski, C. R. N. Tassi, M. Ali, J. Lee, M. Humt, J. Feng, A. Kruspe, R. Triebel, P. Jung, R. Roscher, M. Shahzad, W. Yang, R. Bamler, X. X. Zhu, \"A survey of uncertainty in deep neural networks,\" Artificial Intelligence Review, vol. 56, suppl. 1, pp. 1513-1589, 2023, doi:10.1007/s10462-023-10562-9."]
for i, r in enumerate(REFS, 1): para(f"[{i}] {r}", size=11, align="j", space_after=3)
doc.save(OUT / "PBL_Report.docx"); print("saved", OUT / "PBL_Report.docx")
