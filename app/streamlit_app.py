"""HR Bill Classifier - local demo UI.  Launch:  bash app/run_app.sh   ->  http://127.0.0.1:8501
Upload bill -> OCR -> Task A expense category -> Task B V1 field extraction -> saved under outputs/classified/<category>/"""
import sys, json
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import service as S
import tax_ui
import inbox_ui
import os; os.umask(0o077)   # new files (results, originals, databases) are private to the service account

st.set_page_config(page_title="HR Bill Classifier", page_icon="🧾", layout="wide")

@st.cache_resource(show_spinner="Loading models (OCR, Task A, Task B) - first start only...")
def get_service():
    return S.BillService().load()

KEY_FIELDS = [("merchant_name", "Vendor / merchant"), ("invoice_number", "Invoice number"), ("invoice_date", "Invoice date"), ("gstin", "GSTIN"),
              ("subtotal", "Subtotal"), ("tax_total", "Tax total"), ("cgst", "CGST"), ("sgst", "SGST"), ("igst", "IGST"), ("hsn_sac", "HSN/SAC"),
              ("total_amount", "Total amount"), ("currency", "Currency"), ("payment_method", "Payment method")]

def fmt_conf(c): return "—" if c is None else f"{c:.2f}"
def abs_path(rel): return str((S.OUT / rel).resolve())

# ---------------- navigation ----------------
if "pending_nav" in st.session_state: st.session_state["page"] = st.session_state.pop("pending_nav")
st.sidebar.title("🧾 HR Bill Classifier")
page = st.sidebar.radio("Navigate", ["Upload Bill", "Results", "Classified Bills", "Email Inbox", "Tax Declaration", "HR Review", "Tax Summary"], key="page")
recs_all = S.list_records()
st.sidebar.markdown("---")
st.sidebar.caption(f"Processed bills on disk: **{len(recs_all)}**")
role, actor, role_ok = tax_ui.identity()
# tenant isolation: a record owned by someone else is invisible (HR sees all; unowned legacy demo uploads stay shared)
recs_all = [r for r in recs_all if r.get("owner") in (None, "") or role == "hr" or (role_ok and r.get("owner") == actor)]
svc = get_service()
try:                                  # two saved demo employee profiles (E1, E2): written once into hr.db, survive restarts, never overwrite edits
    from tax import quota as _Q; _Q.seed_default_profiles()
except Exception as _e: st.sidebar.warning(f"Default employee profiles unavailable: {type(_e).__name__}")
for k, v in svc.errors.items(): st.sidebar.error(f"{k} unavailable: {v}")
st.sidebar.caption(f"Models loaded in {svc.load_seconds}s · predictions only, not verified")

# ---------------- page 1: upload ----------------
def page_upload():
    st.title("HR Bill Classifier")
    st.write("Upload a bill (JPG, PNG or PDF). The app reads it with OCR, classifies the **expense category** (what the money was spent on), extracts the bill fields, and files the bill in its category folder.")
    up = st.file_uploader("Drag and drop a bill here, or browse", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=False, key="uploader")
    if up is None:
        st.info("No file selected yet.")
        return
    data = up.getvalue()
    try:
        kind, ext = S.validate_upload(data, up.name)
        prev = S.preview_first_page(data, kind)
    except S.UserError as e:
        st.error(str(e)); return
    except Exception as e:                   # never show a raw traceback for an unreadable upload
        st.error(f"This file could not be read ({type(e).__name__}). Please upload a different JPG, PNG or PDF."); return
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Preview")
        st.image(prev, caption=f"{up.name} ({len(data) / 1024:.0f} KB)" + (" - page 1 shown" if kind == "pdf" else ""), use_container_width=True)
    with c2:
        st.subheader("Ready to process")
        st.write(f"**File:** {up.name}  \n**Type:** {kind.upper()}  \n**Size:** {len(data) / 1e6:.2f} MB")
        if kind == "pdf": st.caption(f"PDFs: all pages (up to {S.MAX_PDF_PAGES}) are OCR'd; classification and field extraction use page 1 only.")
        if st.button("Process Bill", type="primary", use_container_width=True):
            with st.status("Processing bill...", expanded=True) as status:
                st.write("Reading the file, running OCR, classifying, extracting fields, saving...")
                try:
                    rec = svc.process(data, up.name, owner=(actor if role == 'employee' and role_ok else None))
                except S.UserError as e:
                    status.update(label="Upload rejected", state="error"); st.error(str(e)); return
                except Exception as e:
                    status.update(label="Processing failed", state="error"); st.error(f"Unexpected error: {type(e).__name__}: {e}"); return
                state = "complete" if rec["status"] in ("done",) else "error" if rec["status"] in ("classification_failed", "ocr_failed", "ocr_empty", "error") else "complete"
                status.update(label=f"Finished: {rec['status']}", state=state)
            st.session_state["current_id"] = rec["doc_id"]
            st.session_state["pending_nav"] = "Results"
            st.rerun()

# ---------------- page 2: results ----------------
def draw_box(img_path, bbox):
    im = Image.open(img_path).convert("RGB"); d = ImageDraw.Draw(im)
    if bbox: d.rectangle(bbox, outline=(255, 0, 0), width=max(3, im.width // 300))
    return im

def page_results():
    st.title("Results")
    if not recs_all: st.info("No bills processed yet. Use **Upload Bill**."); return
    ids = [r["doc_id"] for r in reversed(recs_all)]
    cur = st.session_state.get("current_id", ids[0]); cur = cur if cur in ids else ids[0]
    labels = {r["doc_id"]: f"{r['doc_id']} · {r['original_filename']} · {r.get('category') or r['status']}" for r in recs_all}
    sel = st.selectbox("Bill", ids, index=ids.index(cur), format_func=lambda i: labels[i]); st.session_state["current_id"] = sel
    rec = S.load_record(sel); rdir = S.OUT / rec["result_dir"]
    for w in rec.get("warnings", []): st.warning(w)
    left, right = st.columns([5, 6])
    # ---- D. original document ----
    with left:
        st.subheader("Original document")
        pages = sorted([*rdir.glob("page_*.jpg"), *rdir.glob("page_*.png")], key=lambda p: int(p.stem.split("_")[1]))
        hl = None
        flds = ((rec.get("task_b") or {}).get("fields") or {})
        with_box = [f for f, v in flds.items() if v.get("bbox")]
        if with_box:
            hl = st.selectbox("Highlight an extracted field on page 1", ["(none)"] + with_box)
        if len(pages) > 1:
            tabs = st.tabs([f"Page {i + 1}" for i in range(len(pages))])
            for i, (t, p) in enumerate(zip(tabs, pages)):
                with t: st.image(draw_box(p, flds[hl]["bbox"]) if (i == 0 and hl and hl != "(none)") else str(p), use_container_width=True)
        elif pages:
            st.image(draw_box(pages[0], flds[hl]["bbox"]) if hl and hl != "(none)" else str(pages[0]), use_container_width=True)
        saved = S.OUT / rec["saved_path"]
        if saved.exists(): st.download_button("Download original file", saved.read_bytes(), file_name=rec["original_filename"], key="dl_orig")
    with right:
        T = st.tabs(["Summary", "Fields", "Details"]); T[0].__enter__()        # Summary: classification, where it was saved, tax rebate | Fields: extraction | Details: profile, suggestions, OCR
        # ---- A. classification + E. saved ----
        st.subheader("A. Classification")
        ta = rec.get("task_a")
        if ta:
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div style='font-size:0.85rem;opacity:.7'>Predicted category</div><div style='font-size:1.6rem;line-height:1.2;word-break:break-word'>{ta['label']}</div>", unsafe_allow_html=True); m2.metric("Confidence", f"{ta['confidence']:.1%}"); m3.metric("Routing", "Auto" if ta["route"] == "auto" else "HR review")
            st.caption(f"Category group: {ta['parent_group']} · {'acceptance threshold' if ta.get('source') == 'second_classifier' else 'routing threshold τ'} = {ta['tau']} · top-3: " + ", ".join(f"{a} {b:.2%}" for a, b in ta["top3"]))
            if ta.get("source") != "second_classifier" and ta["route"] != "auto": st.warning("Confidence below the routing threshold - HR review recommended.")
        else:
            st.error(f"Classification not available: {rec.get('task_a_error') or rec['status']}. The bill was NOT filed in any category.")
        if rec.get("category"): st.success(f"Saved in category **{rec['category']}**")
        else: st.warning(f"Not classified - saved in `unclassified/` (status: {rec['status']}).")
        with st.expander("File locations"): st.code(f"folder:   {abs_path(str(Path(rec['saved_path']).parent))}\nfilename: {rec['saved_filename']}\nresults:  {abs_path(rec['result_dir'])}", language="text")
        # ---- quota + tax rebate for this bill ----
        tax_ui.render_bill_rebate(rec, ((rec.get("task_b_enhanced") or {}).get("fields")) or ((rec.get("task_b") or {}).get("fields")) or {}, key=sel)
        T[0].__exit__(None, None, None); T[1].__enter__()
        # ---- C. fields ----
        st.subheader("Extracted fields")
        if rec.get("task_b_error"):
            st.error(f"Field extraction failed: {rec['task_b_error']}  (OCR and classification above are unaffected)")
        elif rec.get("task_b") is not None:
            rows = []
            for f, label in KEY_FIELDS:
                v = flds.get(f)
                if v:
                    val = ", ".join(v["value"]) if isinstance(v["value"], list) else v["value"]
                    sus = "⚠ amount of 0 is almost certainly wrong" if (f in ("total_amount", "subtotal", "tax_total", "cgst", "sgst", "igst") and isinstance(v["value"], (int, float)) and v["value"] == 0) else ""
                    rows.append({"Field": label, "Value": str(val), "OCR text": v.get("raw_text"), "Confidence": fmt_conf(v.get("confidence")), "Source": v.get("source"), "Check": sus})
                else:
                    rows.append({"Field": label, "Value": "— not extracted", "OCR text": "", "Confidence": "—", "Source": ""})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            other = {f: v for f, v in flds.items() if f not in dict(KEY_FIELDS)}
            if other:
                with st.expander(f"Other fields returned ({len(other)})"):
                    st.dataframe(pd.DataFrame([{"Field": f, "Value": str(v["value"]), "Confidence": fmt_conf(v.get("confidence")), "Source": v.get("source")} for f, v in other.items()]), hide_index=True, use_container_width=True)
            st.caption("Model predictions, not verified ground truth. Missing fields are never guessed; check merchant name and invoice number against the original.")
        T[1].__exit__(None, None, None); T[2].__enter__()
        # ---- document profile (suggested classification, NOT eligibility) ----
        pf = rec.get("profile")
        st.subheader("Document profile (suggested)")
        if not pf: st.caption("Not available for this record.")
        else:
            c1, c2, c3 = st.columns(3); c1.metric("Document type", pf["document_type"]); c2.metric("Benefit category", pf["benefit_category"]); c3.metric("Tax-claim status", pf.get("claim_eligibility", "n/a").replace("_", " "))
            st.caption("Rule-based suggestion, not a trained classifier; it does not say the bill is tax-deductible.")
            with st.expander("Why"):
                for r_ in (pf.get("decision") or {}).get("reasons", []): st.caption("• " + r_)
            if pf["status"] == "needs_review": st.warning("Needs review: evidence is weak, tied or contradictory. Candidates: " + ", ".join(f"{c['claim_type']} ({c['evidence_level']})" for c in pf["candidates"]))
            if pf["validity"]["completeness"] == "incomplete": st.warning("Document looks incomplete - missing: " + ", ".join(pf["validity"]["missing_fields"]))
            if pf.get("ext_fields"): st.dataframe(pd.DataFrame([{"Field": k, "Value": v["value"], "Kind": "extracted from document text", "Source": v.get("source")} for k, v in pf["ext_fields"].items()]), hide_index=True, use_container_width=True)
        # ---- LLM-assisted extraction ----
        llm = rec.get("llm") or {}
        enh = (rec.get("task_b_enhanced") or {}).get("fields")
        if llm.get("status") not in (None, "off", "skipped"): st.subheader("LLM-assisted extraction (local model, validated against OCR)")     # hidden while the LLM is off / not needed
        if llm.get("status") in (None, "off", "skipped"): pass
        elif enh:
            st.caption(f"Model {llm.get('model')} ({llm.get('location')}), {llm.get('seconds')}s. Triggered by: {', '.join(llm.get('reasons') or ['mode=always'])}. "
                       "Evidence score is a heuristic ordinal (not a probability); fields marked ⚠ need a human check.")
            st.dataframe(pd.DataFrame([{"Field": f, "Value": ", ".join(map(str, v["value"])) if isinstance(v["value"], list) else str(v["value"]), "Origin": v["origin"], "Level": v["level"], "Score": v["evidence_score"],
                                        "Review": "⚠" if v["needs_review"] else "", "Evidence": " | ".join(v["evidence"])} for f, v in enh.items()]), hide_index=True, use_container_width=True)
            sg_ = llm.get("suggestions") or {}
            if sg_:
                st.markdown("**LLM suggestions - NOT applied.** The standard extractor found nothing for these fields. Values the LLM produced alone were right only about a quarter of the time on validation data, so they are never written into the fields above; check them against the document.")
                st.dataframe(pd.DataFrame([{"Field": f, "Suggested value": ", ".join(map(str, v["value"])) if isinstance(v["value"], list) else str(v["value"]), "Found in OCR text": "yes (checked)", "Evidence": " | ".join(v["evidence"])} for f, v in sg_.items()]), hide_index=True, use_container_width=True)
            if llm.get("rejected"):
                with st.expander(f"LLM proposals rejected as unsupported by the OCR text ({len(llm['rejected'])})"): st.dataframe(pd.DataFrame(llm["rejected"]), hide_index=True)
            if llm.get("corrections"):
                with st.expander(f"Proposed OCR corrections ({len(llm['corrections'])}; original OCR text is never modified)"): st.dataframe(pd.DataFrame(llm["corrections"]), hide_index=True)
        else: st.warning(f"LLM status: {llm.get('status')} {llm.get('error', '')}")
        # ---- tax suggestions ----
        sg = rec.get("tax_suggestions") or []
        if sg: st.subheader("Tax-section suggestions")
        for i, x in enumerate(sg):
            st.markdown(f"**{tax_ui.LABEL.get(x['claim_type'], x['claim_type'])}** - evidence: {x['evidence_level']} (matched: {', '.join(x['matched'][:5])})")
            for cv in x["caveats"]: st.caption("• " + cv)
            st.caption("Documents HR will need: " + "; ".join(x["required_documents"]) + ". Suggestion only - not an eligibility decision.")
            if st.button("Start a declaration item from this bill", key=f"sg_{sel}_{i}"):
                st.session_state["prefill"] = dict(doc_id=sel, claim_type=x["claim_type"], amount=x.get("amount"), caveats=x["caveats"]); st.session_state["pending_nav"] = "Tax Declaration"; st.rerun()
        # ---- B. OCR ----
        if rec.get("ocr_error"): st.error(f"OCR problem: {rec['ocr_error']}")
        ocr_file = rdir / "ocr.txt"
        txt = ocr_file.read_text(encoding="utf-8") if ocr_file.exists() else ""
        with st.expander(f"OCR text ({rec.get('ocr_words_page1', 0)} words on page 1)"):
            if txt.strip():
                st.text_area("Recognized text", txt, height=300, key=f"ocr_{sel}", label_visibility="collapsed")
                st.download_button("Download OCR text (.txt)", txt.encode("utf-8"), file_name=f"{Path(rec['original_filename']).stem}_ocr.txt", mime="text/plain", key="dl_ocr")
            else: st.info("No OCR text available for this document.")
        with st.expander("Raw record (JSON)"): st.json(rec)
        T[2].__exit__(None, None, None)

# ---------------- page 3: classified bills ----------------
def page_classified():
    st.title("Classified Bills")
    if not recs_all: st.info("No bills processed yet."); return
    df = pd.DataFrame([{"doc_id": r["doc_id"], "Category": r.get("category") or "(unclassified)", "File": r.get("saved_filename"), "Original name": r["original_filename"],
                        "Confidence": (r.get("task_a") or {}).get("confidence"), "Routing": (r.get("task_a") or {}).get("route"), "Status": r["status"], "Uploaded": r["uploaded_at"], "Path": r.get("saved_path")} for r in recs_all])
    st.subheader("Summary (from saved records)")
    c = df["Category"].value_counts()
    cols = st.columns(min(len(c) + 1, 6)); cols[0].metric("Total bills", len(df))
    for i, (k, v) in enumerate(c.items()):
        cols[(i + 1) % len(cols)].metric(k, int(v))
    st.bar_chart(c)
    cats = ["All"] + sorted(df["Category"].unique())
    pick = st.selectbox("Filter by category", cats)
    view = df if pick == "All" else df[df["Category"] == pick]
    st.dataframe(view.drop(columns=["doc_id"]), hide_index=True, use_container_width=True)
    st.caption(f"Files are stored under `{abs_path('classified')}/<category>/` (unclassified bills: `{abs_path('unclassified')}`).")
    present = sum(1 for p in view["Path"].dropna() if (S.OUT / p).exists()); st.caption(f"{present} of {len(view)} listed files exist on disk.")
    if len(view):
        who = st.selectbox("Open a bill", view["doc_id"].tolist(), format_func=lambda i: f"{i} · {df.set_index('doc_id').loc[i, 'Original name']}")
        if st.button("Show results for this bill"):
            st.session_state["current_id"] = who; st.session_state["pending_nav"] = "Results"; st.rerun()

if page == "Email Inbox":
    inbox_ui.page_inbox(role, actor, role_ok)
elif page in ("Tax Declaration", "HR Review", "Tax Summary"):
    {"Tax Declaration": tax_ui.page_declaration, "HR Review": tax_ui.page_hr, "Tax Summary": tax_ui.page_summary}[page](role, actor, role_ok, recs_all)
else:
    {"Upload Bill": page_upload, "Results": page_results, "Classified Bills": page_classified}[page]()
