"""Streamlit page 'Email Inbox': ingested/processed documents with status, source email, document profile, per-field evidence, correction + retry controls, autofill proposal.
All data access goes through ingest.api (tenant isolation enforced there)."""
import json
import pandas as pd
import streamlit as st
import service as S
from ingest import api, db
from tax import autofill as AF, store as ST, engine as E
from tax.store import PermissionDenied

STATE_ICON = {"COMPLETED": "✅", "done": "✅", "REVIEW_REQUIRED": "🟡", "FAILED": "❌", "QUEUED": "⏳", "PROCESSING": "⚙️", "DUPLICATE": "♻️", "IGNORED": "·", "REJECTED": "⛔", "RECEIVED": "📨", "VALIDATING": "🔎"}
PRES = {"extracted": "✓ extracted", "extracted_unverified": "⚠ unverified", "conflict": "⚠ conflict", "illegible": "⚠ illegible", "absent": "— not found"}

def _fmt(x):
    """never show an unverified value as fact"""
    return "—" if x["value"] is None else (str(x["value"]) if x["verified"] else f"{x['value']} (unverified)")

def page_inbox(role, actor, ok):
    st.title("Email Inbox & Documents")
    if not ok: st.info("Enter your ID in the sidebar."); return
    p = api.Principal(role, actor)
    for n in api.notifications(p)[:3]: st.info(f"{n['ts'][:16]} · {n['text']}")
    if role == "hr":
        try: fl = api.list_failures(p)
        except PermissionDenied: fl = []
        if fl:
            st.warning(f"{len(fl)} incoming message(s)/fetch job(s) are retrying or dead-lettered (nothing was lost; see codes below).")
            st.dataframe(pd.DataFrame(fl), use_container_width=True)
            if st.button("Return dead-lettered items to retry"): st.success(f"{api.requeue_failures(p)} item(s) requeued")
    if role == "hr":
        try: ma = api.list_mail_accounts(p)
        except PermissionDenied: ma = []
        with st.expander(f"Connected mailboxes ({len(ma)}) - read-only"):
            if ma: st.dataframe(pd.DataFrame([{"Id": m["public_id"], "Mailbox": m["address"] or ("(configured in worker.env)" if m["is_legacy"] else ""), "Provider": m["provider"], "For employee": m["owner_employee_id"] or "", "Status": m["status"],
                                               "Last successful sync": m["last_sync_at"] or "never", "Last error code": m["last_error_code"] or "", "Failures in a row": m["consecutive_failures"]} for m in ma]), hide_index=True, use_container_width=True)
            else: st.caption("No mailbox registered yet (the worker registers the environment-configured mailbox on its first start).")
            st.caption("Connecting, re-authorising or disconnecting a mailbox is done by the operator on the server (`python -m ingest.run mail-add / mail-disconnect`). This page cannot do it because the app has no real user authentication yet. A status of needs_reauth means the Google authorisation was revoked or expired: run mail-add again for that employee and mailbox.")
    tab_docs, tab_mail = st.tabs(["Documents", "Email history"])
    with tab_docs:
        f = st.columns(5); docs0 = api.list_documents(p)
        sel = lambda col, label, vals: col.selectbox(label, ["(all)"] + sorted({v for v in vals if v}))
        status = sel(f[0], "Status", [d["status"] for d in docs0]); cat = sel(f[1], "Category", [d["category"] for d in docs0]); dt = sel(f[2], "Document type", [d["document_type"] for d in docs0])
        fy = sel(f[3], "Financial year", [d["financial_year"] for d in docs0]); emp = sel(f[4], "Employee", [d["owner"] for d in docs0]) if role == "hr" else "(all)"
        docs = api.list_documents(p, None if status == "(all)" else status, None if cat == "(all)" else cat, None if fy == "(all)" else fy, None if emp == "(all)" else emp, None if dt == "(all)" else dt)
        if not docs: st.info("No documents match."); return
        st.dataframe(pd.DataFrame([{"": STATE_ICON.get(d["status"], ""), "Status": d["status"], "File": d["filename"], "Subject": (d["email"] or {}).get("subject") or "", "Channel": d["channel"], "From": (d["email"] or {}).get("sender") or "", "Category": d["category"], "Category conf.": (f"{d['category_confidence']:.0%}" + ("" if d["category_route"] == "auto" else " ⚠ low") if d["category_confidence"] is not None else "—"), "Merchant": _fmt(d["merchant"]), "Amount": _fmt(d["amount"]), "Date": _fmt(d["date"]), "Review": ("🟡 needs review: " + "; ".join(d["review_reasons"][:3])) if d["needs_review"] else "✓ automated", "Document type": d["document_type"],
                                    "Benefit": d["benefit_category"], "FY": d["financial_year"], "Received": (d["email"] or {}).get("received_at") or d["uploaded_at"], "Employee": d["owner"] or "(unowned)"} for d in docs]), hide_index=True, use_container_width=True)
        doc_id = st.selectbox("Open a document", [d["doc_id"] for d in docs], format_func=lambda i: next(f"{d['filename']} · {d['status']} · {i}" for d in docs if d["doc_id"] == i))
        try: D = api.get_document(p, doc_id)
        except PermissionDenied as e: st.error(str(e)); return
        rec, prof, fr = D["record"], D["profile"], D["field_report"] or {}
        a, b = st.columns([4, 6])
        with a:
            rdir = S.OUT / rec["result_dir"]
            for pg in sorted([*rdir.glob("page_*.png"), *rdir.glob("page_*.jpg")])[:3]: st.image(str(pg), use_container_width=True)
            if D["email"]: st.caption(f"Source email from {D['email']['sender']} · “{D['email']['subject']}” · {D['email']['received_at']}" + (f" · mailbox #{D['email']['account_id']}" if D['email'].get('account_id') else ""))
            else: st.caption(f"Source: {(rec.get('source') or {}).get('channel', 'web upload')}")
            for w in rec.get("warnings", []): st.warning(w)
        with b:
            if D["email"] and D["email"]["state"] in ("REVIEW_REQUIRED", "FAILED") and D["email"].get("error"): st.warning("Why review is needed: " + D["email"]["error"])
            st.subheader("Classification")
            if prof:
                c = st.columns(3); c[0].metric("Expense category (Task A)", prof["expense_category"] or "—"); c[1].metric("Document type", prof["document_type"]); c[2].metric("Benefit category", prof["benefit_category"])
                ta = rec.get("task_a") or {}
                st.caption(f"Expense category confidence {ta.get('confidence', 0):.0%} (calibrated, threshold-routed: {ta.get('route')}). Document/benefit type: **{prof['classification_confidence']['level']} rule evidence** - "
                           f"not a probability. {prof['model_support']['document_type_and_benefit']}.")
                if prof["status"] == "needs_review": st.warning("Ambiguous or weak evidence - labelled needs_review instead of guessing. Candidates: " + ", ".join(f"{c['claim_type']} ({c['evidence_level']})" for c in prof["candidates"]))
                for r_ in (prof.get("decision") or {}).get("reasons", []): st.caption("• " + r_)
                if prof["secondary_claim_types"]: st.caption("Also possibly: " + ", ".join(prof["secondary_claim_types"]))
                if prof["tax_section_candidates"]: st.caption("Could be evidence for: " + ", ".join(prof["tax_section_candidates"]) + ". This is classification, **not** a statement that the amount is deductible.")
                v = prof["validity"]
                if v["completeness"] == "incomplete": st.warning("Document incomplete - missing: " + ", ".join(v["missing_fields"]))
            st.subheader("Fields and evidence")
            rows = []
            for fld, x in {**fr, **{k: v for k, v in D["fields"].items() if v.get("source") == "human_correction"}}.items():
                eff = D["fields"].get(fld, {}); human = eff.get("source") == "human_correction"
                rows.append({"Field": fld, "Value": str(eff.get("value") if human else x.get("value")) if (human or x.get("value") is not None) else "—", "Presence": "✎ human-corrected" if human else PRES.get(x["presence"], x["presence"]),
                             "OCR conf.": x.get("ocr_confidence") if not human else None, "Extractor": (x.get("extraction") or {}).get("source") if not human else f"corrected by {eff.get('corrected_by')}",
                             "Calibrated conf.": (x.get("extraction") or {}).get("calibrated_confidence") if not human else None, "Evidence level (uncalibrated)": (x.get("extraction") or {}).get("evidence_level") if not human else None,
                             "Validation": (x["validation"]["status"] + (": " + ",".join(x["validation"]["failed"]) if x["validation"]["failed"] else "")) if not human else "human"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("OCR confidence, extraction confidence, classification confidence and validation are separate. 'Calibrated conf.' exists only for the V1 extractors (validated on held-out data); LLM evidence levels are ordinal heuristics.")
            with st.expander("Correct a value (recorded with your name, time and reason)"):
                import doc_fields as DF
                fld = st.selectbox("Field", sorted(set(list(fr) + ["merchant_name", "invoice_number", "invoice_date", "total_amount", "subtotal", "tax_total", "gstin"])), key=f"cf_{doc_id}")
                nv = st.text_input("Correct value", key=f"cv_{doc_id}"); why = st.text_input("Reason", key=f"cr_{doc_id}")
                if st.button("Save correction", key=f"cb_{doc_id}"):
                    try: api.correct_field(p, doc_id, fld, nv, why); st.success("Saved."); st.rerun()
                    except (ValueError, PermissionDenied) as e: st.error(str(e))
            if D["corrections"]: st.dataframe(pd.DataFrame(D["corrections"])[["ts", "actor", "role", "field", "original", "corrected", "reason"]], hide_index=True)
        # ---- per-bill quota + income-tax effect (estimate)
        import tax_ui; tax_ui.render_bill_rebate(rec, {k: v for k, v in (D["fields"] or {}).items()}, key=doc_id)
        # ---- tax autofill proposal
        st.subheader("Tax declaration autofill")
        owner = rec.get("owner")
        if not owner: st.caption("Legacy upload without an owner - assign it through the Tax Declaration page manually.")
        elif not (prof and prof.get("primary_claim_type")): st.caption("No tax-document type detected with enough evidence; nothing to autofill.")
        else:
            existing = ST.list_claims(owner, None)
            af = AF.autofill(rec, D["fields"], owner, existing_claims=existing, doc_hashes=[rec.get("sha256")])
            st.write(f"Proposed item: **{af['claim_type']}** · FY {af['fy'] or 'unknown'} · regime {af['regime'] or 'not chosen'} · status **{af['status']}**")
            if af["values"]: st.dataframe(pd.DataFrame([{"Field": v["field"], "Value": str(v["value"]), "Kind": v["kind"], "Source": json.dumps(v["source"], default=str)[:160]} for v in af["values"]]), hide_index=True, use_container_width=True)
            for c in af["conflicts"]: st.error("Conflict: " + c["message"])
            for d in af["duplicates"]: st.warning(f"Possible duplicate of claim {d['claim_id']}: {d['why']}")
            for m in af["missing"]: st.caption("Missing: " + m)
            for n in af["notes"]: st.caption(n)
            if role == "employee" and owner == actor and af["status"] != "denied":
                amt = st.number_input("Amount to claim (₹) - edit if needed", 0.0, value=float(af["amount"] or 0), key=f"am_{doc_id}")
                if st.button("Add as draft claim", key=f"ac_{doc_id}", disabled=bool(af["duplicates"])):
                    try: cl = AF.create_claim(af, owner, actor, edits={"amount": amt} if amt != af["amount"] else None, doc_hashes=[rec.get("sha256")]); st.success(f"Draft claim {cl['id']} created; submit it from Tax Declaration.")
                    except (ST.StoreError, PermissionDenied) as e: st.error(str(e))
            elif role == "hr": st.caption("Employees add their own claims; HR reviews them in HR Review.")
    with tab_mail:
        mails = api.list_emails(p)
        if not mails: st.info("No emails ingested yet."); return
        st.dataframe(pd.DataFrame([{"": STATE_ICON.get(m["status"], ""), "ID": m["id"], "Status": m["status"], "Reason": m["reason"], "From": m["sender"], "Subject": m["subject"], "Received": m["received_at"], "Attachments": m["n_attachments"], "Employee": m["employee_id"]} for m in mails]), hide_index=True, use_container_width=True)
        eid = st.selectbox("Email", [m["id"] for m in mails])
        try: det = api.email_detail(p, eid)
        except PermissionDenied as e: st.error(str(e)); return
        st.dataframe(pd.DataFrame([{"#": a["idx"], "": STATE_ICON.get(a["state"], ""), "File": a["filename"], "State": a["state"], "Pages": a["pages"], "Code": a["error_code"], "What to do": a["error"], "Attachment ID": a["id"]} for a in det["attachments"]]), hide_index=True, use_container_width=True)
        retry = [a["id"] for a in det["attachments"] if a["state"] == "FAILED" and a["error_code"] in ("processing_error", "retries_exhausted")]
        if retry:
            r = st.selectbox("Retry a failed attachment", retry)
            if st.button("Retry"):
                try: api.retry_attachment(p, r); st.success("Queued again."); st.rerun()
                except (ValueError, PermissionDenied) as e: st.error(str(e))
        with st.expander("Processing history"): st.dataframe(pd.DataFrame(det["history"]), hide_index=True)
