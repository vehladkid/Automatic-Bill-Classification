# LLM-assisted extraction and HR tax declaration module

Added on top of the unchanged OCR -> Task A (3 models + fusion) -> Task B V1 pipeline. **No model weight, dataset split, test file or reported metric was modified.**
Everything below is an *estimate / assistant*: a human (HR) decides.

## 1. Pipeline
`upload -> OCR -> Task A category -> Task B V1 fields -> [LLM assist: propose -> validate against OCR -> fuse] -> tax-section suggestion -> employee declaration -> HR review -> tax summary`

| Step | Code | Notes |
|---|---|---|
| LLM client | `app/llm/client.py`, `config.py` | Ollama-compatible HTTP API, temperature 0, JSON output. **Loopback only** unless `HRBC_LLM_ALLOW_REMOTE=1` *and* `HRBC_LLM_REMOTE_ACK=<exact string>`; otherwise status `privacy_blocked` and nothing is sent. Default mode `off`. |
| When it runs | `app/llm/enhance.py::needs_llm` | `auto`: only if merchant/date/total missing or low-confidence, total = 0, GSTIN checksum invalid, mean OCR confidence < 0.85, or amounts fail arithmetic. `always` / `off` also available. |
| Input to the LLM | OCR lines + boxes (0-1000 units) + page image (long side 1280, optional) | V1 predictions are *not* shown to the LLM, so agreement is independent evidence. |
| Validation | `app/llm/validate.py` | amounts must occur in OCR numbers; dates must be valid, plausible and printed; GSTIN must pass format and checksum and be in OCR or a checksum-valid confusable-character repair (O/0, I/1, S/5, B/8...); text must match an OCR line (>=0.85); IDs must be in OCR or one confusable-char away; HSN/SAC digits must occur in OCR. Failures are *rejected and logged*, never used. |
| OCR corrections | `validate_correction` | Original string must exist in OCR; change must be confusable-characters only; result must be structurally valid (GSTIN checksum / valid date / number). The original OCR text is never overwritten; corrections are stored with their verdict. |
| Fusion | `app/llm/fuse.py` | Agreement of V1 and a grounded LLM value raises the score; conflicts keep V1 unless V1 fails a hard check and the LLM value passes; LLM-only values are added but always flagged for human review; arithmetic checks (total = subtotal + tax, tax = CGST+SGST+IGST) add or subtract evidence. `evidence_score` is an **ordinal heuristic, not a probability**. |
| Suggestion | `app/tax/suggest.py`, `config/tax_suggest.json` | Keyword evidence -> candidate claim type + caveats + required documents. Fuel/food/hotel bills produce nothing; a hospital bill is not an 80D premium; school bills flag non-tuition items. Never says "eligible". |
| Declaration + HR | `app/tax/store.py`, `app/tax_ui.py` | SQLite (`outputs/hr/hr.db`), append-only audit log. Roles enforced in the data layer: employees touch only their own claims; only `hr` decides; nobody decides their own claim; reject / request-documents need a reason; approval without a linked document needs a written reason; editing a decided claim resets it to draft. |
| Tax engine | `app/tax/engine.py` + `config/tax_rules/fy*.json` | Old and new regime, slabs, 87A rebate with marginal relief, surcharge with marginal relief, 4% cess, per-claim explanations, duplicate and double-count prevention, `approved` vs `projected` counting. |

## 2. Tax rules are data, and their verification status is shown in the UI
`config/tax_rules/fy2026-27.json` and `fy2025-26.json` hold every rate, cap, window and metro list. A missing financial year raises an error instead of guessing.
**Verification status: `SECONDARY_SOURCES_ONLY`** (checked 2026-10-19 against published summaries of Budget 2025/2026, the Income-tax Act 2025 and the draft/final Income-tax Rules 2026; sources are listed in the file). They were **not** checked against the official gazette text. A payroll/tax professional must confirm each value (especially: the Rs 200 meal rate, the 8-city HRA metro list, 80CCD(2) 14%/10%, and the Income-tax Act 2025 section renumbering) and then change `verification.status`. Every tax screen shows this banner.

Modelled: HRA (old regime; least of HRA received / rent - 10% basic+DA / 50% or 40% of basic+DA; landlord PAN above Rs 1,00,000 rent), meal vouchers (per-meal cap, both regimes, limited to employer-provided value), 80C (Rs 1.5L cap shared by payroll EPF, investments, tuition, home-loan principal), tuition fees (tuition component only, 2 children, India, full-time), 80D (self/family and parents, senior limits, preventive check-up Rs 5,000, no cash premium), 80E (interest only, 8 years), 24(b) self-occupied (Rs 2L), 80CCD(1B), 80CCD(2), 80EEB (loans sanctioned 1-Apr-2019 to 31-Mar-2023 only), professional tax, standard deduction.
**Deliberately not auto-decided:** EV lease/reimbursement perquisites, let-out property, any "other" item - they stay at 0 eligible, status `pending_review`, until HR records a reason-coded override. Not modelled: capital gains, perquisite valuation, LTA, 80G/80GG, arrears, multiple employers, NRI rules.
Output is an **estimate**, not an official payroll/TDS computation.

## 3. How to run
```bash
cd /home/tejvir-singh/Desktop/ML-project
.venv/bin/python -m unittest discover -s tests -v     # 97 tests, no GPU or LLM needed
bash app/run_app.sh                                    # http://127.0.0.1:8501  (pages: Upload Bill, Results, Classified Bills, Tax Declaration, HR Review, Tax Summary)
```
Optional local LLM (nothing is downloaded by the app itself):
```bash
# 1. install Ollama (https://ollama.com), then   ollama pull qwen2.5vl:3b      (~3 GB; check VRAM: the OCR+Task A+Task B models already use ~2 GB of the 6 GB GPU)
# 2. HRBC_LLM_MODE=auto bash app/run_app.sh
# 3. measured evaluation (writes only reports/taskB/llm/):
.venv/bin/python scripts/taskB/llm_eval.py --split val --limit 40     # smoke on validation first
.venv/bin/python scripts/taskB/llm_eval.py --split test               # real test documents, results cached per document
```
Environment variables: `config/env.example`.

## 4. Measured results
**LLM accuracy before/after: MEASURED on the validation split - see section G (no accuracy gain; LLM off by default). Test split NOT yet run.**

**Automated tests (97, all passing):** `tests/test_tax.py` (slabs, rebate/surcharge relief, HRA, every deduction cap and rejection, duplicates, missing documents, regime restrictions), `tests/test_llm.py` (invented values rejected, OCR-correction validation, privacy guard, gating, fusion/review flags, unavailable server), `tests/test_workflow.py` (permissions, audit trail, HR approve/reject/needs-documents loop, suggestions). Tests check the implemented rules; they do not prove the rules match the law.

## 5. Known limits
- No authentication (role selector only); production needs SSO.
- Tax rules unverified against official text (above). Salary components are self-declared by the employee.
- Suggestions are keyword evidence, not a classifier; they miss unusual phrasing and are trained on nothing.
- Task A has no school/loan/insurance/EV categories; such bills file under `miscellaneous` and the tax suggester works from text.
- LLM latency on a 6 GB laptop GPU is expected to be seconds per bill; not measured yet.

---
# Addendum: employee-document taxonomy, profile, autofill (second iteration)

## A. Audit result (what already existed)
Task A (16 expense categories; 3 models + fusion, calibrated) and Task B V1 (34-field extractor) are trained and unchanged. **A document-type classifier was never trained** (Task C was removed from scope; `document_type` exists only as a dataset label). There was **no email code, no HTTP backend, and zero real rent / school / 80C / 80D / loan / EV / Sodexo documents** in the dataset.

## B. Taxonomy (`config/taxonomy.json`, `app/doc_profile.py`, `app/doc_fields.py`)
Seven separate parts per document: `document_type`, `expense_category` (Task A, passed through untouched), `benefit_category` (ev_benefit, meal_benefit, hra_rent, education_fee, investment_80c, health_insurance, loan, other_benefit, none), `tax_section_candidates` (classification, **not** eligibility), `validity` (completeness, missing fields, financial year), `fields`/`field_report`, `evidence` + confidences. Multi-label: a primary claim type plus secondary labels; ties or weak evidence give `needs_review`; no signal gives `expense_bill` / `unknown`.
**The new labels are RULES, not trained classes** (`model_support.document_type_and_benefit = "rules_only - NOT a trained classifier"`, `calibrated: false`). I did not retrain any model: there is no labelled data for them. `data_needed` per type is listed in the taxonomy file. Backward compatibility: Task A predictions and all existing record fields are unchanged; `profile`, `field_report`, `owner`, `source` are additive.

## C. Measured: false-positive rate of the new rule labels on real ordinary bills
`scripts/eval_profile_rules.py` -> `reports/profile_rules_specificity.json`. 3,285 real labelled expense documents (none is a tax-document type, so any tax label is a false positive):
| | n | false tax-document label | needs_review |
|---|---|---|---|
| before whole-word matching fix | 3,285 | 277 (8.4%) | 27 |
| after fix (all) | 3,285 | **13 (0.4%)** | 16 (0.5%) |
| after fix (test split only) | 386 | 3 (0.8%) | 3 (0.8%) |
The first run exposed substring matches ("ppf", "nsc", "lic" inside ordinary words); I fixed the matcher and removed the ambiguous "lic" term, so the post-fix numbers are **not an untouched test** (the bills were used to find the bug). Remaining false positives: "ppf" on some Malaysian fuel receipts and Form 16 / salary-slip documents labelled `non_expense` that legitimately mention rent or insurance. **Recall for the new categories is NOT measured and cannot be** - no labelled examples exist. Phrase overlap (e.g. "tuition fee" + "fee receipt") double-counts evidence.

## D. Field-level confidence (`app/field_report.py`)
Four separate things per field: OCR confidence (of segments under the box), extraction (source; V1 isotonic-calibrated confidence only where it exists; LLM evidence level flagged uncalibrated), deterministic validation (amount>0, date plausibility, GSTIN format/checksum, tax/total arithmetic), and presence (`extracted`, `extracted_unverified`, `conflict`, `illegible`, `absent`). LLM self-reported confidence is never used. The LLM stage now also proposes rent/school/insurance/loan fields, which pass through the same OCR-grounding rule.

## E. Autofill (`app/tax/autofill.py`)
Each value is labelled `extracted | employee_declared | payroll_verified | calculated | assumption` with its source document/evidence. Financial year comes only from the document date; no date, no FY, no amount, or an FY without configured rules => `needs_review` and claim creation is refused (nothing is fabricated). Conflicts (tenant name vs profile, profile without HRA, differing amount/PAN for the same rent month), duplicates (same document hash or same bill identity) and missing items (landlord PAN, tuition breakup, regime) are listed. School fees are never pre-filled as deductible. Profile values become `payroll_verified` only after an HR user other than the owner verifies the profile; editing the profile removes verification. Claim edits are stored with old/new values, actor, role and time.

## F. NOT done / not measured
- LLM before/after benchmark: **run on the validation split only** (section G). The frozen TEST split has not been run through the LLM.
- Classification macro-F1, per-class P/R/F1 and confusion matrix for the new categories: **not possible** without labelled data.
- No calibration for LLM-derived or extension fields.
- Tax rules still `SECONDARY_SOURCES_ONLY`.

---
# G. Local LLM benchmark (third iteration) - measured results

**Setup.** Ollama 0.40.6 (user-space, loopback only) + `qwen2.5vl:3b` (3.8B params, Q4_K_M GGUF, 3.2 GB; vision-capable, verified by running both text-only and image requests). RTX 4050 6 GB.
Data: the **295 real validation documents** of the Task B split (970 labelled field instances over 10 fields; synthetic excluded; no test data used, no tuning on test). Same OCR words and the same scorer (`scripts/taskB/evaluate.py`) for every system. Raw model outputs cached in `reports/taskB/llm/`; scoring is offline and repeatable. Scripts: `llm_bench.py`, `llm_policy_eval.py`.

## Field extraction, text-only LLM (OCR lines + boxes; 294/295 valid answers) - validation, 295 docs
| System | Precision | Recall | Micro-F1 | TP / FP / FN |
|---|---|---|---|---|
| Existing pipeline (V1) | 0.757 | 0.657 | **0.703** | 637 / 205 / 333 |
| LLM output as-is (no validation) | 0.529 | 0.555 | 0.542 | 538 / 479 / 432 |
| LLM output, OCR-grounded only | 0.655 | 0.481 | 0.555 | 467 / 246 / 503 |
| V1 + LLM fused, LLM-only values written as fields (old behaviour) | 0.702 | 0.684 | 0.692 | 663 / 282 / 307 |
| **V1 + LLM fused, LLM-only values as suggestions (new default)** | 0.758 | 0.658 | 0.704 | 638 / 204 / 332 |
Paired bootstrap (docs resampled, 1000x) of F1 minus V1: raw -0.162 [-0.198, -0.124]; grounded -0.149; fused(add) -0.011 [-0.022, +0.002]; **default policy +0.001 [-0.006, +0.009]** - i.e. **no measurable accuracy gain**. Real-Indian slice (small): V1 0.628, default policy 0.659 (n tiny; not significant).
Validation shows what is and is not useful: LLM-only values were correct 24% of the time (25/103) and are therefore never written into fields; 788 of 1,897 LLM proposals (42%) were rejected as not found in the OCR text; where V1 and the LLM **agree** 87% of values are correct (423/488) vs 72% for V1-only (167/232) and ~40% in conflicts. The measurable benefit is a **review-triage signal**, not accuracy: fused evidence level high/medium/low -> precision 0.855 / 0.759 / 0.361. Hallucination (a value filled where the ground truth has none): V1 15.7%, raw LLM 36.7%, default policy 15.7% (unchanged).
Per field (F1 V1 -> fused-add): total_amount 0.838 -> 0.853, tax_total 0.869 -> 0.874, invoice_number 0.505 -> 0.522, merchant_name 0.553 -> 0.537, subtotal 0.791 -> 0.730, currency 0.538 -> 0.506 (small supports for cgst/sgst/gstin: n = 8-11).

## Latency / resources (text mode, measured during the run)
mean 4.6 s, median 3.5 s, p90 10.9 s per document; 1,066 prompt / 344 output tokens on average; model resident VRAM ~4.3 GB (idle 15 MB); estimated energy 0.023 kWh (~INR 0.2) for 295 docs **assuming 60 W** (assumption, not measured); API cost 0. The 5.77 GB "peak" in the raw JSON is contaminated by my own concurrent OCR smoke test - ignore it.
**The model and the OCR/Task A/B stack do not fit on the 6 GB GPU together** (an OCR call failed with the model resident). Default `HRBC_LLM_KEEP_ALIVE=0` unloads the model after each request; the cost is a model reload (about 4-6 s) per document. Not measured end to end.

## Vision mode - NOT successfully benchmarked
225 of 295 vision calls returned an empty answer. Cause found: the Ollama server aborts generation with `prediction aborted, token repeat limit reached` (seen directly) - the model loops on repeated receipt lines (e.g. `Rp.0 x1 Rp.0`); with `format=json` this surfaces as an empty body. It is deterministic for some CORD receipts (reproduced after a full model reset), not transient (an earlier note here guessed otherwise - wrong). A text-only retry fallback was added (`run_llm`), but a re-run still aborted after 80 documents and I stopped investigating. Only documents where vision succeeded can be compared: on **74 documents where both modes answered** (68 SROIE, 6 CORD - the easy receipts, V1 F1 0.910): V1 0.910; text fused (default) 0.906; vision fused (default) 0.906; vision raw 0.812. **No accuracy benefit from vision over text, +2.4 s latency** (7.05 s vs 4.64 s). Treat any vision conclusion as inconclusive.

## Classification decision policy (explicit, versioned: `doc_profile.decide`, policy `profile-v2-policy`)
The LLM may confirm or break a tie, never override: (1) its opinion counts only if the OCR words it quotes as evidence really are in the document; (2) rule evidence + LLM disagreement => `needs_review`; (3) an LLM-only tax label => `needs_review` (never a label); (4) weak/tied rule evidence => `needs_review` unless a grounded LLM opinion matches exactly one tied candidate; (5) the trained Task A expense category is never changed; low Task A confidence only adds a review reason; (6) every tax-type document is `claim_eligibility: pending_verification` and `auto_claim_allowed: false`.
Measured on the same 295 real ordinary validation bills (all non-tax, so any tax label is a false positive; `llm_policy_eval.py`, text mode): rules alone 0 tax labels; **LLM alone proposed a tax kind on 99 docs (33.6%), only 3 with evidence found in the text**; combined policy 0 tax labels, 5 sent to review (1.7%), 0 contradictions. Takeaway: the LLM's raw document-kind opinion is unreliable; the grounding requirement is what makes it harmless. **Recall on genuine rent/school/80C/80D/loan/EV documents remains unmeasured (no labelled examples).** Task A accuracy and macro-F1 are unchanged by design (the LLM never touches the category).

## Other additions this iteration
`app/llm/service.py` (content-addressed cache holding model JSON only, single inference slot, retry only on transport failure, structural schema validation, logging of timings/token counts only); zero-vs-missing rule (an amount of 0 is accepted only for tax/discount/round-off style fields and only next to a label naming that field; `total_amount = 0` is never accepted); more extension fields (payment/student reference, fee period, vehicle details, eligible fee amount); output cap of 1,024 tokens + recovery of truncated JSON (found on validation: repetition loops in the HSN list); claim results now carry four separate amounts: claimed, on-document, eligible, final-approved (None until HR approves), and a claim above its document amount is limited to the document amount; Results page shows the document profile, decision reasons and LLM suggestions separately from extracted fields.

## Verdict and recommendation
**The LLM does not meaningfully improve extraction accuracy on validation data, and costs ~4.6 s/document plus a GPU that the rest of the pipeline also needs. It stays OFF by default** (`HRBC_LLM_MODE=off`). If enabled, use text-only (`HRBC_LLM_USE_IMAGE=0`), `auto` mode, and treat the output as review triage and suggestions. Not yet done: the single frozen-config run on the TEST split (`llm_bench.py --split test --modes text`, ~25 min), vision-mode repetition-abort fix, end-to-end latency with model unload, measuring recall for the new categories (needs labelled real documents). Tests: 209 unit tests + 26 app self-checks pass; frozen test file hash and V1 snapshot unchanged.
