# Draft permission requests (NOT SENT)

Nothing here has been sent, submitted or accepted; a human must review, choose the channel and send. Replace the bracketed fields. No e-mail addresses are listed because none were verified on the primary pages; use the channel named for each source.

## Common wording (adapt per source)
> Hello [owner / maintainers],
> I am building an internal, non-commercial research system that classifies Indian expense and tax-evidence documents (rent receipts, salary slips, bank statements, etc.). I would like to use **[dataset name and URL]** and would be grateful for written answers to the following:
> 1. **Rights and provenance.** Who owns the copyright in the images, and under what permission were they collected? [For scraped/web images: were they taken from sources that allow re-use?]
> 2. **Training and evaluation.** May we use the images to train and evaluate machine-learning classifiers, and to report aggregate metrics?
> 3. **Commercial use.** Is any commercial use permitted (for example inside a product that customers pay for)? If not, is a separate licence available?
> 4. **Processing.** May we run OCR on the images, store derived text/features and keep a private copy of the files?
> 5. **Redistribution.** We do not plan to redistribute the images. Would redistribution of derived annotations only (labels and identifiers, no images) be acceptable?
> 6. **Personal data.** Do any images contain real individuals' personal data? If so, how was consent handled?
> 7. **Attribution.** How should we cite the dataset?
> Your reply (e-mail or a note on the dataset page) will be stored with our licence register as the permission record. We will keep the data out of our approved training set until we have it.
> Thank you, [name, affiliation, contact]

## Per source
| source | channel to use | what to add to the common request | what a sufficient answer looks like |
|---|---|---|---|
| `in_financial` - Kaggle `mehaksingal/personal-financial-dataset-for-india` | the dataset's Kaggle discussion/contact option for owner "Mia" (no e-mail shown) | ask specifically whether she owns the images or merely collected them from Google Images; whether items are real filled documents or specimens; whether any show real people | a written statement of ownership/permission for training and evaluation, or a list of images she can license. Otherwise it stays blocked |
| SROIE - ICDAR 2019 RRC | the RRC portal (register, read the terms shown at download, use its contact form); the organisers are the data owners | ask for the data terms for non-commercial and commercial ML use and whether the Hugging Face mirror's CC-BY-4.0 label is authoritative | the RRC terms text or a written confirmation. Until then excluded |
| `hf_medical` - HF `destryet/medical_bills` (mirror `sansverse/medical-bill-samples`) | Community tab on each dataset page | ask who produced the bills, whether issuer/patient permission exists, whether redaction was applied | a licence and a provenance statement. Medical bills are sensitive: without both, keep blocked |
| ReceiptBench - HF `T0rl/ReceiptBench` | Community tab | the README is empty: ask where the images come from and whether people consented | provenance + consent statement, or per-image licences |
| `user_real` | internal: project owner | no external request; the owner records an attestation | owner states they collected the 198 receipts, may use them for training/evaluation, and that no card or personal data is visible |
| `manual_in` | internal: project owner | no external request | owner states the 368 Indian bills were collected by them and that no third-party personal data is retained |
| IDRBT Cheque Image Dataset | IDRBT (Institute for Development and Research in Banking Technology) via its website contact; authors Dansena, Bag, Pal (2017) | 112 cheques written by volunteers; ask for a licence statement permitting ML training/evaluation and derived annotations; ask whether volunteers consented | a written licence. The annotations-only release (CC BY-SA 4.0, arXiv 2606.20682) needs the images, so it does not help alone |
| Roboflow `drumil-shah/form-16` | Roboflow account message to the uploader (needs a signed-in human) | 45 images with PAN/name fields: ask for provenance and whether they are specimens | provenance + confirmation that no real personal data is present. Download needs an account: not attempted |

## Owner attestation template (for `manual_in`, `user_real`, `owner_own_document`)
"I, [name], confirm that I collected or hold the documents listed in [batch id], that I am entitled to use them for training and evaluating an internal document-classification model, that third-party personal data has been removed or never present, and that I understand the originals will be kept in the offline vault and only redacted copies used. Signed [date]." Store the signed copy in the vault and give it a `CR-` id used as `permission_ref`.
