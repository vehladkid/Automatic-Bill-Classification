# Decision summary (all numbers read from outputs/*.json)

| Question | Answer | Evidence |
|---|---|---|
| Is XLM-R classification superiority established (real test set, n=386)? | **No - inconclusive (category C)** | 5 pre-registered seeds: macro-F1 difference +0.052, 95% hierarchical interval -0.005 to +0.127. Ten seeds (secondary): +0.047, -0.007 to +0.118; all 10 seeds above TF-IDF; seed-only interval excludes 0 but ignores the finite test set |
| Practically meaningful (lower bound > 0.02)? | **Not established** | lower bound below zero |
| Is the cost justified? | **Not established** | measured ratios XLM-R/TF-IDF: size 25x, CPU end-to-end latency 45x, peak RSS 2.9x; GPU latency still 3.7x TF-IDF on CPU |
| Unseen synthetic challenge set (576 docs, own generator) | XLM-R ahead in clean (+0.109) and noisy (+0.197) conditions, intervals exclude 0; long-preamble condition confounded (both models predict non_expense) | measures wording/noise robustness only; not real-world accuracy |
| Unseen REAL set (Portuguese, silver labels, 2 classes) | both models poor; difference +0.111, -0.056 to +0.344 inconclusive | domain/language shift + labelling-convention mismatch; XLM-R accuracy 0.12-0.41 across seeds |
| Hybrid (TF-IDF first, XLM-R when TF-IDF confidence < validation-chosen threshold) | routes 14% of documents to XLM-R; macro-F1 0.879 (TF-IDF 0.831, XLM-R 0.883); hybrid - TF-IDF +0.047 [-0.000, +0.115] inconclusive | estimated CPU latency 11.1 ms vs 1.55 (TF-IDF) and 70.6 (XLM-R); memory/disk not reduced |
| Recommendation | keep TF-IDF + LR in production; evaluate XLM-R (or a TF-IDF-first hybrid) on a larger real, labelled, template-disjoint set | Section 4.5.12-4.5.14 of the report |

Production pipeline, dataset splits and the original experiments were not modified.
