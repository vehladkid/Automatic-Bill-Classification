"""Phase 8 - why did BiLSTM, BiLSTM+attention, LayoutLMv3 and LexRank score lower than TF-IDF? Separates (1) observed underperformance, (2) MEASURED implementation/data facts, (3) plausible-but-UNVERIFIED explanations,
(4) the experiments that would be needed to establish causation. Reads saved artefacts; tokenisation-only measurements; no training.  Run low priority."""
import sys, json, re, collections
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "taskA"))
from xlmr_lib import *
from common_a import words as row_words
T = ROOT / "experiments/text_vs_neural/outputs"; tr, va, te = splits(); real_te = [r for r in te if not r["synthetic"]]; res = {}
tf = json.loads((T / "m1_tfidf.json").read_text())["calibrated"]["real"]
def per_class_gap(j):
    pc = j["calibrated"]["real"]["per_class"]; return {c: dict(support=v["support"], f1=v["f1"], tfidf_f1=tf["per_class"][c]["f1"], delta=v["f1"] - tf["per_class"][c]["f1"]) for c, v in pc.items()}
# ---------------- BiLSTM / attention ----------------
TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]"); tokz = lambda t: [re.sub(r"\d", "0", w) for w in TOK.findall(t.lower())]
freq = collections.Counter(w for r in tr for w in tokz(r["text"])[:256]); vocab = {w for w, c in freq.most_common(40000) if c >= 2}
tok_test = [tokz(r["text"]) for r in real_te]; all_tok = [w for t in tok_test for w in t[:256]]; oov = float(np.mean([w not in vocab for w in all_tok])); alpha_tok = [w for w in all_tok if re.fullmatch(r"[a-z]+", w)]; oov_alpha = float(np.mean([w not in vocab for w in alpha_tok]))
long_alpha = float(np.mean([len(w) > 12 for w in alpha_tok]))
for key, fn, name in (("bilstm", "m_bilstm.json", "BiLSTM"), ("bilstm_attn", "m_bilstm_attn.json", "BiLSTM + attention")):
    j = json.loads((T / fn).read_text()); seeds = j["all_seeds"]; c = j["calibrated"]["real"]; gap = per_class_gap(j); worst = sorted(gap.items(), key=lambda kv: kv[1]["delta"])[:4]
    res[key] = dict(name=name, observed=dict(macro_f1=c["macro_f1"], accuracy=c["accuracy"], delta_macro_f1_vs_tfidf=c["macro_f1"] - tf["macro_f1"]),
                    measured=dict(params=j["config"]["params"], train_docs=len(tr), vocab_size=j["config"]["vocab_size"], embeddings="learned from scratch (no pretraining)", epochs_run_per_seed=[s["epochs_run"] for s in seeds], best_epoch_per_seed=[s["best_epoch"] for s in seeds], max_epochs=j["config"]["max_epochs"],
                                  val_macro_f1_per_seed=[s["val_real_macro_f1"] for s in seeds], train_loss_at_last_epoch=[s["history"][-1]["train_loss"] for s in seeds], seed_test_accuracy=[s["accuracy"] for s in j["seeds_test_real"]], seed_test_macro_f1=[s["macro_f1"] for s in j["seeds_test_real"]],
                                  oov_share_of_real_test_word_tokens=oov, oov_share_of_alphabetic_tokens=oov_alpha, share_alphabetic_tokens_longer_than_12_chars=long_alpha, truncation_over_256_word_tokens_real_test_pct=float(100 * np.mean([len(t) > 256 for t in tok_test])),
                                  classes_with_largest_loss_vs_tfidf=[dict(cls=k, **v) for k, v in worst], ece_calibrated=c["ece"]),
                    unverified_hypotheses=["Too little labelled text (about 4,800 documents) to learn word embeddings and recurrence from scratch, whereas a linear model over n-gram counts needs far fewer examples.",
                                           "The word tokeniser meets merged words because OCR drops spaces (e.g. 'GSTID:000433614848'), which become unknown tokens; character n-grams in TF-IDF are insensitive to this.",
                                           "Early stopping on a small validation set (391 real documents) and the large seed-to-seed variation mean that the selected seed is partly luck."],
                    experiments_needed=["Pretrained embeddings or a character/sub-word tokeniser with the same architecture (isolates the tokeniser/OOV effect).", "More seeds and a learning-curve experiment (accuracy vs number of training documents) to test the data-size explanation.", "Same model on whitespace-repaired OCR text."])
# ---------------- LayoutLMv3 ----------------
from transformers import AutoTokenizer
ltk = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base", add_prefix_space=True); ref = json.loads((ROOT / "reports/taskA/layoutlmv3.json").read_text()); lj = json.loads((T / "m6_layoutlmv3.json").read_text()); c = lj["calibrated"]["real"]
counts = []
for r in real_te:
    ws = [t for t, _ in row_words(r)] or ["empty"]; counts.append(len(ltk(ws, boxes=[[0, 0, 1, 1]] * len(ws), add_special_tokens=True)["input_ids"]))
counts = np.array(counts); gap = per_class_gap(lj); worst = sorted(gap.items(), key=lambda kv: kv[1]["delta"])[:4]
res["layoutlmv3"] = dict(name="LayoutLMv3-base (production checkpoint)", observed=dict(macro_f1=c["macro_f1"], accuracy=c["accuracy"], delta_macro_f1_vs_tfidf=c["macro_f1"] - tf["macro_f1"], delta_accuracy_vs_tfidf=c["accuracy"] - tf["accuracy"]),
                         measured=dict(epochs=ref["config"]["epochs"], train_max_len=ref["config"]["max_len"], inference_max_len=512, val_real_macro_f1_by_epoch=[round(h["val_real_macro_f1"], 4) for h in ref["log"]], val_acc_by_epoch=[round(h["val_acc"], 4) for h in ref["log"]], train_seconds=ref["log"][-1]["sec"],
                                       real_test_docs_over_448_tokens_pct=float(100 * (counts > 448).mean()), real_test_docs_over_512_tokens_pct=float(100 * (counts > 512).mean()), word_limit_at_inference=512, uses_image_and_layout=True,
                                       classes_with_largest_loss_vs_tfidf=[dict(cls=k, **v) for k, v in worst], ece_calibrated=c["ece"], accuracy_gap_small_macro_f1_gap_large=bool(abs(c["accuracy"] - tf["accuracy"]) < 0.03 and tf["macro_f1"] - c["macro_f1"] > 0.1)),
                         unverified_hypotheses=["Macro-F1 is dragged down by a few small classes (see classes_with_largest_loss_vs_tfidf) while accuracy, dominated by frequent classes, is close to TF-IDF; with n < 15 per class this is noisy.",
                                                "Fine-tuning for 6 epochs with a 448-token window on ~4,800 documents (batch 16) may under-train a 125M-parameter layout model; validation macro-F1 was still flat-to-rising at the last epoch.",
                                                "The image branch adds little for bills whose class is decided by words; the layout/image inputs are not a disadvantage per se."],
                         experiments_needed=["Train longer / with more seeds and compare validation curves.", "Ablate the image and the boxes (text-only LayoutLM) to see what the extra modalities contribute.", "Per-class analysis on a larger test set."])
# ---------------- LexRank ----------------
L = json.loads((T / "m3_lexrank.json").read_text()); sw = {round(r["frac"], 2) if r["frac"] else None: r for r in L["sweep"]}; ctrl = L.get("controls", [])
res["lexrank"] = dict(name="LexRank summarisation + TF-IDF/LR", observed=dict(macro_f1_50pct=sw[0.5]["macro_f1_real"], accuracy_50pct=sw[0.5]["accuracy_real"], delta_macro_f1_vs_full_text=sw[0.5]["macro_f1_real"] - sw[None]["macro_f1_real"]),
                      measured=dict(sweep=[dict(frac=r["frac"], words=r["words_per_doc"], reduction_pct=r["reduction_pct"], accuracy=r["accuracy_real"], macro_f1=r["macro_f1_real"]) for r in L["sweep"]], controls=[dict(config=c["config"], macro_f1=c["macro_f1_real"], accuracy=c["accuracy_real"], reduction_pct=c["reduction_pct"]) for c in ctrl],
                                    units="OCR lines (bills have no sentences)", lines_with_real_structure_pct=L["lines_coverage"], ocr_task="not applicable: LexRank cannot read pixels"),
                      unverified_hypotheses=["Class evidence on a bill is spread over short lines (merchant, key words, totals); dropping lines removes some of it, and centrality of a line within the bill is not related to its usefulness for classification."],
                      experiments_needed=["Supervised line selection (keep lines whose terms are most predictive) compared with random/first/LexRank at the same budget.", "Paired bootstrap of LexRank vs random-lines on the same documents (the controls here were single or 3-seed runs)."])
(FIN / "outputs" / "other_models_analysis.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: dict(observed=v["observed"], measured={a: b for a, b in v["measured"].items() if not isinstance(b, (list, dict)) or a in ("epochs_run_per_seed", "best_epoch_per_seed", "val_real_macro_f1_by_epoch", "seed_test_accuracy")}) for k, v in res.items()}, indent=1, default=float)[:3800])
