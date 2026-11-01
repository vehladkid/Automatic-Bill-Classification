"""Shared helpers for the final XLM-R vs TF-IDF study: split loading (re-uses the verified exp_common of the earlier study, unchanged), tokenisation policies, checkpoint save/load."""
import sys, json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; FIN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments" / "text_vs_neural"))
from exp_common import *            # noqa: E402,F401  (load via splits(), softmax, cls_metrics, fit_T, evaluate_model, paired_bootstrap, time_fn ...)
PRETRAINED = ROOT / "experiments/text_vs_neural/pretrained/xlm-roberta-base"

def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(PRETRAINED))

def encode_docs(tok, rows, maxlen, chunk=False, stride=64, max_chunks=6):
    """-> (list of token-id lists, list of doc indices). maxlen includes the two special tokens (<s>, </s>).
    chunk=False: first maxlen tokens only (the rest of the document is discarded).  chunk=True: overlapping windows of maxlen tokens (stride = overlap in tokens), at most max_chunks per document."""
    ids, owner = [], []
    for i, r in enumerate(rows):
        if not chunk:
            ids.append(tok(r["text"], truncation=True, max_length=maxlen)["input_ids"]); owner.append(i)
        else:
            e = tok(r["text"], truncation=True, max_length=maxlen, stride=stride, return_overflowing_tokens=True)["input_ids"][:max_chunks]
            for c in e: ids.append(c); owner.append(i)
    return ids, np.array(owner)

def save_trainable(model, path):
    """only the parameters that were trained (the word-embedding matrix is frozen = identical to the pretrained one), stored as fp16 to fit the disk"""
    from safetensors.torch import save_file
    sd = {n: p.detach().cpu().half().contiguous() for n, p in model.named_parameters() if p.requires_grad}; save_file(sd, str(path)); return sum(v.numel() for v in sd.values())

def load_model(trainable_path=None, num_labels=16):
    """pretrained XLM-R + (optional) fine-tuned trainable tensors; weights are cast back to fp32"""
    from transformers import XLMRobertaForSequenceClassification
    m = XLMRobertaForSequenceClassification.from_pretrained(str(PRETRAINED), num_labels=num_labels)
    if trainable_path:
        from safetensors.torch import load_file
        sd = {k: v.float() for k, v in load_file(str(trainable_path)).items()}; miss = m.load_state_dict(sd, strict=False); assert not miss.unexpected_keys, miss.unexpected_keys
    return m
