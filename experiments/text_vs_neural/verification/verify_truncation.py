"""How much text do the length limits cut off? XLM-R (256 SentencePiece tokens) and the BiLSTMs (256 word tokens), per split. Tokenisation only."""
import sys, json, re
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "scripts/taskA")); from common_a import load
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(str(ROOT / "experiments/text_vs_neural/pretrained/xlm-roberta-base")); TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]"); out = {}
for s in ("train", "val", "test"):
    rows = load(s); real = [not r["synthetic"] for r in rows]; xl = np.array([len(tk(r["text"], truncation=False)["input_ids"]) for r in rows]); bw = np.array([len(TOK.findall(r["text"].lower())) for r in rows]); real = np.array(real)
    out[s] = dict(n=len(rows), xlmr_tokens_median=float(np.median(xl)), xlmr_over_256_pct=float(100 * (xl > 256).mean()), xlmr_over_256_pct_real=float(100 * (xl[real] > 256).mean()), bilstm_tokens_median=float(np.median(bw)), bilstm_over_256_pct=float(100 * (bw > 256).mean()), bilstm_over_256_pct_real=float(100 * (bw[real] > 256).mean()))
    print(s, {k: round(v, 1) for k, v in out[s].items()})
json.dump(out, open(ROOT / "experiments/text_vs_neural/verification/truncation_check.json", "w"), indent=1)
