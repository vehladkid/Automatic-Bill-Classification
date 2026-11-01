"""Redraws the Section 4.4 comparison figures with spelled-out labels (no CER/WER/ECE/CI/LR abbreviations) WITHOUT touching experiments/text_vs_neural:
it executes a patched in-memory copy of aggregate.py whose figure folder is docs/report/figures_spelled and whose tables/results.json writes are disabled.
usage: PYTHONPATH=<dir with matplotlib> .venv/bin/python scripts/relabel_comparison_figures.py"""
import sys, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; EXPD = ROOT / "experiments/text_vs_neural"; OUTF = ROOT / "docs/report/figures_spelled"; OUTF.mkdir(parents=True, exist_ok=True)
src = (EXPD / "aggregate.py").read_text()
src = src.replace('FIG = EXP / "figures"; FIG.mkdir(exist_ok=True)', f'FIG = Path(r"{OUTF}"); FIG.mkdir(exist_ok=True)')
src = src.replace('with open(EXP / "table_', 'with open(Path("/dev/null") if False else (Path(r"' + str(OUTF) + '") / "_discard_table_"), "w") as _unused_, open(EXP / "table_') if False else src
src = re.sub(r'with open\(EXP / ("table_[A-Za-z_]+\.csv"), "w", newline=""\) as fh:', r'with open(Path(r"' + str(OUTF) + r'") / "_discard.csv", "w", newline="") as fh:', src)
src = src.replace('save_json("results.json"', '(lambda *a, **k: None)("results.json"')
for a, b in (("95% group-bootstrap CI", "95% group-bootstrap confidence interval"), ('"ECE (10 bins, lower is better)"', '"Expected calibration error (10 bins, lower is better)"'), ('label="CER strict"', 'label="Character error rate, strict"'), ('label="CER case/space-insensitive"', 'label="Character error rate, case/space-insensitive"'),
             ('label="WER strict"', 'label="Word error rate, strict"'), ('label="WER case-insensitive"', 'label="Word error rate, case-insensitive"'), ('.replace("TF-IDF + Logistic Regression", "TF-IDF + LR")', '.replace("TF-IDF + Logistic Regression", "TF-IDF + logistic regression")'), ('"LexRank 50% + TF-IDF/LR"', '"LexRank 50% + TF-IDF/logistic regression"')):
    src = src.replace(a, b)
sys.path.insert(0, str(EXPD)); import os; os.chdir(EXPD)
g = {"__file__": str(EXPD / "aggregate.py"), "__name__": "__main__"}; exec(compile(src, "aggregate_spelled", "exec"), g)
for f in OUTF.glob("_discard*"): f.unlink()
print("figures:", sorted(p.name for p in OUTF.glob("*.png")))
