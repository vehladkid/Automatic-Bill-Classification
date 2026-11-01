"""Pre-registered selection among the EXPLORATORY XLM-R variants: the variant with the highest VALIDATION real-only macro-F1 of seed 0 (test data are not read). Writes configs/selected_variant.json."""
import json, sys
from pathlib import Path
FIN = Path(__file__).resolve().parent; cands = ["len512", "chunk256", "ep10"]; rows = []
for n in cands:
    p = FIN / "runs" / f"{n}_s0" / "config.json"
    if p.exists():
        c = json.loads(p.read_text()); rows.append(dict(name=n, status=c["status"], val=c["best_val_real_macro_f1"], epoch=c["selected_epoch"], train_s=c["train_seconds"]))
ok = [r for r in rows if r["status"] == "completed"]; best = max(ok, key=lambda r: (r["val"], -r["train_s"])) if ok else None
(FIN / "configs" / "selected_variant.json").write_text(json.dumps(dict(candidates=rows, selected=best["name"] if best else None, rule="highest seed-0 validation real-only macro-F1 among completed exploratory variants (primary256 is pre-registered separately)"), indent=1)); print(json.dumps(rows), "->", best and best["name"])
