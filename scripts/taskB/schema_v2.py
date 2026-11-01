"""Task B V2 schema = frozen V1 schema (schema.py, unchanged) + V2 additions. V1 scalar `hsn_sac` stays as-is for V1 code; V2 uses a list.
Additions: hsn_sac -> id_list (unique codes, digits only), taxable_value, discount, round_off (signed). `subtotal` keeps its V1 meaning (explicit invoice subtotal)."""
import re, json
from pathlib import Path
from schema import FIELDS as FIELDS_V1, GROUPS as GROUPS_V1, normalize as normalize_v1, parse_amount
FIELDS = dict(FIELDS_V1); FIELDS["hsn_sac"] = "id_list"
FIELDS.update({"taxable_value": "amount", "discount": "amount", "round_off": "signed_amount"})
GROUPS = {k: list(v) for k, v in GROUPS_V1.items()}
GROUPS["universal"] += ["taxable_value", "discount", "round_off"]
def norm_hsn_list(v):
    if v in (None, "", []): return []
    items = v if isinstance(v, (list, tuple)) else re.split(r"[,;/ ]+", str(v))
    out = []
    for x in items:
        d = re.sub(r"\D", "", str(x))
        if d and d not in out: out.append(d)
    return out
def parse_signed(s):
    t = str(s).strip(); neg = t.startswith("-") or t.startswith("(-") or t.endswith("-")
    a = parse_amount(re.sub(r"[()\-]", "", t)); return None if a is None else (-a if neg else a)
def normalize(field, value, **kw):
    t = FIELDS[field]
    if t == "id_list": return norm_hsn_list(value)
    if t == "signed_amount": return None if value in (None, "") else parse_signed(value)
    if field not in FIELDS_V1:   # new V2 amount fields
        return None if value in (None, "") else parse_amount(value)
    return normalize_v1(field, value, **kw)
def values_match(field, a, b):
    t = FIELDS[field]
    if t == "id_list": return set(a or []) == set(b or [])      # set equality; per-element P/R reported separately
    if t in ("amount", "signed_amount"): return a is not None and b is not None and abs(a - b) <= 0.005
    return a == b and a is not None
def save_schema(path):
    json.dump({"version": "taskB-v2", "fields": FIELDS, "groups": GROUPS, "changes_vs_v1": {"hsn_sac": "scalar id -> list of unique digit-only codes", "taxable_value": "new", "discount": "new (invoice-level printed discount; kind recorded in metadata)", "round_off": "new, signed as printed", "subtotal": "unchanged meaning: explicit invoice subtotal only; never derived, never = taxable value"}, "unchanged": "V1 schema.py / models/taskB/field_schema.json"}, open(path, "w"), indent=1)
if __name__ == "__main__":
    save_schema(Path(__file__).resolve().parents[2] / "models/taskB/v2/field_schema_v2.json")
