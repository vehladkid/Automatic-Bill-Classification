"""Post-V1 sanity filter. V1 (frozen) sometimes emits total_amount = 0.0 (seen on 8 validation documents, all wrong; the real total was printed). A zero total is reported as missing
instead of as a confident-looking wrong value. Nothing is invented: the field is removed and the removal is recorded."""
def drop_implausible(fields):
    kept, dropped = {}, []
    for f, p in (fields or {}).items():
        v = (p or {}).get("value")
        if f == "total_amount" and isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
            dropped.append(dict(field=f, value=v, reason="a total of 0 is not plausible; treated as missing")); continue
        kept[f] = p
    return kept, dropped
