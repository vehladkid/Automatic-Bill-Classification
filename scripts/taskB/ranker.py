"""Model B1: candidate generation + gradient-boosted candidate ranking per field (sklearn HistGradientBoosting; CPU).
 candidates: amounts / dates / 15-char tokens (GSTIN) / id-like tokens (+2-3 word windows) / text lines, ':'-suffixes and 2-3 line windows.
 features: value statistics, page position, line shape, hashed left-context / previous-line / next-line words. One classifier per field.
 decision: best candidate per field; abstain if score < tau (tuned on validation); confidence = isotonic-calibrated P(correct) fit on validation."""
import re, hashlib, pickle
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from schema import *
from words import AMT_RE, DATE_RES
from baseline import lines_of, line_amounts, infer_locale, infer_dayfirst, ADDR_KW
from data import bbox_of

HV = HashingVectorizer(n_features=192, analyzer=str.split, alternate_sign=False, norm=None, binary=True, lowercase=False)
AMOUNT_F = ["subtotal", "tax_total", "total_amount", "cgst", "sgst", "igst"]; DATE_F = ["invoice_date", "travel_date", "check_in", "check_out"]
ID_F = ["invoice_number", "hsn_sac", "pnr", "ticket_number", "employee_id", "room_number", "prescription_number", "train_number", "flight_number", "bus_number", "vehicle_number"]
TEXT_F = ["merchant_name", "merchant_address", "employee_name", "passenger_name", "patient_name", "doctor_name", "pharmacy_name", "origin", "destination"]
TYPE_OF = {**{f: "amount" for f in AMOUNT_F}, **{f: "date" for f in DATE_F}, "gstin": "gstin", **{f: "id" for f in ID_F}, **{f: "text" for f in TEXT_F}}
STOP = re.compile(r"[^a-z]+")
def toks(s, pre, k=8): return " ".join(pre + t for t in STOP.split(s.lower()) if 1 < len(t) < 18)[:400]

def struct(d):
    L = lines_of(d["words"], d["boxes"]); full = "\n".join(l["text"] for l in L); H = max(d["height"], 1); W = max(d["width"], 1)
    w2l = {i: li for li, l in enumerate(L) for i in l["idx"]}
    return dict(L=L, full=full, H=H, W=W, w2l=w2l, locale=infer_locale(full), dayfirst=infer_dayfirst(full), maxh=max((l["h"] for l in L), default=1))

def ctx(S, d, li, first_word_idx):
    """hashed context text for a candidate starting at word index in line li"""
    L = S["L"]; l = L[li]; left = [d["words"][i] for i in l["idx"] if i < first_word_idx][-6:]
    s = toks(" ".join(left), "L_") + " " + toks(L[li - 1]["text"] if li > 0 else "", "P_") + " " + toks(L[li + 1]["text"] if li + 1 < len(L) else "", "N_")
    return s

def cands_amount(d, S):
    out = []; allv = []
    for li, l in enumerate(S["L"]):
        for v, idx in line_amounts(l, d["words"], S["locale"]): allv.append((v, idx, li))
    if not allv: return out
    vals = sorted({v for v, _, _ in allv}, reverse=True); mx = max(vals) or 1; cnt = {}
    for v, _, _ in allv: cnt[v] = cnt.get(v, 0) + 1
    per_line = {}
    for v, idx, li in allv: per_line.setdefault(li, []).append(v)
    for k, (v, idx, li) in enumerate(allv):
        b = bbox_of(d, idx) or [0, 0, 0, 0]; l = S["L"][li]; raw = " ".join(d["words"][i] for i in idx)
        dense = [np.log1p(abs(v)), vals.index(v) / max(len(vals), 1), v / mx, float(v == mx), len(re.sub(r"^.*[.,]", "", raw)) if re.search(r"[.,]\d{1,3}$", raw) else 0, float(bool(re.search(r"[₹$€]|rs|rm|rp", raw, re.I))),
                 (b[1] + b[3]) / 2 / S["H"], (b[0] + b[2]) / 2 / S["W"], li / len(S["L"]), cnt[v], len(per_line[li]), float(per_line[li][-1] == v), per_line[li].index(v) / len(per_line[li]), l["h"] / S["maxh"], float(v == int(v)), float(0 < v <= 100), len(re.sub(r"\D", "", raw)), k / len(allv)]
        out.append(dict(norm=v, disp=raw, idx=idx, dense=dense, ctx=ctx(S, d, li, idx[0])))
    return out

def cands_date(d, S):
    out = []; n = 0
    for li, l in enumerate(S["L"]):
        seen = set()
        for rx in DATE_RES:
            for m in rx.finditer(l["text"]):
                dd = parse_date(m.group(), S["dayfirst"])
                if not dd or (m.start(), dd) in seen: continue
                seen.add((m.start(), dd)); first = [i for i in l["idx"] if d["words"][i] in m.group().split() or m.group().startswith(d["words"][i])] or l["idx"][:1]
                idx = [i for i in l["idx"] if d["words"][i] in m.group().replace(",", " ").split()] or first; b = bbox_of(d, idx)
                out.append(dict(norm=dd, disp=m.group(), idx=idx, dense=[(b[1] + b[3]) / 2 / S["H"], (b[0] + b[2]) / 2 / S["W"], li / len(S["L"]), n, int(dd[:4]) - 2000, int(dd[5:7]), int(dd[8:]), float(bool(re.search(r"[a-z]{3}", m.group(), re.I)))], ctx=ctx(S, d, li, idx[0])))
                n += 1
    for c in out: c["dense"].append(len(out))
    return out

def cands_gstin(d, S):
    out = []; cl = [(i, re.sub(r"[^A-Z0-9]", "", w.upper())) for i, w in enumerate(d["words"])]; cl = [(i, repair_gstin(c) or c) for i, c in cl if len(c) == 15]
    for k, (i, c) in enumerate(cl):
        li = S["w2l"][i]; b = d["boxes"][i]
        dense = [float(gstin_valid_format(c)), float(gstin_checksum_ok(c)), float(bool(re.match(r"^\d{2}", c))), sum(ch.isdigit() for ch in c), float(c[13:14] == "Z"), (b[1] + b[3]) / 2 / S["H"], k, len(cl), float(bool(re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z]", c)))]
        out.append(dict(norm=c, disp=d["words"][i], idx=[i], dense=dense, ctx=ctx(S, d, li, i)))
    return out

def cands_id(d, S):
    out = []; W = d["words"]; cl = [norm_id(w) for w in W]
    for i, c in enumerate(cl):
        for n in (1, 2, 3):
            if i + n > len(W) or (n > 1 and S["w2l"][i] != S["w2l"][i + n - 1]): break
            cc = "".join(cl[i:i + n])
            if not (3 <= len(cc) <= 25) or not re.search(r"\d", cc): continue
            if n > 1 and (len(cc) < 5): continue
            b = bbox_of(d, list(range(i, i + n))); li = S["w2l"][i]; raw = " ".join(W[i:i + n]); dg = sum(ch.isdigit() for ch in cc)
            dense = [n, len(cc), dg, dg / len(cc), float("/" in raw), float("-" in raw), float(cc.isdigit()), float(cc[0].isalpha()), float(bool(re.fullmatch(r"\d+[.,]\d{2}", raw))), float(bool(parse_date(raw))), float(len(cc) == 15), float(len(cc) in (10, 11, 12) and cc.isdigit()),
                     (b[1] + b[3]) / 2 / S["H"], (b[0] + b[2]) / 2 / S["W"], li / len(S["L"]), S["L"][li]["idx"].index(i) / len(S["L"][li]["idx"]), float(W[i - 1].endswith(":")) if i else 0.0]
            out.append(dict(norm=cc, disp=raw, idx=list(range(i, i + n)), dense=dense, ctx=ctx(S, d, li, i) + " " + toks(" ".join(W[max(0, i - 2):i]), "B_")))
    return out

def cands_text(d, S):
    out = []; L = S["L"]; W = d["words"]
    def mk(idx, li, kind, li_end=None):
        raw = " ".join(W[i] for i in idx); b = bbox_of(d, idx); letters = sum(c.isalpha() for c in raw) or 1; ln = max(len(raw), 1)
        dense = [li / len(L), (b[1] + b[3]) / 2 / S["H"], (b[0] + b[2]) / 2 / S["W"], (b[3] - b[1]) / S["maxh"], len(idx), ln, letters / ln, sum(c.isdigit() for c in raw) / ln, sum(c.isupper() for c in raw) / letters, raw.count(","), float(":" in raw),
                 float(bool(ADDR_KW.search(raw))), float(bool(re.search(r"\b\d{10}\b|\+\d{2}", raw))), float(bool(re.search(r"\b\d{5,6}\b", raw))), kind, (li_end - li + 1) if li_end is not None else 1, float(bool(re.search(r"gst|tax|invoice|bill|total|date", raw, re.I)))]
        return dict(norm=norm_text(raw), disp=raw, idx=idx, dense=dense, ctx=toks(raw, "T_") + " " + toks(L[li - 1]["text"] if li else "", "P_") + " " + toks(" ".join(W[i] for i in L[li]["idx"] if i < idx[0])[-40:], "K_"))
    for li, l in enumerate(L):
        if len(re.findall(r"[A-Za-z]", l["text"])) < 2 and not re.search(r"[A-Za-z]{2}", l["text"]): continue
        out.append(mk(l["idx"], li, 0))
        for p, i in enumerate(l["idx"]):                       # suffix after a 'label:' token
            if W[i].endswith(":") and p + 1 < len(l["idx"]): out.append(mk(l["idx"][p + 1:], li, 1)); break
        for n in (2, 3):
            if li + n <= len(L): out.append(mk([i for k in range(li, li + n) for i in L[k]["idx"]], li, 2, li + n - 1))
        low = [W[i].lower().strip(":") for i in l["idx"]]      # 'From: X To: Y' on one line -> separate X / Y candidates (origin / destination)
        if "from" in low:
            a = low.index("from"); b = low.index("to", a + 1) if "to" in low[a + 1:] else None
            if b is not None and b > a + 1: out.append(mk(l["idx"][a + 1:b], li, 3))
            if b is not None and b + 1 < len(low): out.append(mk(l["idx"][b + 1:], li, 4))
    return out

GEN = {"amount": cands_amount, "date": cands_date, "gstin": cands_gstin, "id": cands_id, "text": cands_text}

def featurize(cs):
    if not cs: return np.zeros((0, 1), np.float32)
    D = np.array([c["dense"] for c in cs], np.float32); X = HV.transform([c["ctx"] for c in cs]).toarray().astype(np.float32); return np.hstack([D, X])

def label_of(f, c, g):
    if g is None: return 0
    t = FIELDS[f]
    if t == "amount": return int(amount_eq(c["norm"], g))
    if t in ("text", "place"): return int(c["norm"] == g or similarity(c["norm"], g) >= 0.9) if f not in ("origin", "destination") else int(norm_place(c["disp"]) == g or (len(g) > 3 and g in c["norm"]))
    return int(c["norm"] == g)

def build_cands(docs, types=None):
    """-> {doc_id: {type: [cands]}} ; structure computed once per doc"""
    out = {}
    for d in docs:
        if not d["words"]: out[d["document_id"]] = {}; continue
        S = struct(d); out[d["document_id"]] = {t: GEN[t](d, S) for t in (types or GEN)}
    return out

def make_xy(docs, CD, field):
    t = TYPE_OF[field]; X = []; y = []; own = []
    for d in docs:
        if field not in d["supported_fields"]: continue
        cs = CD[d["document_id"]].get(t, [])
        if not cs: continue
        g = d["fields"].get(field); g = g["norm"] if g else None; F = featurize(cs); X.append(F); y += [label_of(field, c, g) for c in cs]; own += [d["document_id"]] * len(cs)
    return (np.vstack(X), np.array(y), own) if X else (None, None, None)

def train_field(docs, CD, field):
    X, y, _ = make_xy(docs, CD, field)
    if X is None or y.sum() < 5: return None
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_depth=7, min_samples_leaf=10, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
    return clf.fit(X, y)

def predict_doc(models, d, CD, taus=None, fields=None):
    out = {}
    for f, clf in models.items():
        if fields and f not in fields: continue
        cs = CD[d["document_id"]].get(TYPE_OF[f], [])
        if not cs: continue
        p = clf.predict_proba(featurize(cs))[:, 1]; best = {}
        for c, s in zip(cs, p):
            k = c["norm"]
            if k not in best or s > best[k][0]: best[k] = (float(s), c)
        s, c = max(best.values(), key=lambda t: t[0])
        out[f] = dict(value=c["disp"], norm=c["norm"], word_idx=c["idx"], bbox=bbox_of(d, c["idx"]), score=s, source="ranker", alternatives=sorted([(round(a, 4), k) for k, (a, _) in best.items()], reverse=True)[1:3])
    return out
