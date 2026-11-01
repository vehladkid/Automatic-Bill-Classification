"""Applies the human label review (visual + OCR-text) done 2026-10-05 as entries in datasets/metadata/label_overrides.json (merged with earlier ones).
Every override keeps its reason. Re-run create_splits.py + validate_dataset.py afterwards (no re-dedup needed). Idempotent."""
import json, re
from common_a import *
MY = "my_receipts__"
MANUAL = {  # id -> (new label, reason)
 MY+"2023_us_wynn_20231209_014": ("food", "Wynn steakhouse check, regex had put it in transport"),
 MY+"2021_us_snacks-20211211_002": ("food", "restaurant check at hotel casino ('Snacks'), not a lodging bill"),
 MY+"2021_us_snacks-20211211_019": ("food", "restaurant check at hotel casino ('Snacks')"), MY+"2022_us_snacks_20221210_001": ("food", "restaurant check at hotel casino ('Snacks')"),
 MY+"2022_us_snacks_20221210_015": ("food", "restaurant check at hotel casino ('Snacks')"),
 MY+"2021_hr_20210828_VALAMAR": ("hotel", "Valamar Argosy hotel bill, filed as food"), MY+"2024_nl_motelone_20240203": ("hotel", "Motel One invoice, regex had put it in train"),
 MY+"2024_nl_happyitaly_20240306_001": ("food", "restaurant (Happy Italy), regex train"), MY+"2023_be_docks_20231017_004": ("food", "Docks dine-in order, regex train"),
 MY+"2024_us_mcdonald_20240529": ("food", "McDonald's, regex train"), MY+"2023_de_mcdonalds_20230512": ("food", "McDonald's, regex transport"), MY+"2023_us_subway_20231209_012": ("food", "Subway sandwich shop, regex transport"),
 MY+"2021_us_coffeebean-20211211_016": ("food", "Coffee Bean cafe, regex train"), MY+"2024_cz_cloudone_25240713": ("hotel", "The Cloud One Prague hotel, regex train"),
 MY+"2021_hr_2021-09-02-hotel_amarin": ("hotel", "Hotel Amarin bill, regex medical"),
 MY+"2022_de_deichmann-20220611": ("miscellaneous", "Deichmann shoe store, regex transport"), MY+"2023_de_ueding_20230320": ("miscellaneous", "IT goods invoice, regex transport"),
 MY+"2022_us_walmart_20221228": ("miscellaneous", "Walmart retail, regex transport"), MY+"2021_de_cyberport-20210403_Rechnung": ("miscellaneous", "Cyberport electronics store, regex transport"),
 MY+"2021_de_cyberport-20210813_Rechnung": ("miscellaneous", "Cyberport electronics store, regex hotel"), MY+"2023_de_europcar_20231009": ("miscellaneous", "car-rental contract, regex fuel"),
 MY+"2023_us_citycar_19830107_Payments": ("miscellaneous", "car-rental payments, regex fuel"), MY+"2024_de_dm_20240701": ("miscellaneous", "dm drugstore, regex fuel"),
 MY+"2022_es_grupo-20220610_010": ("miscellaneous", "supermarket (Grupo Dia), regex train"),
 MY+"2018_ca_public_transport_translink_20180908_016": ("transport", "Vancouver TransLink local transit, regex train"), MY+"2019_nl_public_transport_gvbcharge_20191019_007": ("transport", "Amsterdam GVB local transit, regex train"),
 "receiptbench__2_SubwayTicket_91": ("train", "Cape Town Metrorail commuter-rail ticket, typed 'subway' by source"), "receiptbench__2_FlightTicket_465": ("train", "Trenitalia regional train ticket, typed 'flight' by source"),
 "receiptbench__2_TrainTicket_38": ("train", "ReceiptBench type was already train; kept (listed for audit trail)") ,
}
# ids found by rank in the audit CSV (receiptbench / sroie / pt)
BY_SUBSTR = [("receiptbench__2_TrainTicket_38", None), ]
def main():
    ov = json.loads((META / "label_overrides.json").read_text()) if (META / "label_overrides.json").exists() else {}
    man = {}
    for l in open(META / "manifest_dedup.jsonl"):
        r = json.loads(l); man[r["document_id"]] = r
    n = 0
    def put(i, c, why):
        nonlocal n
        if i in man and man[i]["expense_category"] != c and (i not in ov or ov[i]["expense_category"] != c): ov[i] = {"expense_category": c, "reason": why}; n += 1
    for i, (c, why) in MANUAL.items():
        if not why.startswith("ReceiptBench type was already"): put(i, c, why)
    # SROIE: Tealive bubble-tea shop sits at a Shell station; company/address matched 'SHELL'
    for i, r in man.items():
        if r["source_dataset"] == "sroie" and r["expense_category"] == "fuel":
            t = " ".join(w["text"] for w in json.loads((DS / r["ann_path"]).read_text()).get("words", []))
            if re.search(r"tealive|bubble tea|restaurant|cafe|\bmamak\b", t, re.I) and not re.search(r"petrol|ron ?9[5-7]|diesel|pump", t, re.I): put(i, "food", "SROIE: food outlet at a fuel station, rule matched the station name")
    # ReceiptBench India 'PurchaseTicket' docs are restaurant/cafe bills (not generic purchases) -> food, except the few non-food ones
    KEEP = {"TASMAC": "miscellaneous", "JAWED HABIB": "miscellaneous", "Ambience Mall": "miscellaneous", "THELEELA AMBIENCE": "hotel", "The Leela Ambience": "hotel"}
    for i, r in man.items():
        if r["source_dataset"] == "receiptbench" and "PurchaseTicket" in i and r["country"] == "IN" and r["expense_category"] == "miscellaneous":
            m = (r.get("merchant") or "").strip(); c = next((v for k, v in KEEP.items() if k.lower() in m.lower()), "food")
            put(i, c, "ReceiptBench Indian 'PurchaseTicket': dine-in/restaurant bill (checked by merchant + text)" if c == "food" else f"ReceiptBench Indian 'PurchaseTicket': {m} is not a food outlet")
    # PT hotel docs that are restaurant tickets of a hotel restaurant
    for i, r in man.items():
        if r["source_dataset"] == "invoices_pt" and r["expense_category"] == "hotel" and "terra nostra" in (r.get("merchant") or ""): put(i, "food", "hotel-restaurant ticket (food items), not a lodging invoice")
    # retailers filed as food by the my_receipts text rule -> miscellaneous (groceries/drugstores per taxonomy)
    RET = r"\brewe\b|\bedeka\b|\bpenny\b|\baldi\b|\blidl\b|kaufland|\bnetto\b|rossmann|drogerie|supermarkt|safeway|walgreens|\bcvs\b|\bcoop\b|carrefour|tesco|sainsbury|\bspar\b|woolworths|trader joe|whole foods|old navy"
    for i, r in man.items():
        if r["source_dataset"] == "my_receipts" and r["expense_category"] == "food" and r["label_quality"] == "rule_high" and r["ann_path"]:
            t = " ".join(w["text"] for w in json.loads((DS / r["ann_path"]).read_text()).get("words", []))
            if re.search(RET, t[:400], re.I): put(i, "miscellaneous", "grocery/drugstore retailer filed as food by regex")
    (META / "label_overrides.json").write_text(json.dumps(ov, indent=1, ensure_ascii=False)); print("overrides total", len(ov), "new", n)
    import collections; print(collections.Counter(v["expense_category"] for v in ov.values()))
if __name__ == "__main__": main()
