"""Applicability gating: a category-specific field is only emitted when the document text contains evidence words for it
(no Task A dependency). Prevents e.g. 'check_in' on an electricity bill. Evaluated in final_eval (emission rate on unsupported documents + metric impact)."""
import re
EVID = {"check_in": r"check[- ]?in|arriv", "check_out": r"check[- ]?out|depart", "travel_date": r"journey|travel|depart|\bdep\b|boarding", "origin": r"\bfrom\b|origin|depart|source", "destination": r"\bto\b|destination|arriv|\bdest",
        "pnr": r"pnr", "ticket_number": r"ticket|tkt", "room_number": r"room|rm\.? no", "number_of_nights": r"night", "patient_name": r"patient|\bpt\b", "doctor_name": r"doctor|\bdr\b|consultant|physician|dr\.", "prescription_number": r"\brx\b|prescription|presc",
        "pharmacy_name": r"pharmacy|chemist|medical|apotheke|medplus|pharma", "employee_name": r"employee|\bemp\b|staff", "employee_id": r"employee|\bemp\b|staff", "hsn_sac": r"hsn|\bsac\b", "passenger_name": r"passenger|pax|traveller|traveler",
        "train_number": r"train", "flight_number": r"flight|\bfl\b", "bus_number": r"\bbus\b", "vehicle_number": r"vehicle|veh\b|reg\.? ?no|car no|\bvno\b", "igst": r"igst", "cgst": r"cgst", "sgst": r"sgst|utgst", "gstin": r"gst|tin\b"}
def gate(words, preds):
    text = " ".join(words).lower(); out = {}
    for f, p in preds.items():
        rx = EVID.get(f)
        if rx and not re.search(rx, text): continue
        out[f] = p
    return out
