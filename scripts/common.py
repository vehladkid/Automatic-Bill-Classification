"""Shared paths + project taxonomy. Single source of truth for labels."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets"
PUBLIC, INDIAN, SYNTH = DS / "public", DS / "indian", DS / "synthetic"
INTERIM, PROCESSED, META, SPLITS = DS / "interim", DS / "processed", DS / "metadata", DS / "splits"
SEED = 42

# Task A: flat expense_category -> parent group (the hierarchy HR policy will hang rules on).
CATEGORIES = {
    "flight": "travel", "train": "travel", "bus": "travel", "transport": "travel",  # transport = cab/auto/taxi
    "hotel": "accommodation", "rent": "other_business",
    "food": "food",
    "electricity": "utilities", "water": "utilities", "gas": "utilities", "internet": "utilities",
    "fuel": "transport_fuel", "parking": "transport_fuel",
    "medical": "medical",
    "miscellaneous": "other_business",  # office supplies, groceries, equipment, subscriptions
    "non_expense": "non_expense",       # letters, forms, memos, ... (negatives for the gate)
}
# Task C: document_type is independent of category (an invoice can be food OR hotel OR office).
DOC_TYPES = ["receipt", "invoice", "utility_bill", "travel_document", "hotel_bill",
             "fuel_receipt", "medical_bill", "other", "non_expense_document"]
# Default doc type when a source only knows the category (manual folders, synthetic).
DEFAULT_DOC_TYPE = {
    "flight": "travel_document", "train": "travel_document", "bus": "travel_document",
    "transport": "receipt", "hotel": "hotel_bill", "rent": "invoice", "food": "receipt",
    "electricity": "utility_bill", "water": "utility_bill", "gas": "utility_bill",
    "internet": "utility_bill", "fuel": "fuel_receipt", "parking": "receipt",
    "medical": "medical_bill", "miscellaneous": "invoice", "non_expense": "non_expense_document",
}
# label_quality values usable for Task A training/eval ("unlabeled"/"weak_rule" are not).
TRUSTED = {"manual_folder", "source_label", "synthetic", "rule_high", "manual_review"}

def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
