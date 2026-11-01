import json
from schema import *
TB = PROCESSED / "taskB"
def load(split):
    return [json.loads(l) for l in open(TB / f"{ {'val': 'validation'}.get(split, split)}.jsonl")]
def bbox_of(doc, idx):
    return [min(doc["boxes"][i][0] for i in idx), min(doc["boxes"][i][1] for i in idx), max(doc["boxes"][i][2] for i in idx), max(doc["boxes"][i][3] for i in idx)] if idx else None
