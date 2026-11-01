#!/usr/bin/env bash
# waits for the two checkpoint downloads, then runs the remaining GPU jobs one after another (never two GPU jobs at once)
cd "$(dirname "$0")"; PY=../../.venv/bin/python
until grep -qE "^(complete|INCOMPLETE)" logs/fetch_trocr.log; do sleep 5; done
if grep -q "^complete" logs/fetch_trocr.log; then HF_HUB_OFFLINE=1 $PY ocr_exp.py trocr > logs/ocr_trocr.log 2>&1; else echo "TrOCR download incomplete" > logs/ocr_trocr.log; fi
until grep -qE "^(complete|INCOMPLETE)" logs/fetch_xlmr.log; do sleep 5; done
if grep -q "^complete" logs/fetch_xlmr.log; then HF_HUB_OFFLINE=1 $PY m5_xlmr.py > logs/m5_xlmr.log 2>&1; else echo "XLM-R download incomplete" > logs/m5_xlmr.log; fi
echo ALL_DONE > logs/run_rest.done
