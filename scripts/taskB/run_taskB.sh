#!/usr/bin/env bash
# Rebuild Task B end to end (needs the Task A dataset/manifest to exist). Run from repo root: bash scripts/taskB/run_taskB.sh
set -euo pipefail
cd "$(dirname "$0")"; PY=../../.venv/bin/python
$PY generate_synthetic_b.py        # synthetic_b (HSN/SAC, hotel stay, medical, employee, passenger/ticket/bus), seed 4242
$PY build_taskB.py                 # canonical annotations + alignment + audit + schema mapping
$PY run_baseline.py                # regex/keyword baseline
OMP_NUM_THREADS=8 $PY run_ranker.py && OMP_NUM_THREADS=8 $PY cv_ranker.py   # candidate ranker + group CV
$PY build_cache_b.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 $PY ie_layoutlm.py   # GPU ~45 min (process may need to be killed after 'DONE')
$PY final_eval.py && $PY make_report.py
