#!/usr/bin/env bash
# sensitivity analysis: seeds 5-9 of the primary recipe (see PREREGISTRATION.md addendum); nothing else runs on the machine meanwhile
cd "$(dirname "$0")"; PY=../../.venv/bin/python; export HF_HUB_OFFLINE=1
for s in 5 6 7 8 9; do $PY xlmr_run.py --name primary256 --seed $s > logs/primary256_s$s.out 2>&1; done
echo EXTRA_DONE > logs/extra.done
