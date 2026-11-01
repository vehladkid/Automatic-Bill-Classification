#!/usr/bin/env bash
# Sequential GPU queue (never two GPU jobs at once). Logs in logs/. HF_HUB_OFFLINE=1.
cd "$(dirname "$0")"; PY=../../.venv/bin/python; export HF_HUB_OFFLINE=1
while pgrep -f "xlmr_run.py --name primary256 --seed 0" > /dev/null; do sleep 5; done        # the seed-0 run already started by hand
for s in 1 2 3 4; do $PY xlmr_run.py --name primary256 --seed $s > logs/primary256_s$s.out 2>&1; done
$PY xlmr_run.py --name len512   --seed 0 --maxlen 512 --bs 16 --accum 2 > logs/len512_s0.out 2>&1
$PY xlmr_run.py --name chunk256 --seed 0 --chunk 1 --maxlen 256 --stride 64 > logs/chunk256_s0.out 2>&1
$PY xlmr_run.py --name ep10     --seed 0 --epochs 10 > logs/ep10_s0.out 2>&1
$PY select_variant.py > logs/select_variant.out 2>&1
SEL=$($PY -c "import json;print(json.load(open('configs/selected_variant.json'))['selected'] or '')")
if [ -n "$SEL" ]; then
  case $SEL in
    len512)   ARGS="--maxlen 512 --bs 16 --accum 2";;
    chunk256) ARGS="--chunk 1 --maxlen 256 --stride 64";;
    ep10)     ARGS="--epochs 10";;
  esac
  for s in 1 2 3 4; do $PY xlmr_run.py --name $SEL --seed $s $ARGS > logs/${SEL}_s$s.out 2>&1; done
fi
echo QUEUE_DONE > logs/queue.done
