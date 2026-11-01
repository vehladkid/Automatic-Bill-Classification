#!/usr/bin/env bash
# one fresh process per model, nothing else running (demo servers idle); outputs/resources/*.json
cd "$(dirname "$0")"; PY=../../.venv/bin/python; export HF_HUB_OFFLINE=1
for m in tfidf bilstm bilstm_attn; do $PY resource_bench.py $m > logs/bench_$m.out 2>&1; done
$PY resource_bench.py xlmr primary256_s0 > logs/bench_xlmr_primary256_s0.out 2>&1
$PY resource_bench.py xlmr len512_s0 > logs/bench_xlmr_len512_s0.out 2>&1
$PY resource_bench.py layoutlmv3 > logs/bench_layoutlmv3.out 2>&1
echo BENCH_DONE > logs/bench.done
