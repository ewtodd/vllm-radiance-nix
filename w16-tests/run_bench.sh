#!/usr/bin/env bash
# usage: run_bench.sh <model-dir> <out.json> [KEY=VALUE ...]
set -uo pipefail
MODEL="$1"; OUT="$2"; shift 2
source /scratch/w16/env.sh
export PYTHONPATH=/scratch/w16/shadow:/scratch/w16
for kv in "$@"; do export "$kv"; done
exec /nix/store/bfy6kchprah5ywrj2s9c6fqslfcp9vkm-python3-3.14.7-env/bin/python \
  /scratch/w16/bench_llm.py --model "$MODEL" --mode both --out "$OUT" \
  --max-model-len 16384 --prefill-tokens 8192 --decode-tokens 128
