#!/usr/bin/env bash
# usage: serve_eval.sh <model-dir> <port> <tag> [KEY=VALUE ...]
set -uo pipefail
MODEL="$1"; PORT="$2"; TAG="$3"; shift 3
source /scratch/w16/env.sh
export PYTHONPATH=/scratch/w16/shadow:/scratch/w16
for kv in "$@"; do export "$kv"; done
exec /nix/store/bfy6kchprah5ywrj2s9c6fqslfcp9vkm-python3-3.14.7-env/bin/vllm serve "$MODEL" \
  --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enforce-eager --attention-backend R4D --enable-prefix-caching --mamba-cache-mode align \
  --max-num-batched-tokens 4096 --max-num-seqs 16 --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$TAG"
