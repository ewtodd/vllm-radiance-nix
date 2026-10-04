#!/usr/bin/env bash
# usage: run_serve_logits.sh <model-dir> <out.pt> [KEY=VALUE ...]
set -uo pipefail
MODEL="$1"; OUT="$2"; shift 2
source /scratch/w16/env.sh
# Model Runner V2 rejects custom logits processors; the capture uses V1. Model forward numerics
# are runner-independent, and the online serve tests run V2 where it matters.
export VLLM_USE_V2_MODEL_RUNNER=0
if [ "${RADIANCE_TEST_SHADOW:-1}" = "1" ]; then
  export PYTHONPATH=/scratch/w16/shadow:/scratch/w16
else
  # Stock regression arm: same scripts, the deployed site-packages radiance modules.
  export PYTHONPATH=/scratch/w16
fi
export RADIANCE_LOGITS_OUT="$OUT"
for kv in "$@"; do export "$kv"; done
exec /nix/store/bfy6kchprah5ywrj2s9c6fqslfcp9vkm-python3-3.14.7-env/bin/python \
  /scratch/w16/logits_serve.py --model "$MODEL" --attention-backend R4D
