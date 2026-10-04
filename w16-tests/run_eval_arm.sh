#!/usr/bin/env bash
# usage: run_eval_arm.sh <model-dir> <port> <tag> <weights-label> <activations-label> [KEY=VALUE ...]
# Starts the eval serve, waits for health, runs gsm8k thinking/nothink (300) and gsm8k-chat
# medium (200), then stops the serve. Runs under the campaign protocol (V1 runner, eager).
set -uo pipefail
MODEL="$1"; PORT="$2"; TAG="$3"; WEIGHTS="$4"; ACTS="$5"; shift 5
mkdir -p /scratch/w16/eval

bash /scratch/w16/serve_eval.sh "$MODEL" "$PORT" "$TAG" VLLM_USE_V2_MODEL_RUNNER=0 "$@" \
  > "/scratch/w16/eval/serve_$TAG.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null' EXIT

ok=0
for i in $(seq 1 120); do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then break; fi
  code=$(curl -s -m 2 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health")
  if [ "$code" = "200" ]; then ok=1; break; fi
  sleep 5
done
if [ "$ok" != "1" ]; then
  echo "SERVE FAILED for $TAG; tail:"
  tail -20 "/scratch/w16/eval/serve_$TAG.log"
  exit 1
fi
echo "serve healthy after ~$((i * 5))s, running benches"
bash /scratch/w16/run_check_source.sh "$PORT" "$TAG" "$WEIGHTS" "$ACTS" 300
