#!/usr/bin/env bash
# usage: [BENCHES=all|raw|chat] run_check_source.sh <port> <tag> <weights> <activations> [limit]
set -uo pipefail
# HF_HOME for the datasets cache; the endpoint itself needs no radiance env.
source /scratch/w16/env.sh >/dev/null 2>&1 || true
PORT="$1"; TAG="$2"; WEIGHTS="$3"; ACTS="$4"; LIMIT="${5:-300}"
BENCHES="${BENCHES:-all}"
BIN=/scratch/w16/check-source-result/bin/check-source
OUT=/scratch/w16/eval
mkdir -p "$OUT"

run() {
  echo "[$(date +%H:%M:%S)] $TAG $*"
  "$BIN" "$@" || echo "[$(date +%H:%M:%S)] FAILED $TAG $*"
}

RAW=(--base-url "http://127.0.0.1:$PORT" --model "$TAG" --concurrency 8
     --weights "$WEIGHTS" --activations "$ACTS" --kv-cache fp16)
CHAT=(--base-url "http://127.0.0.1:$PORT" --model "$TAG" --concurrency 4
      --weights "$WEIGHTS" --activations "$ACTS" --kv-cache fp16)

if [ "$BENCHES" = "all" ] || [ "$BENCHES" = "raw" ]; then
  run gsm8k "${RAW[@]}" --mode thinking --limit "$LIMIT" --max-tokens 3072 \
    --fewshot-seed 1234 --out "$OUT/$TAG-gsm8k-thinking.json"
fi
if [ "$BENCHES" = "all" ] || [ "$BENCHES" = "raw" ] || [ "$BENCHES" = "nothink" ]; then
  run gsm8k "${RAW[@]}" --mode nothink --limit "$LIMIT" --max-tokens 1024 \
    --fewshot-seed 1234 --out "$OUT/$TAG-gsm8k-nothink.json"
fi
if [ "$BENCHES" = "all" ] || [ "$BENCHES" = "chat" ]; then
  run gsm8k-chat "${CHAT[@]}" --level medium --seed 1234 --limit 200 --max-tokens 12288 \
    --out "$OUT/$TAG-gsm8k-chat-medium-s1234.json"
fi
echo "[$(date +%H:%M:%S)] DONE $TAG ($BENCHES)"
