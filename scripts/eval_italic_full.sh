#!/bin/bash
# Full 10k ITALIC eval via the official harness (run_eval.py + vLLM), both modes.
# Usage: eval_italic_full.sh <model_path_or_id> <served_name> [max_model_len]
set -euo pipefail

MODEL="$1"; NAME="$2"; MAXLEN="${3:-4096}"
PY=~/ai/vllm-cu129/bin
ITALIC=~/ai/ITALIC
PORT=8000

cd "$ITALIC"
echo "=== serving $MODEL as '$NAME' ==="
$PY/vllm serve "$MODEL" --served-model-name "$NAME" --port $PORT \
  --max-model-len "$MAXLEN" --gpu-memory-utilization 0.85 \
  > "serve_${NAME}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# wait for readiness (max ~5 min)
for i in $(seq 1 150); do
  if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$NAME"; then
    echo "server ready after ${i}0s"; break
  fi
  sleep 2
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "SERVER DIED — tail of log:"; tail -20 "serve_${NAME}.log"; exit 1
  fi
done

for FAST in true false; do
  MODE=$([ "$FAST" = true ] && echo fast || echo slow)
  echo "=== eval $NAME mode=$MODE ==="
  $PY/python run_eval.py --config-name config \
    model="$NAME" fast=$FAST num_threads=64 \
    provider_kwargs.base_url="http://localhost:$PORT/v1" \
    2>&1 | grep -iE "Metrics|accuracy|error" | tail -5
done

kill $SERVER_PID 2>/dev/null || true
sleep 5
echo "=== DONE $NAME ==="
