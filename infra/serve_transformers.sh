#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the pinned local base snapshot or edited checkpoint}"

PORT=${PORT:-8000}

transformers serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --device cuda:0 \
  --dtype bfloat16 \
  --reasoning off \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --default-seed 3819 \
  --log-level warning
