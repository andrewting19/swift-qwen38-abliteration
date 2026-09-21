#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_vllm.sh ar|mtp1|mtp2|mtp3}
extra=()
case "$mode" in
  ar) ;;
  mtp1) extra=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}') ;;
  mtp2) extra=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}') ;;
  mtp3) extra=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}') ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

exec vllm serve andrewting/Swift-Qwen3.8-27B-Abliterated \
  --served-model-name swift-abliterated \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 \
  --language-model-only \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "${extra[@]}"
