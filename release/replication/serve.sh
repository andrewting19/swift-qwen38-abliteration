#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-$PWD/swift-qwen38-runtime}
source "$script_dir/runtime.env"

server=$root/runtime/build/bin/llama-server
target=$root/Swift-Qwen3.8-27B-Abliterated-UD-Q3_K_XL.gguf
draft=$root/Qwen3.8-27B-DFlash2-Q4_K_M.gguf

for required in "$server" "$target" "$draft"; do
  [[ -s "$required" ]] || { echo "missing required file: $required" >&2; exit 1; }
done

export LD_LIBRARY_PATH="$root/runtime/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_CUDA_GRAPH_OPT=1

# Sequential reads prevent slow random mmap faults on some rented hosts.
dd if="$target" of=/dev/null bs=64M status=none
dd if="$draft" of=/dev/null bs=64M status=none

exec "$server" \
  --model "$target" \
  --alias swift-abliterated-q3 \
  --spec-draft-model "$draft" \
  --spec-type draft-dflash \
  --spec-draft-ngl all \
  --spec-draft-type-k "$SWIFT_DRAFT_KV" \
  --spec-draft-type-v "$SWIFT_DRAFT_KV" \
  --spec-draft-n-max "$SWIFT_DRAFT_N_MAX" \
  --host "$SWIFT_HOST" \
  --port "$SWIFT_PORT" \
  --ctx-size "$SWIFT_CONTEXT_SIZE" \
  --parallel 1 \
  --batch-size "$SWIFT_BATCH_SIZE" \
  --ubatch-size "$SWIFT_UBATCH_SIZE" \
  --cache-type-k "$SWIFT_TARGET_KV" \
  --cache-type-v "$SWIFT_TARGET_KV" \
  --flash-attn on \
  --gpu-layers all \
  --fit off \
  --threads "$SWIFT_THREADS" \
  --threads-batch "$SWIFT_THREADS" \
  --poll 50 \
  --poll-draft 1 \
  --ctx-checkpoints 4 \
  --cache-ram 0 \
  --cache-prompt \
  --cache-reuse 0 \
  --no-context-shift \
  --jinja \
  --reasoning-format deepseek \
  --reasoning on \
  --reasoning-effort medium \
  --backend-sampling \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --repeat-penalty 1.0 \
  --metrics \
  --no-webui \
  --timeout 7200
