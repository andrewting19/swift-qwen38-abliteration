#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_llama_server.sh ar|mtp|dflash [root]}
root=${2:-/workspace/swift-abliterated-q3}
server=$root/runtime/build/bin/llama-server
export LD_LIBRARY_PATH="$root/runtime/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
target=$root/Swift-Qwen3.8-27B-Abliterated-UD-Q3_K_XL.gguf
draft_default=$root/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
if [[ ! -s "$draft_default" ]]; then
  draft_default=$root/Qwen3.8-27B-DFlash2-Q4_0.gguf
fi
draft=${DFLASH_MODEL:-$draft_default}
context=${CONTEXT_SIZE:-262144}
draft_n=${DRAFT_N_MAX:-4}
mtp_n=${MTP_N_MAX:-3}
batch_size=${BATCH_SIZE:-8192}
ubatch_size=${UBATCH_SIZE:-2048}
export GGML_CUDA_GRAPH_OPT=${GGML_CUDA_GRAPH_OPT:-1}

# Read the GGUFs in order before mmap starts. This avoids slow random page
# faults on some Vast overlay disks.
dd if="$target" of=/dev/null bs=64M status=none
if [[ "$mode" == dflash ]]; then
  dd if="$draft" of=/dev/null bs=64M status=none
fi

common=(
  --model "$target"
  --alias swift-abliterated-q3
  --host 127.0.0.1
  --port 17070
  --ctx-size "$context"
  --parallel 1
  --batch-size "$batch_size"
  --ubatch-size "$ubatch_size"
  --cache-type-k q4_0
  --cache-type-v q4_0
  --flash-attn on
  --gpu-layers all
  --fit off
  --threads 16
  --threads-batch 16
  --poll 50
  --ctx-checkpoints 4
  --cache-ram 0
  --cache-prompt
  --cache-reuse 0
  --no-context-shift
  --jinja
  --reasoning-format deepseek
  --reasoning on
  --reasoning-effort medium
  --backend-sampling
  --temperature 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --repeat-penalty 1.0
  --metrics
  --no-webui
  --timeout 7200
)

case "$mode" in
  ar)
    spec=(--spec-type none)
    ;;
  mtp)
    spec=(
      --spec-type draft-mtp
      --spec-draft-ngl all
      --spec-draft-type-k q4_0
      --spec-draft-type-v q4_0
      --spec-draft-n-max "$mtp_n"
    )
    ;;
  dflash)
    [[ -s "$draft" ]] || { echo "missing DFlash model: $draft" >&2; exit 1; }
    spec=(
      --spec-draft-model "$draft"
      --spec-type draft-dflash
      --spec-draft-ngl all
      --spec-draft-type-k q4_0
      --spec-draft-type-v q4_0
      --spec-draft-n-max "$draft_n"
      --poll-draft 1
    )
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

exec "$server" "${common[@]}" "${spec[@]}"
