#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID to the served model name}"
: "${BASE_URL:?Set BASE_URL to the full endpoint, for example http://127.0.0.1:8000/v1/chat/completions}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

MODE=${MODE:-quick}
mkdir -p "$OUTPUT_DIR"

run_task() {
  local task=$1
  local limit=$2
  local extra=()
  if [[ "$limit" != "0" ]]; then
    extra+=(--limit "$limit")
  fi
  lm_eval run \
    --model local-chat-completions \
    --model_args "model=${MODEL_ID},base_url=${BASE_URL},num_concurrent=16,max_retries=3,tokenized_requests=false" \
    --tasks "$task" \
    --apply_chat_template \
    --system_instruction "You are a helpful assistant." \
    --gen_kwargs "temperature=0" \
    --log_samples \
    --output_path "$OUTPUT_DIR/$task" \
    "${extra[@]}"
}

if [[ "$MODE" == "quick" ]]; then
  run_task mmlu_pro 500
  run_task gsm8k_cot 256
  run_task ifeval 200
elif [[ "$MODE" == "final" ]]; then
  run_task mmlu_pro 0
  run_task gsm8k_cot 0
  run_task ifeval 0
else
  echo "MODE must be quick or final" >&2
  exit 2
fi
