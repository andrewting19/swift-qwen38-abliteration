#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RUN_DIRECTORY DIRECTION_FILE DIRECTION_REPORT CANDIDATE_KEY" >&2
  exit 2
fi

attention_type_run_dir=$1
attention_type_direction_file=$2
attention_type_direction_report=$3
attention_type_candidate_key=$4
attention_type_ack=I_UNDERSTAND_THIS_LOADS_AND_EDITS_A_55GB_MODEL
attention_type_mlp_args=()
attention_type_full_args=()
attention_type_linear_args=()

for attention_type_layer in $(seq 32 63); do
  attention_type_mlp_args+=(--mlp-layer "$attention_type_layer")
  if (( attention_type_layer % 4 == 3 )); then
    attention_type_full_args+=(--attention-layer "$attention_type_layer")
  else
    attention_type_linear_args+=(--attention-layer "$attention_type_layer")
  fi
done

run_attention_type() {
  local attention_type_name=$1
  shift
  local attention_type_output="$attention_type_run_dir/attention-type/$attention_type_name"
  python scripts/screen_generation_subspaces.py \
    --config configs/orca_style_no_embedding.toml \
    --candidate-file "$attention_type_direction_file" \
    --candidate-report "$attention_type_direction_report" \
    --candidate-key "$attention_type_candidate_key" \
    --screen-name "$attention_type_name" \
    --output-dir "$attention_type_output" \
    "${attention_type_mlp_args[@]}" \
    "$@" \
    --harmful-limit 16 \
    --safe-limit 16 \
    --max-new-tokens 48 \
    --batch-size 8 \
    --sufficient-removal 0.75 \
    --acknowledge "$attention_type_ack" \
    2>&1 | tee "$attention_type_run_dir/logs/${attention_type_name}.log"
}

run_attention_type late_half_mlp_with_full_attention_only \
  "${attention_type_full_args[@]}"
run_attention_type late_half_mlp_with_linear_attention_only \
  "${attention_type_linear_args[@]}"
