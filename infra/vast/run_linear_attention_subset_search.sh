#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RUN_DIRECTORY DIRECTION_FILE DIRECTION_REPORT CANDIDATE_KEY" >&2
  exit 2
fi

subset_run_dir=$1
subset_direction_file=$2
subset_direction_report=$3
subset_candidate_key=$4
subset_ack=I_UNDERSTAND_THIS_LOADS_AND_EDITS_A_55GB_MODEL
subset_mlp_args=()
subset_early_linear_args=()
subset_late_linear_args=()

for subset_layer in $(seq 32 63); do
  subset_mlp_args+=(--mlp-layer "$subset_layer")
  if (( subset_layer % 4 != 3 )); then
    if (( subset_layer <= 47 )); then
      subset_early_linear_args+=(--attention-layer "$subset_layer")
    else
      subset_late_linear_args+=(--attention-layer "$subset_layer")
    fi
  fi
done

run_subset() {
  local subset_name=$1
  shift
  local subset_output="$subset_run_dir/linear-attention-subsets/$subset_name"
  if [[ -f "$subset_output/${subset_name}_report.json" ]]; then
    echo "Resume: $subset_name is already complete."
    return
  fi
  python scripts/screen_generation_subspaces.py \
    --config configs/orca_style_no_embedding.toml \
    --candidate-file "$subset_direction_file" \
    --candidate-report "$subset_direction_report" \
    --candidate-key "$subset_candidate_key" \
    --screen-name "$subset_name" \
    --output-dir "$subset_output" \
    "${subset_mlp_args[@]}" \
    "$@" \
    --harmful-limit 16 \
    --safe-limit 16 \
    --max-new-tokens 48 \
    --batch-size 8 \
    --sufficient-removal 0.75 \
    --acknowledge "$subset_ack" \
    2>&1 | tee "$subset_run_dir/logs/${subset_name}.log"
}

run_subset late_half_mlp_with_early_linear_attention \
  "${subset_early_linear_args[@]}"
run_subset late_half_mlp_with_late_linear_attention \
  "${subset_late_linear_args[@]}"
