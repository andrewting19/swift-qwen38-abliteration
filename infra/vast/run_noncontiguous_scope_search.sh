#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RUN_DIRECTORY DIRECTION_FILE DIRECTION_REPORT CANDIDATE_KEY" >&2
  exit 2
fi

noncontiguous_run_dir=$1
noncontiguous_direction_file=$2
noncontiguous_direction_report=$3
noncontiguous_candidate_key=$4
noncontiguous_ack=I_UNDERSTAND_THIS_LOADS_AND_EDITS_A_55GB_MODEL

run_layer_set() {
  local noncontiguous_name=$1
  shift
  local noncontiguous_output="$noncontiguous_run_dir/noncontiguous/$noncontiguous_name"
  if [[ -f "$noncontiguous_output/${noncontiguous_name}_report.json" ]]; then
    echo "Resume: $noncontiguous_name is already complete."
    return
  fi
  local noncontiguous_layer_args=()
  for noncontiguous_layer in "$@"; do
    noncontiguous_layer_args+=(--target-layer "$noncontiguous_layer")
  done
  python scripts/screen_generation_subspaces.py \
    --config configs/orca_style_no_embedding.toml \
    --candidate-file "$noncontiguous_direction_file" \
    --candidate-report "$noncontiguous_direction_report" \
    --candidate-key "$noncontiguous_candidate_key" \
    --screen-name "$noncontiguous_name" \
    --output-dir "$noncontiguous_output" \
    "${noncontiguous_layer_args[@]}" \
    --harmful-limit 16 \
    --safe-limit 16 \
    --max-new-tokens 48 \
    --batch-size 8 \
    --sufficient-removal 0.75 \
    --acknowledge "$noncontiguous_ack" \
    2>&1 | tee "$noncontiguous_run_dir/logs/${noncontiguous_name}.log"
}

noncontiguous_32_39=($(seq 32 39))
noncontiguous_48_55=($(seq 48 55))
noncontiguous_56_63=($(seq 56 63))
noncontiguous_48_63=($(seq 48 63))

run_layer_set union_32_39_48_63 \
  "${noncontiguous_32_39[@]}" "${noncontiguous_48_63[@]}"
run_layer_set union_32_39_48_55 \
  "${noncontiguous_32_39[@]}" "${noncontiguous_48_55[@]}"
run_layer_set union_32_39_56_63 \
  "${noncontiguous_32_39[@]}" "${noncontiguous_56_63[@]}"
