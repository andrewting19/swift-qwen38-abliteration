#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RUN_DIRECTORY DIRECTION_FILE DIRECTION_REPORT CANDIDATE_KEY" >&2
  exit 2
fi

scope_run_dir=$1
scope_direction_file=$2
scope_direction_report=$3
scope_candidate_key=$4
scope_ack=I_UNDERSTAND_THIS_LOADS_AND_EDITS_A_55GB_MODEL

scope_configs=(
  configs/orca_style_no_embedding.toml
  configs/late_half_no_embedding.toml
  configs/late_40_63_no_embedding.toml
  configs/late_48_63_no_embedding.toml
  configs/late_52_63_no_embedding.toml
  configs/band_32_39_no_embedding.toml
  configs/band_32_47_no_embedding.toml
  configs/band_32_55_no_embedding.toml
  configs/huihui_band.toml
  configs/orca_style_mixer_only.toml
  configs/orca_style_mlp_only.toml
)

mkdir -p "$scope_run_dir/scope" "$scope_run_dir/logs"
for scope_config in "${scope_configs[@]}"; do
  scope_name=$(basename "$scope_config" .toml)
  scope_output="$scope_run_dir/scope/$scope_name"
  if [[ -f "$scope_output/${scope_name}_report.json" ]]; then
    echo "Resume: $scope_name is already complete."
    continue
  fi
  python scripts/screen_generation_subspaces.py \
    --config "$scope_config" \
    --candidate-file "$scope_direction_file" \
    --candidate-report "$scope_direction_report" \
    --candidate-key "$scope_candidate_key" \
    --screen-name "$scope_name" \
    --output-dir "$scope_output" \
    --harmful-limit 16 \
    --safe-limit 16 \
    --max-new-tokens 48 \
    --batch-size 8 \
    --sufficient-removal 0.75 \
    --acknowledge "$scope_ack" \
    2>&1 | tee "$scope_run_dir/logs/scope-${scope_name}.log"
done
