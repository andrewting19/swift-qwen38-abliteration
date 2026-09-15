#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RUN_DIRECTORY DIRECTION_FILE DIRECTION_REPORT CANDIDATE_KEY" >&2
  exit 2
fi

component_run_dir=$1
component_direction_file=$2
component_direction_report=$3
component_candidate_key=$4
component_ack=I_UNDERSTAND_THIS_LOADS_AND_EDITS_A_55GB_MODEL
component_layer_args=()
for component_layer in $(seq 32 63); do
  component_layer_args+=(--target-layer "$component_layer")
done

for component_attention_alpha in 0.25 0.50 0.75; do
  component_suffix=${component_attention_alpha/./_}
  component_name="late_half_mlp_1_attention_${component_suffix}"
  component_output="$component_run_dir/component-alpha/$component_name"
  if [[ -f "$component_output/${component_name}_report.json" ]]; then
    echo "Resume: $component_name is already complete."
    continue
  fi
  python scripts/screen_generation_subspaces.py \
    --config configs/orca_style_no_embedding.toml \
    --candidate-file "$component_direction_file" \
    --candidate-report "$component_direction_report" \
    --candidate-key "$component_candidate_key" \
    --screen-name "$component_name" \
    --output-dir "$component_output" \
    "${component_layer_args[@]}" \
    --alpha 1.0 \
    --attention-alpha "$component_attention_alpha" \
    --mlp-alpha 1.0 \
    --harmful-limit 16 \
    --safe-limit 16 \
    --max-new-tokens 48 \
    --batch-size 8 \
    --sufficient-removal 0.75 \
    --acknowledge "$component_ack" \
    2>&1 | tee "$component_run_dir/logs/${component_name}.log"
done
