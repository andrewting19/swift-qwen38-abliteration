#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RUN_DIRECTORY DIRECTION_FILE DIRECTION_REPORT CANDIDATE_KEY" >&2
  exit 2
fi

review_run_dir=$1
review_direction_file=$2
review_direction_report=$3
review_candidate_key=$4
review_name=exact_full_rank1_kl
review_output="$review_run_dir/safe-kl-review/$review_name"
review_ack=I_UNDERSTAND_THIS_LOADS_AND_EDITS_A_55GB_MODEL

python scripts/screen_generation_subspaces.py \
  --config configs/orca_style_full.toml \
  --candidate-file "$review_direction_file" \
  --candidate-report "$review_direction_report" \
  --candidate-key "$review_candidate_key" \
  --screen-name "$review_name" \
  --output-dir "$review_output" \
  --harmful-limit 16 \
  --safe-limit 16 \
  --max-new-tokens 48 \
  --batch-size 8 \
  --sufficient-removal 0.75 \
  --acknowledge "$review_ack" \
  2>&1 | tee "$review_run_dir/logs/${review_name}.log"
