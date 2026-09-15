#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RUN_DIRECTORY DIRECTION_FILE DIRECTION_REPORT CANDIDATE_KEY" >&2
  exit 2
fi

refine_run_dir=$1
refine_direction_file=$2
refine_direction_report=$3
refine_candidate_key=$4
refine_ack=I_UNDERSTAND_THIS_LOADS_AND_EDITS_A_55GB_MODEL

for refine_alpha in 0.99 0.98 0.995; do
  refine_suffix=${refine_alpha/./_}
  refine_name="full_rank1_alpha_${refine_suffix}"
  refine_output="$refine_run_dir/high-alpha/$refine_name"
  if [[ -f "$refine_output/${refine_name}_report.json" ]]; then
    echo "Resume: $refine_name is already complete."
    continue
  fi
  python scripts/screen_generation_subspaces.py \
    --config configs/orca_style_full.toml \
    --candidate-file "$refine_direction_file" \
    --candidate-report "$refine_direction_report" \
    --candidate-key "$refine_candidate_key" \
    --screen-name "$refine_name" \
    --output-dir "$refine_output" \
    --alpha "$refine_alpha" \
    --harmful-limit 16 \
    --safe-limit 16 \
    --max-new-tokens 48 \
    --batch-size 8 \
    --sufficient-removal 0.75 \
    --acknowledge "$refine_ack" \
    2>&1 | tee "$refine_run_dir/logs/${refine_name}.log"

  if python - "$refine_output/${refine_name}_report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
raise SystemExit(0 if report["sufficient_candidates"] else 1)
PY
  then
    echo "Stop: $refine_name passed the short-generation screen."
    break
  fi
done
