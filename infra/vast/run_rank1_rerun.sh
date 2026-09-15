#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_DIRECTORY" >&2
  exit 2
fi

rank1_run_dir=$1
rank1_ack=I_UNDERSTAND_THIS_LOADS_AND_EDITS_A_55GB_MODEL
mkdir -p "$rank1_run_dir/logs"

export PYTHONPATH=src:.

python scripts/capture_rank1_search.py \
  --output-dir "$rank1_run_dir/capture" \
  --batch-size 4 \
  --resume \
  --acknowledge "$rank1_ack" \
  2>&1 | tee "$rank1_run_dir/logs/capture.log"

if [[ ! -f "$rank1_run_dir/directions/direction_report.json" ]]; then
  python scripts/build_filtered_rank1_directions.py \
    --capture-dir "$rank1_run_dir/capture" \
    --output-dir "$rank1_run_dir/directions" \
    2>&1 | tee "$rank1_run_dir/logs/build-directions.log"
fi

set +e
python scripts/screen_rank1_proxy.py \
  --capture-dir "$rank1_run_dir/capture" \
  --direction-dir "$rank1_run_dir/directions" \
  --output-dir "$rank1_run_dir/proxy" \
  --pilot-output-dir "$rank1_run_dir/pilot" \
  --limit-per-group 16 \
  --finalist-count 4 \
  --pilot-max-new-tokens 32 \
  --batch-size 8 \
  --acknowledge "$rank1_ack" \
  2>&1 | tee "$rank1_run_dir/logs/proxy-and-pilot.log"
rank1_proxy_exit=${PIPESTATUS[0]}
set -e
if [[ $rank1_proxy_exit -eq 2 ]]; then
  echo "No rank-1 candidate passed the proxy gates. Stop before full generation."
  exit 0
fi
if [[ $rank1_proxy_exit -ne 0 ]]; then
  exit "$rank1_proxy_exit"
fi

python scripts/judge_wildguard_tree.py \
  --responses-root "$rank1_run_dir/pilot" \
  --output-root "$rank1_run_dir/pilot-wildguard" \
  --batch-size 16 \
  2>&1 | tee "$rank1_run_dir/logs/wildguard.log"

python scripts/summarize_rank1_pilot.py \
  --proxy-report "$rank1_run_dir/proxy/proxy_report.json" \
  --responses-root "$rank1_run_dir/pilot" \
  --judgments-root "$rank1_run_dir/pilot-wildguard" \
  --output "$rank1_run_dir/rank1-pilot-summary.json" \
  2>&1 | tee "$rank1_run_dir/logs/summary.log"

find "$rank1_run_dir" -type f ! -name artifact-manifest.sha256 -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  > "$rank1_run_dir/artifact-manifest.sha256"
