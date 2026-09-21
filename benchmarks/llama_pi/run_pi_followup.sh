#!/usr/bin/env bash
set -euo pipefail

root=${1:-/workspace/swift-abliterated-q3}
repo=${TASK_REPO:-/workspace/pi-agent-task}
session=${PI_SESSION:?set PI_SESSION to the session JSONL path}
results=$root/results/pi-agent-followup
server_pid=$results/server.pid

mkdir -p "$results"
"$root/run_llama_server.sh" dflash "$root" > "$results/server.log" 2>&1 &
echo $! > "$server_pid"

cleanup() {
  if [[ -s "$server_pid" ]]; then
    pid=$(cat "$server_pid")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      wait "$pid" || true
    fi
  fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 240); do
  if curl -fsS --max-time 2 http://127.0.0.1:17070/health >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo "server did not become ready" >&2; exit 1; }

export PI_CODING_AGENT_DIR=$root/pi-config
export PI_OFFLINE=1
cd "$repo"
pi \
  --provider local-swift \
  --model swift-abliterated-q3 \
  --thinking medium \
  --session "$session" \
  --no-extensions \
  --no-skills \
  --no-context-files \
  --mode json \
  -p "A hidden contract test found one remaining failure. For a permanent HTTP 401, _request_json raises 'model request failed at https://model.test/v1/models: HTTP Error 401: failure'. The task requires every final failure message to include the request URL and attempt count, including a non-retryable failure after one attempt. Fix this without retrying the 401. Add or update a focused test for this exact case, run the focused tests and the complete Python suite, and report the exact results." \
  > "$results/pi.jsonl" \
  2> "$results/pi.stderr"

git status --short > "$results/git-status.txt"
git diff --stat > "$results/git-diff-stat.txt"
git diff > "$results/git-diff.patch"
echo "Pi follow-up completed: $results"
