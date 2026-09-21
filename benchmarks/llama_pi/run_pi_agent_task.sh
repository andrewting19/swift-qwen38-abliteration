#!/usr/bin/env bash
set -euo pipefail

mode=${1:-dflash}
root=${2:-/workspace/swift-abliterated-q3}
repo=${TASK_REPO:-/workspace/pi-agent-task}
results=$root/results/pi-agent-$mode
server_pid=$results/server.pid

mkdir -p "$results" "$root/pi-config"
cp "$root/models.json" "$root/pi-config/models.json"

"$root/run_llama_server.sh" "$mode" "$root" > "$results/server.log" 2>&1 &
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

curl -fsS http://127.0.0.1:17070/metrics > "$results/metrics-before.txt"
export PI_CODING_AGENT_DIR=$root/pi-config
export PI_OFFLINE=1

cd "$repo"
pi \
  --provider local-swift \
  --model swift-abliterated-q3 \
  --thinking medium \
  --no-extensions \
  --no-skills \
  --no-context-files \
  --mode json \
  -p "$(cat "$root/TASK.md")" \
  > "$results/pi.jsonl" \
  2> "$results/pi.stderr"

curl -fsS http://127.0.0.1:17070/metrics > "$results/metrics-after.txt"
git status --short > "$results/git-status.txt"
git diff --stat > "$results/git-diff-stat.txt"
git diff > "$results/git-diff.patch"

set +e
.venv/bin/python -m unittest discover -s tests -v > "$results/python-tests.log" 2>&1
python_status=$?
set -e
printf 'python_tests=%s\n' "$python_status" > "$results/test-status.txt"

echo "$mode Pi agent task completed: $results"
