#!/usr/bin/env bash
set -euo pipefail

root=${1:-/workspace/swift-abliterated-q3}
repo=${TASK_REPO:-/workspace/pi-agent-task}
session=${PI_SESSION:-$root/long-context/vast-real-pi-94k-20260822.jsonl}
results=$root/results/pi-long-gate
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
  -p "Review the current uncommitted retry implementation in /workspace/pi-agent-task. Use read-only shell and file tools to inspect the diff and run the focused retry tests. Do not edit files. Then write a detailed technical review of about 900 words. Cover correctness, edge cases, test evidence, and any remaining risk. End with the exact line LONG_PI_GATE_COMPLETE." \
  > "$results/pi.jsonl" \
  2> "$results/pi.stderr"

echo "Pi long-context gate completed: $results"
