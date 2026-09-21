#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_pi_replay.sh ar|mtp|dflash [root]}
root=${2:-/workspace/swift-abliterated-q3}
results=$root/results/$mode
server_log=$results/server.log
server_pid=$results/server.pid
session=$results/session.jsonl

mkdir -p "$results" "$root/pi-config"
cp "$root/models.json" "$root/pi-config/models.json"
cp "$root/replay-source.jsonl" "$session"

"$root/run_llama_server.sh" "$mode" "$root" > "$server_log" 2>&1 &
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

for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 http://127.0.0.1:17070/health >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:17070/health > "$results/health.json"
curl -fsS http://127.0.0.1:17070/metrics > "$results/metrics-before.txt"

export PI_CODING_AGENT_DIR=$root/pi-config
export PI_OFFLINE=1
start=$(date +%s.%N)
pi \
  --provider local-swift \
  --model swift-abliterated-q3 \
  --thinking medium \
  --session "$session" \
  --no-tools \
  --no-extensions \
  --no-skills \
  --no-context-files \
  --mode json \
  -p \
  "Using the completed implementation work and test evidence in this session, write a detailed technical handoff of about 700 words. Explain the problem, the implemented behavior, important edge cases, changed files, exact verification results, and any unrelated remaining failures. Do not call tools. End with the exact line HANDOFF_COMPLETE." \
  > "$results/pi.jsonl" \
  2> "$results/pi.stderr"
end=$(date +%s.%N)

curl -fsS http://127.0.0.1:17070/metrics > "$results/metrics-after.txt"
python3 - "$start" "$end" > "$results/wall-seconds.txt" <<'PY'
import sys
print(float(sys.argv[2]) - float(sys.argv[1]))
PY

grep -q 'HANDOFF_COMPLETE' "$results/pi.jsonl"
echo "$mode replay completed"
