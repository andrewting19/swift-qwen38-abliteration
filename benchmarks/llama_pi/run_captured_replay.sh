#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_captured_replay.sh ar|mtp|dflash [tag] [root]}
tag=${2:-$mode}
root=${3:-/workspace/swift-abliterated-q3}
capture=${CAPTURE:-$root/long-context/captures-native114k-fixed/002-chat-completions.json}
max_tokens=${MAX_TOKENS:-2048}
results=$root/results/captured-$tag
server_log=$results/server.log
server_pid=$results/server.pid
api_key=$results/api-key

mkdir -p "$results"
printf '%s\n' local-benchmark > "$api_key"

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

ready=0
for _ in $(seq 1 240); do
  if curl -fsS --max-time 2 http://127.0.0.1:17070/health >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo "server did not become ready" >&2; exit 1; }

curl -fsS http://127.0.0.1:17070/health > "$results/health.json"
curl -fsS http://127.0.0.1:17070/metrics > "$results/metrics-before.txt"
python3 "$root/replay_captured_pi.py" "$capture" \
  --base-url http://127.0.0.1:17070 \
  --api-key-file "$api_key" \
  --max-tokens "$max_tokens" \
  | tee "$results/result.json"
curl -fsS http://127.0.0.1:17070/metrics > "$results/metrics-after.txt"

echo "$mode captured replay completed: $results"
