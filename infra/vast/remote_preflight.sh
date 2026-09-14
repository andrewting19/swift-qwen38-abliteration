#!/usr/bin/env bash
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version
df -BG /workspace 2>/dev/null || df -BG .
python3 - <<'PY'
import os
import shutil
import subprocess
free = shutil.disk_usage('/workspace' if __import__('pathlib').Path('/workspace').exists() else '.').free
if free < 250 * 1024**3:
    raise SystemExit('Need at least 250 GiB free disk before model download.')
memory_mb = subprocess.run(
    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
    check=True,
    capture_output=True,
    text=True,
).stdout
largest_mb = max(int(line.strip()) for line in memory_mb.splitlines() if line.strip())
if largest_mb < 75 * 1024:
    raise SystemExit('Need at least 75 GiB on one GPU.')
ram_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
if ram_bytes < 32 * 1024**3:
    raise SystemExit('Need at least 32 GiB of host RAM for shard editing.')
print('REMOTE_HARDWARE_PREFLIGHT_OK')
PY
