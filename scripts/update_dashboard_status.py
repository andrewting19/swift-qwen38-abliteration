#!/usr/bin/env python3
"""Refresh local dashboard status using safe aggregates only."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "runs/gpu/20260914-a100-51065040/status.json"
DEFAULT_CONFIG = ROOT / ".dashboard-runtime.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def run(args: list[str], timeout: float = 15) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return completed.returncode, completed.stdout, completed.stderr


def remote_command(host: str, command: str, identity: str | None, timeout: float, port: int | None = None) -> tuple[int, str, str]:
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    if identity:
        ssh += ["-o", "IdentitiesOnly=yes", "-i", identity]
    if port:
        ssh += ["-p", str(port)]
    return run(ssh + [host, command], timeout)


def count_remote_jsonl(host: str | None, paths: list[str], identity: str | None, port: int | None = None) -> dict[str, int]:
    if not host or not paths:
        return {}
    # Missing future output files are normal. Count each file only after it exists.
    command = "for p in " + " ".join(shlex.quote(p) for p in paths) + '; do [ -f "$p" ] && wc -l -- "$p"; done; true'
    code, stdout, _ = remote_command(host, command, identity, 20, port)
    if code:
        return {}
    counts: dict[str, int] = {}
    for line in stdout.splitlines():
        match = re.match(r"^\s*(\d+)\s+(.+?)\s*$", line)
        if match:
            # Preserve the safe path key. Basename-only keys collide across arms.
            counts[match.group(2)] = int(match.group(1))
    return counts


def remote_runtime(host: str | None, identity: str | None, process_pattern: str | None, port: int | None = None) -> dict[str, Any]:
    if not host:
        return {}
    process_active = None
    if process_pattern:
        # The bracket expression prevents grep from matching its own command.
        escaped = re.escape(process_pattern)
        check = f"ps -eo args= | grep -E -- '[{escaped[0]}]{escaped[1:]}' >/dev/null"
        code, _, _ = remote_command(host, check, identity, 10, port)
        process_active = code == 0
    code, stdout, _ = remote_command(host, "nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits", identity, 12, port)
    runtime: dict[str, Any] = {"process_active": process_active} if process_active is not None else {}
    if code == 0 and stdout.strip():
        parts = [part.strip() for part in stdout.splitlines()[0].split(",")]
        if len(parts) >= 3:
            runtime.update(gpu=parts[0], memory_used_mib=int(float(parts[1])), utilization_percent=int(float(parts[2])))
    return runtime


def vast_credit(enabled: bool) -> float | None:
    if not enabled:
        return None
    code, stdout, _ = run(["uvx", "--from", "vastai", "vastai", "show", "user", "--raw"], timeout=30)
    if code:
        return None
    try:
        value = json.loads(stdout).get("credit")
        return float(value) if value is not None else None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def refresh(status_path: Path, config: dict[str, Any], args: argparse.Namespace) -> None:
    snapshot = json.loads(status_path.read_text()) if status_path.exists() else {}
    overlay: dict[str, Any] = {}
    overlay_value = config.get("overlay_file")
    if overlay_value:
        overlay_path = Path(overlay_value)
        if not overlay_path.is_absolute():
            overlay_path = ROOT / overlay_path
        overlay = load_config(overlay_path)
    host = args.remote_host or config.get("remote_host")
    identity = args.identity or config.get("identity_file")
    port = args.ssh_port or config.get("ssh_port")
    paths = args.remote_jsonl or config.get("remote_jsonl", [])
    if isinstance(paths, str):
        paths = [paths]
    candidate_paths = args.candidate_jsonl or config.get("candidate_jsonl", [])
    if isinstance(candidate_paths, str):
        candidate_paths = [candidate_paths]
    process_pattern = args.process_pattern or config.get("process_pattern")
    counts = count_remote_jsonl(host, paths, identity, port)
    runtime = remote_runtime(host, identity, process_pattern, port)
    expected = int(args.expected_outputs or config.get("expected_outputs", 768))
    # When a subset is configured, count only those output paths toward the gate.
    observed = sum(counts.get(path, 0) for path in candidate_paths) if candidate_paths else sum(counts.values())
    arm_counts: dict[str, int] = {}
    for path, count in counts.items():
        arm = Path(path).parent.name
        arm_counts[arm] = arm_counts.get(arm, 0) + count
    candidate_started = any(arm != "base" and count > 0 for arm, count in arm_counts.items())
    candidate_names = [item.get("name") for item in snapshot.get("direction_quality", {}).get("candidates", []) if item.get("name")]
    arm_order = ["base", *candidate_names]
    per_arm_expected = max(1, expected // max(len(arm_order), 1))
    active_arm = next((arm for arm in arm_order if arm_counts.get(arm, 0) < per_arm_expected), None)
    candidate_states = []
    for item in snapshot.get("direction_quality", {}).get("candidates", []):
        updated = dict(item)
        name = str(item.get("name", ""))
        count = arm_counts.get(name, 0)
        updated["state"] = "done" if count >= per_arm_expected else "running" if name == active_arm else "queued"
        candidate_states.append(updated)
    if active_arm == "base":
        screen_status = f"base arm · {arm_counts.get('base', 0)}/{per_arm_expected}"
    elif active_arm:
        screen_status = f"{active_arm} · {arm_counts.get(active_arm, 0)}/{per_arm_expected}"
    else:
        screen_status = "response generation complete"
    now = utc_now()
    patch: dict[str, Any] = {
        "updated_at": now,
        "direction_quality": {"status": screen_status, "candidates": candidate_states},
        "runtime": {
            "jsonl_line_counts": counts,
            "arm_output_counts": arm_counts,
            "expected_outputs": expected,
            "observed_outputs": observed,
            "candidate_outputs_started": candidate_started,
            **runtime,
        },
    }
    if counts:
        patch["progress"] = {"phase_detail": f"Reversible screen: {observed}/{expected} output records observed (line counts may lag buffering)", "percent": min(99, round(55 + 35 * observed / max(expected, 1), 1))}
    credit = vast_credit(args.read_vast_credit or bool(config.get("read_vast_credit")))
    if credit is not None:
        patch["credit"] = {"balance_usd": credit, "note": "Live Vast.ai credit read at last refresh"}
    if runtime.get("gpu"):
        patch["rental"] = {"gpu": runtime["gpu"], "gpu_memory_used_mib": runtime.get("memory_used_mib"), "gpu_utilization_percent": runtime.get("utilization_percent")}
    started_at = parse_utc(snapshot.get("rental", {}).get("started_at"))
    hourly_rate = snapshot.get("rental", {}).get("hourly_rate_usd")
    if started_at and hourly_rate is not None:
        elapsed_hours = max(0.0, (parse_utc(now) - started_at).total_seconds() / 3600)
        patch.setdefault("rental", {}).update(
            elapsed_hours=round(elapsed_hours, 3),
            estimated_cost_usd=round(elapsed_hours * float(hourly_rate), 3),
        )
    if runtime.get("process_active") is False and process_pattern and observed < expected:
        patch["error"] = "Screen process is not active before all expected output records were written."
    else:
        patch["error"] = None
    if observed >= expected and runtime.get("process_active") is False:
        patch["run"] = {"status": "screen complete"}
        patch["progress"] = {
            "phase": "offline_scoring",
            "phase_label": "Offline scoring",
            "phase_detail": "All reversible response records are complete. Scoring is ready.",
            "label": "Reversible screen complete",
            "percent": 90,
            "steps": [
                {"label": "Preflight", "state": "done"},
                {"label": "GPU capture", "state": "done"},
                {"label": "Direction analysis", "state": "done"},
                {"label": "Reversible screen", "state": "done"},
                {"label": "Final evaluation", "state": "active"},
            ],
        }
    merged = deep_merge(snapshot, patch)
    # The operator-written overlay is authoritative for experiment decisions and
    # labels. Runtime polling still supplies counts, credit, and GPU telemetry.
    merged = deep_merge(merged, overlay)
    # These maps are complete snapshots. Do not retain keys from an older poll.
    merged["runtime"]["jsonl_line_counts"] = counts
    merged["runtime"]["arm_output_counts"] = arm_counts
    write_atomic(status_path, merged)
    print(json.dumps({"status": str(status_path), "updated_at": patch["updated_at"], "counts": counts, "runtime": runtime, "credit_updated": credit is not None}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--remote-host")
    parser.add_argument("--identity")
    parser.add_argument("--ssh-port", type=int)
    parser.add_argument("--remote-jsonl", action="append", default=[])
    parser.add_argument("--candidate-jsonl", action="append", default=[])
    parser.add_argument("--process-pattern")
    parser.add_argument("--expected-outputs", type=int)
    parser.add_argument("--read-vast-credit", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    while True:
        refresh(args.status, config, args)
        if not args.watch:
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
