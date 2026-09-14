#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from urllib.request import urlopen


def download_verified(url: str, expected_hash: str, destination: Path) -> dict:
    with urlopen(url, timeout=60) as response:
        content = response.read()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"Hash mismatch for {destination.name}: {actual_hash}")
    parsed = json.loads(content)
    count = len(parsed["questions"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return {"path": str(destination), "sha256": actual_hash, "count": count}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="benchmarks/sources.toml")
    parser.add_argument("--output-dir", default="benchmarks/data")
    args = parser.parse_args()
    with Path(args.sources).open("rb") as handle:
        sources = tomllib.load(handle)
    spec = sources["cybermetric"]
    output = Path(args.output_dir)
    report = {
        "small": download_verified(
            spec["small_url"], spec["small_sha256"], output / "CyberMetric-80-v1.json"
        ),
        "final": download_verified(
            spec["final_url"], spec["final_sha256"], output / "CyberMetric-500-v1.json"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Do not print benchmark questions.
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
