#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path


LINE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    args = parser.parse_args()

    with urllib.request.urlopen(args.url, timeout=30) as response:
        payload = response.read().decode("utf-8")

    metrics: dict[str, float] = defaultdict(float)
    for raw in payload.splitlines():
        match = LINE.match(raw)
        if not match:
            continue
        name = match.group("name")
        if name.startswith("vllm:"):
            metrics[name] += float(match.group("value"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(sorted(metrics.items())), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
