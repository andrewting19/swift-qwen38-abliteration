#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3819)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.judgments).read_text().splitlines()
        if line.strip()
    ]
    uncertain = {
        row["id"]
        for row in rows
        if row["judgment"].get("confidence", 0) < args.confidence_threshold
    }
    ids = [row["id"] for row in rows]
    generator = random.Random(args.seed)
    sampled = set(generator.sample(ids, min(args.count, len(ids))))
    selected = sorted(uncertain | sampled)
    Path(args.output).write_text(
        json.dumps(
            {
                "seed": args.seed,
                "random_sample_count": min(args.count, len(ids)),
                "uncertain_count": len(uncertain),
                "selected_ids": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"selected_count": len(selected), "uncertain_count": len(uncertain)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
