#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from swift_abliteration.metrics import coherence_means, refusal_rate

COHERENCE_DIMENSIONS = [
    "readability",
    "relevance",
    "consistency",
    "non_repetition",
    "completeness",
]


def read_jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["refusal", "coherence"], required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = read_jsonl(args.base)
    candidate = read_jsonl(args.candidate)
    if len(base) != len(candidate):
        raise ValueError("Base and candidate judgment counts differ.")
    if [item["id"] for item in base] != [item["id"] for item in candidate]:
        raise ValueError("Base and candidate judgment IDs or order differ.")
    if args.mode == "refusal":
        base_rate = refusal_rate(base)
        candidate_rate = refusal_rate(candidate)
        report = {
            "mode": args.mode,
            "count": len(base),
            "base_refusal_rate": base_rate,
            "candidate_refusal_rate": candidate_rate,
            "absolute_refusal_rate_change": candidate_rate - base_rate,
            "removed_fraction_of_base_refusals": (
                (base_rate - candidate_rate) / base_rate if base_rate else None
            ),
        }
    else:
        base_means = coherence_means(base, COHERENCE_DIMENSIONS)
        candidate_means = coherence_means(candidate, COHERENCE_DIMENSIONS)
        report = {
            "mode": args.mode,
            "count": len(base),
            "base_means": base_means,
            "candidate_means": candidate_means,
            "candidate_minus_base": {
                name: candidate_means[name] - base_means[name]
                for name in COHERENCE_DIMENSIONS
            },
        }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
