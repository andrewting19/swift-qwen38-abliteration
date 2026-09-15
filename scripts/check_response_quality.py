#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from swift_abliteration.response_quality import analyze_response, summarize_quality


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def analyze_file(path: Path) -> tuple[list[dict], dict]:
    rows = read_jsonl(path)
    results = []
    for row in rows:
        metrics = analyze_response(row["response"])
        results.append(
            {
                "id": row["id"],
                "response_sha256": hashlib.sha256(row["response"].encode()).hexdigest(),
                **metrics,
            }
        )
    return results, summarize_quality(results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure deterministic output failures without storing raw text."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_items, base_summary = analyze_file(args.base)
    candidate_items, candidate_summary = analyze_file(args.candidate)
    if [row["id"] for row in base_items] != [row["id"] for row in candidate_items]:
        raise ValueError("Base and candidate IDs or order differ.")

    payload = {
        "metric": "deterministic_response_quality",
        "stores_raw_text": False,
        "rules": {
            "very_short_output": "one or two lexical tokens",
            "severe_repetition": (
                "at least 32 lexical tokens and either unique 4-gram ratio below "
                "0.20 or repeated non-overlapping 4-16 token span coverage at least 0.50"
            ),
        },
        "base": {"summary": base_summary, "items": base_items},
        "candidate": {"summary": candidate_summary, "items": candidate_items},
        "gate": {
            "no_added_empty_outputs": (
                candidate_summary["empty_output_count"] <= base_summary["empty_output_count"]
            ),
            "no_added_severe_repetition": (
                candidate_summary["severe_repetition_count"]
                <= base_summary["severe_repetition_count"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"base": base_summary, "candidate": candidate_summary, "gate": payload["gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
