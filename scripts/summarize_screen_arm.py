#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from swift_abliteration.metrics import coherence_means

GROUPS = (
    "standard_harmful",
    "matched_harmful",
    "standard_harmless",
    "matched_harmless",
)
REFUSAL = {"hard_refusal", "soft_refusal"}
COHERENCE_DIMENSIONS = (
    "readability",
    "relevance",
    "consistency",
    "non_repetition",
    "completeness",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write safe aggregate metrics for one reversible screen arm."
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--judgments-first", type=Path, required=True)
    parser.add_argument("--judgments-second", type=Path, required=True)
    parser.add_argument("--judgments-final", type=Path, required=True)
    parser.add_argument("--kl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, object] = {
        "arm": args.arm,
        "groups": {},
        "coherence": {},
        "kl": json.loads(args.kl.read_text(encoding="utf-8")),
    }
    for group in GROUPS:
        responses = read_jsonl(args.responses_dir / f"{group}.jsonl")
        first = read_jsonl(args.judgments_first / f"{group}_refusal.jsonl")
        second = read_jsonl(args.judgments_second / f"{group}_refusal.jsonl")
        final_path = args.judgments_final / f"{group}_refusal.jsonl"
        final = read_jsonl(final_path)
        ids = [row["id"] for row in responses]
        if ids != [row["id"] for row in first] or ids != [row["id"] for row in second] or ids != [row["id"] for row in final]:
            raise ValueError(f"IDs or order differ for {group}.")
        labels = Counter(row["judgment"]["label"] for row in final)
        refusal_count = sum(labels[label] for label in REFUSAL)
        agreements = sum(
            (left["judgment"]["label"] in REFUSAL)
            == (right["judgment"]["label"] in REFUSAL)
            for left, right in zip(first, second, strict=True)
        )
        result["groups"][group] = {
            "count": len(final),
            "labels": dict(sorted(labels.items())),
            "refusal_count": refusal_count,
            "refusal_rate": refusal_count / len(final),
            "empty_output_count": sum(
                not row["response"].strip() for row in responses
            ),
            "binary_judge_agreement": agreements / len(final),
            "final_judgment_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        }

    coherence_passes = []
    for path in (
        args.judgments_first / "standard_harmless_coherence.jsonl",
        args.judgments_second / "standard_harmless_coherence.jsonl",
    ):
        rows = read_jsonl(path)
        coherence_passes.append(coherence_means(rows, COHERENCE_DIMENSIONS))
    result["coherence"] = {
        "passes": coherence_passes,
        "mean": {
            dimension: sum(values[dimension] for values in coherence_passes)
            / len(coherence_passes)
            for dimension in COHERENCE_DIMENSIONS
        },
    }
    result["coherence"]["overall_mean"] = sum(
        result["coherence"]["mean"].values()
    ) / len(COHERENCE_DIMENSIONS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "arm": args.arm, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
