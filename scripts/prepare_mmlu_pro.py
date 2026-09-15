#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a fixed MMLU-Pro subset without printing question text."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3819)
    parser.add_argument("--dataset", default="TIGER-Lab/MMLU-Pro")
    parser.add_argument(
        "--revision", default="b189ec765aa7ed75c8acfea42df31fdae71f97be"
    )
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive.")

    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset, revision=args.revision, split="test"
    )
    if args.limit > len(dataset):
        raise ValueError("limit exceeds the test split size.")
    generator = np.random.default_rng(args.seed)
    indices = np.sort(generator.choice(len(dataset), size=args.limit, replace=False))
    questions = []
    for index in indices:
        row = dataset[int(index)]
        questions.append(
            {
                "source_index": int(index),
                "question": row["question"],
                "choices": list(row["options"]),
                "answer": int(row["answer_index"]),
            }
        )
    payload = json.dumps({"questions": questions}, ensure_ascii=False, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")
    report = {
        "dataset": args.dataset,
        "revision": args.revision,
        "split": "test",
        "seed": args.seed,
        "count": len(questions),
        "output": str(args.output),
        "sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
