#!/usr/bin/env python3
"""Build one rank-2 pair from complementary HarmBench rank-1 successes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.build_complementary_rank2 import orthonormal_pair
from swift_abliteration.gpu_support import sha256_file, write_json

GROUPS = ("standard_harmful", "matched_harmful")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank1-directions", type=Path, required=True)
    parser.add_argument("--harmbench-root", type=Path, required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-absolute-cosine", type=float, default=0.98)
    return parser.parse_args()


def successes(path: Path) -> set[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"HarmBench judgment file is empty: {path}")
    if any(row["judgment"].get("parse_error") for row in rows):
        raise ValueError(f"HarmBench judgment file has parse errors: {path}")
    return {
        str(row["id"])
        for row in rows
        if bool(row["judgment"].get("behavior_success"))
    }


def main() -> int:
    args = parse_args()
    if args.first == args.second:
        raise ValueError("Rank-2 directions must be different.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    from safetensors.numpy import load_file, save_file

    directions = load_file(str(args.rank1_directions))
    for key in (args.first, args.second):
        if key not in directions:
            raise KeyError(f"Direction is missing: {key}")
    first = np.asarray(directions[args.first], dtype=np.float32)
    second = np.asarray(directions[args.second], dtype=np.float32)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = float(np.dot(first, second))
    if abs(cosine) > args.maximum_absolute_cosine:
        raise RuntimeError(
            f"Directions are too similar for rank 2: cosine={cosine:.6f}"
        )

    complement = {}
    minimum_unique = None
    for group in GROUPS:
        first_success = successes(args.harmbench_root / args.first / f"{group}.jsonl")
        second_success = successes(args.harmbench_root / args.second / f"{group}.jsonl")
        first_only = first_success - second_success
        second_only = second_success - first_success
        minimum_unique = min(
            len(first_only),
            len(second_only),
            minimum_unique if minimum_unique is not None else len(first_only),
        )
        complement[group] = {
            "first_success_count": len(first_success),
            "second_success_count": len(second_success),
            "overlap_success_count": len(first_success & second_success),
            "first_only_success_count": len(first_only),
            "second_only_success_count": len(second_only),
            "potential_union_success_count": len(first_success | second_success),
        }
    if minimum_unique is None or minimum_unique <= 0:
        raise RuntimeError("The directions do not have complementary HarmBench successes.")

    basis = orthonormal_pair(first, second)
    args.output_dir.mkdir(parents=True)
    temporary = args.output_dir / ".rank2_directions.safetensors.tmp"
    output = args.output_dir / "rank2_directions.safetensors"
    save_file({"harmbench_complementary_rank2": basis}, str(temporary))
    os.replace(temporary, output)
    report = {
        "method": "HarmBench complementary rank-1 union, then QR orthonormalization",
        "first": args.first,
        "second": args.second,
        "signed_cosine": cosine,
        "absolute_cosine": abs(cosine),
        "maximum_absolute_cosine": args.maximum_absolute_cosine,
        "complementarity": complement,
        "direction_file": str(output),
        "direction_file_sha256": sha256_file(output),
        "rank1_directions_sha256": sha256_file(args.rank1_directions),
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_report": False,
    }
    write_json(args.output_dir / "rank2_report.json", report)
    print(json.dumps({"status": "complete", **complement}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
