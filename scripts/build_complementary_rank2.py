#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

from swift_abliteration.gpu_support import sha256_file, write_json


HARMFUL_GROUPS = ("standard_harmful", "matched_harmful")
SIGNALS = ("arditi_anywhere_refusal", "opening_refusal")


def orthonormal_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    values = np.stack((first, second)).astype(np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("Two direction vectors are required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Direction vectors must be finite.")
    q, _ = np.linalg.qr(values.T, mode="reduced")
    # Safetensors requires a C-contiguous array. The transpose is otherwise a
    # strided view whose serialized row layout can differ from the QR result.
    basis = np.ascontiguousarray(q.T, dtype=np.float32)
    if basis.shape[0] != 2 or not np.allclose(
        basis @ basis.T, np.eye(2), atol=2e-5, rtol=2e-5
    ):
        raise ValueError("Directions are linearly dependent or QR failed.")
    return basis


def removed_labels(base: list[bool], candidate: list[bool]) -> np.ndarray:
    base_values = np.asarray(base, dtype=bool)
    candidate_values = np.asarray(candidate, dtype=bool)
    if base_values.shape != candidate_values.shape:
        raise ValueError("Prompt-level refusal labels do not match.")
    return np.logical_and(base_values, ~candidate_values)


def pair_effect(first: dict, second: dict, base: dict) -> dict:
    details = {}
    potential = []
    gains = []
    for group in HARMFUL_GROUPS:
        details[group] = {}
        for signal in SIGNALS:
            base_labels = base["groups"][group][f"{signal}_labels"]
            first_labels = first["groups"][group][f"{signal}_labels"]
            second_labels = second["groups"][group][f"{signal}_labels"]
            first_removed = removed_labels(base_labels, first_labels)
            second_removed = removed_labels(base_labels, second_labels)
            union = np.logical_or(first_removed, second_removed)
            base_count = int(np.asarray(base_labels, dtype=bool).sum())
            denominator = max(base_count, 1)
            first_rate = float(first_removed.sum() / denominator)
            second_rate = float(second_removed.sum() / denominator)
            union_rate = float(union.sum() / denominator)
            gain = union_rate - max(first_rate, second_rate)
            details[group][signal] = {
                "first_removed_count": int(first_removed.sum()),
                "second_removed_count": int(second_removed.sum()),
                "union_removed_count": int(union.sum()),
                "overlap_removed_count": int(
                    np.logical_and(first_removed, second_removed).sum()
                ),
                "potential_union_removal": union_rate,
                "complementary_gain": gain,
            }
            potential.append(union_rate)
            gains.append(gain)
    return {
        "details": details,
        "minimum_potential_union_removal": min(potential),
        "mean_potential_union_removal": float(np.mean(potential)),
        "minimum_complementary_gain": min(gains),
        "mean_complementary_gain": float(np.mean(gains)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build rank-2 pairs that removed complementary rank-1 refusals."
    )
    parser.add_argument("--rank1-report", type=Path, required=True)
    parser.add_argument("--rank1-directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=12)
    parser.add_argument("--maximum-pairs", type=int, default=12)
    parser.add_argument("--maximum-absolute-cosine", type=float, default=0.98)
    parser.add_argument("--maximum-uses-per-direction", type=int, default=4)
    args = parser.parse_args()
    if args.maximum_pairs <= 0 or args.pool_size < 2:
        raise ValueError("At least two directions and one pair are required.")

    from safetensors.numpy import load_file, save_file

    report = json.loads(args.rank1_report.read_text(encoding="utf-8"))
    if report.get("uses_final_test") is not False:
        raise RuntimeError(
            "Rank-1 report does not prove that final-test data was unused."
        )
    if report.get("sufficient_rank1"):
        raise RuntimeError("Rank 1 is sufficient. Do not build rank 2.")
    directions = load_file(str(args.rank1_directions))
    pool = [key for key in report["ranked_eligible"] if key in directions][
        : args.pool_size
    ]
    if len(pool) < 2:
        raise RuntimeError("Fewer than two eligible rank-1 directions are available.")

    candidates = []
    base = report["base"]
    for first_key, second_key in combinations(pool, 2):
        first = np.asarray(directions[first_key], dtype=np.float32)
        second = np.asarray(directions[second_key], dtype=np.float32)
        first_unit = first / np.linalg.norm(first)
        second_unit = second / np.linalg.norm(second)
        cosine = float(np.dot(first_unit, second_unit))
        if abs(cosine) > args.maximum_absolute_cosine:
            continue
        try:
            basis = orthonormal_pair(first_unit, second_unit)
        except ValueError:
            continue
        effect = pair_effect(
            report["results"][first_key], report["results"][second_key], base
        )
        candidates.append(
            {
                "first": first_key,
                "second": second_key,
                "absolute_cosine": abs(cosine),
                "signed_cosine": cosine,
                "effect": effect,
                "basis": basis,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["effect"]["minimum_potential_union_removal"],
            item["effect"]["minimum_complementary_gain"],
            item["effect"]["mean_potential_union_removal"],
            -item["absolute_cosine"],
        ),
        reverse=True,
    )

    selected = []
    use_counts = {key: 0 for key in pool}
    for item in candidates:
        if len(selected) >= args.maximum_pairs:
            break
        if (
            use_counts[item["first"]] >= args.maximum_uses_per_direction
            or use_counts[item["second"]] >= args.maximum_uses_per_direction
        ):
            continue
        selected.append(item)
        use_counts[item["first"]] += 1
        use_counts[item["second"]] += 1
    if not selected:
        raise RuntimeError("No complementary rank-2 pair passed the cosine limit.")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tensors = {}
    metadata = {}
    for index, item in enumerate(selected, start=1):
        key = f"rank2_{index:02d}"
        tensors[key] = item.pop("basis")
        metadata[key] = item
    temporary = args.output_dir / ".rank2_directions.safetensors.tmp"
    output = args.output_dir / "rank2_directions.safetensors"
    save_file(tensors, str(temporary))
    os.replace(temporary, output)
    write_json(
        args.output_dir / "rank2_report.json",
        {
            "method": "prompt-level complementary rank-1 union, then QR orthonormalization",
            "rank1_report": str(args.rank1_report),
            "rank1_report_sha256": sha256_file(args.rank1_report),
            "rank1_directions": str(args.rank1_directions),
            "rank1_directions_sha256": sha256_file(args.rank1_directions),
            "pool": pool,
            "pool_size": len(pool),
            "maximum_absolute_cosine": args.maximum_absolute_cosine,
            "maximum_uses_per_direction": args.maximum_uses_per_direction,
            "candidate_count": len(metadata),
            "candidates": metadata,
            "direction_file": str(output),
            "direction_file_sha256": sha256_file(output),
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_report": False,
        },
    )
    print(
        json.dumps(
            {"status": "complete", "candidate_count": len(metadata)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
