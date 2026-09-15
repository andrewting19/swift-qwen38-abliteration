#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from swift_abliteration.direction_study import (
    bootstrap_consensus_stability,
    cosine_similarity,
    normalized_average,
    standardized_separation,
    winsorized_direction,
)
from swift_abliteration.gpu_support import sha256_file, write_json
from swift_abliteration.refusal_scoring import outcome_masks


SOURCES = ("standard", "matched")
POSITIONS = ("prompt_end", "first_output")


def load_group(
    capture_dir: Path, name: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    tensors = load_file(capture_dir / "groups" / f"{name}.safetensors")
    with np.load(capture_dir / "groups" / f"{name}.npz") as values:
        scores = values["refusal_scores"].copy()
    return tensors, scores


def filtered_pair(
    capture_dir: Path, source: str, split: str, position: str, layer: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    harmful_name = f"{source}_{split}_harmful"
    harmless_name = f"{source}_{split}_harmless"
    harmful_tensors, harmful_scores = load_group(capture_dir, harmful_name)
    harmless_tensors, harmless_scores = load_group(capture_dir, harmless_name)
    harmful_mask, harmless_mask = outcome_masks(harmful_scores, harmless_scores)
    key = f"{position}_layer_{layer}"
    harmful = harmful_tensors[key][harmful_mask]
    harmless = harmless_tensors[key][harmless_mask]
    return (
        harmful,
        harmless,
        {
            "harmful_total": int(len(harmful_scores)),
            "harmful_kept": int(harmful_mask.sum()),
            "harmless_total": int(len(harmless_scores)),
            "harmless_kept": int(harmless_mask.sum()),
        },
    )


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build simple outcome-filtered winsorized rank-1 candidates."
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--winsor-quantile", type=float, default=0.995)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--minimum-train-per-class", type=int, default=8)
    parser.add_argument("--minimum-eval-per-class", type=int, default=4)
    parser.add_argument("--minimum-bootstrap-median", type=float, default=0.90)
    parser.add_argument("--shortlist-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3819)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    capture_manifest_path = args.capture_dir / "capture_manifest.json"
    manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    layers = [int(value) for value in manifest["direction_layers"]]
    directions: dict[str, np.ndarray] = {}
    reports: dict[str, dict] = {}

    pairs: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray, dict]] = {}
    for source in SOURCES:
        for split in ("direction", "evaluation"):
            for position in POSITIONS:
                for layer in layers:
                    pairs[(source, split, position, layer)] = filtered_pair(
                        args.capture_dir, source, split, position, layer
                    )

    for position in POSITIONS:
        for layer in layers:
            source_values = {}
            for source in SOURCES:
                harmful, harmless, train_counts = pairs[
                    (source, "direction", position, layer)
                ]
                eval_harmful, eval_harmless, eval_counts = pairs[
                    (source, "evaluation", position, layer)
                ]
                if min(len(harmful), len(harmless)) < args.minimum_train_per_class:
                    continue
                if (
                    min(len(eval_harmful), len(eval_harmless))
                    < args.minimum_eval_per_class
                ):
                    continue
                direction, threshold, changed = winsorized_direction(
                    harmful, harmless, args.winsor_quantile
                )
                bootstrap = bootstrap_consensus_stability(
                    [(harmful, harmless)],
                    direction,
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                    winsor_quantile=args.winsor_quantile,
                )
                key = f"{source}_{position}_layer_{layer}_winsor_995"
                directions[key] = direction.astype(np.float32)
                source_values[source] = (direction, harmful, harmless)
                separations = {}
                for eval_source in SOURCES:
                    other_harmful, other_harmless, _ = pairs[
                        (eval_source, "evaluation", position, layer)
                    ]
                    separations[eval_source] = standardized_separation(
                        other_harmful, other_harmless, direction
                    )
                reports[key] = {
                    "source": source,
                    "position": position,
                    "layer": layer,
                    "train_counts": train_counts,
                    "evaluation_counts": eval_counts,
                    "winsor_threshold": threshold,
                    "changed_fraction": changed,
                    "bootstrap_cosine": summarize(bootstrap),
                    "heldout_standardized_separation": separations,
                    "minimum_heldout_separation": min(separations.values()),
                }

            if len(source_values) == len(SOURCES):
                source_directions = [source_values[source][0] for source in SOURCES]
                consensus = normalized_average(source_directions)
                bootstrap = bootstrap_consensus_stability(
                    [
                        (source_values[source][1], source_values[source][2])
                        for source in SOURCES
                    ],
                    consensus,
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                    winsor_quantile=args.winsor_quantile,
                )
                key = f"consensus_{position}_layer_{layer}_winsor_995"
                directions[key] = consensus.astype(np.float32)
                separations = {}
                for eval_source in SOURCES:
                    harmful, harmless, _ = pairs[
                        (eval_source, "evaluation", position, layer)
                    ]
                    separations[eval_source] = standardized_separation(
                        harmful, harmless, consensus
                    )
                reports[key] = {
                    "source": "consensus",
                    "position": position,
                    "layer": layer,
                    "source_direction_cosine": cosine_similarity(
                        source_directions[0], source_directions[1]
                    ),
                    "bootstrap_cosine": summarize(bootstrap),
                    "heldout_standardized_separation": separations,
                    "minimum_heldout_separation": min(separations.values()),
                }

    eligible = [
        (key, report)
        for key, report in reports.items()
        if report["bootstrap_cosine"]["median"] >= args.minimum_bootstrap_median
        and report["minimum_heldout_separation"] > 0
    ]
    selected = []
    per_position = max(1, args.shortlist_size // len(POSITIONS))
    for position in POSITIONS:
        ranked = sorted(
            (item for item in eligible if item[1]["position"] == position),
            key=lambda item: (
                item[1]["minimum_heldout_separation"],
                item[1]["bootstrap_cosine"]["median"],
            ),
            reverse=True,
        )
        selected.extend(key for key, _ in ranked[:per_position])
    if len(selected) < args.shortlist_size:
        remaining = sorted(
            (item for item in eligible if item[0] not in selected),
            key=lambda item: item[1]["minimum_heldout_separation"],
            reverse=True,
        )
        selected.extend(
            key for key, _ in remaining[: args.shortlist_size - len(selected)]
        )
    if not selected:
        raise RuntimeError("No direction passed the statistical shortlist gates.")

    direction_path = args.output_dir / "rank1_directions.safetensors"
    save_file(directions, direction_path)
    write_json(
        args.output_dir / "direction_report.json",
        {
            "capture_manifest_sha256": sha256_file(capture_manifest_path),
            "direction_file_sha256": sha256_file(direction_path),
            "winsor_quantile": args.winsor_quantile,
            "bootstrap_samples": args.bootstrap_samples,
            "candidate_count": len(directions),
            "candidates": reports,
            "shortlist": selected,
            "shortlist_size": len(selected),
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_report": False,
        },
    )
    print(json.dumps({"status": "complete", "shortlist": selected}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
