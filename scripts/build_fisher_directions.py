#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from swift_abliteration.direction_study import (
    cosine_similarity,
    normalized_average,
    ridge_fisher_direction,
    standardized_separation,
)
from swift_abliteration.gpu_support import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build covariance-aware Fisher refusal directions."
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=52)
    parser.add_argument(
        "--shrinkage", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2, 0.5]
    )
    parser.add_argument("--regularization-ratio", type=float, default=1e-4)
    args = parser.parse_args()

    from safetensors.numpy import load_file, save_file

    activations = load_file(str(args.capture))
    layer = args.layer
    tensors = {}
    report = {
        "method": "ridge_fisher_within_class_covariance",
        "layer": layer,
        "source_capture": str(args.capture.resolve()),
        "source_capture_sha256": sha256_file(args.capture),
        "uses_final_test": False,
        "candidates": {},
    }
    groups = {
        source: (
            activations[f"{source}_direction_harmful_layer_{layer}"],
            activations[f"{source}_direction_harmless_layer_{layer}"],
        )
        for source in ("standard", "matched")
    }
    evaluation = {
        source: (
            activations[f"{source}_evaluation_harmful_layer_{layer}"],
            activations[f"{source}_evaluation_harmless_layer_{layer}"],
        )
        for source in ("standard", "matched")
    }
    for shrinkage in args.shrinkage:
        label = str(shrinkage).replace(".", "p")
        source_values = {}
        source_details = {}
        for source in ("standard", "matched"):
            direction, details = ridge_fisher_direction(
                groups[source][0],
                groups[source][1],
                shrinkage,
                args.regularization_ratio,
            )
            key = f"{source}_fisher_layer_{layer}_shrink_{label}"
            tensors[key] = direction
            source_values[source] = direction
            source_details[source] = details
        consensus = normalized_average(
            [source_values["standard"], source_values["matched"]]
        ).astype("float32")
        key = f"consensus_fisher_layer_{layer}_shrink_{label}"
        tensors[key] = consensus
        report["candidates"][key] = {
            "source_details": source_details,
            "source_cosine": cosine_similarity(
                source_values["standard"], source_values["matched"]
            ),
            "heldout_standardized_separation": {
                source: standardized_separation(
                    evaluation[source][0], evaluation[source][1], consensus
                )
                for source in ("standard", "matched")
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tensor_path = args.output_dir / "fisher_directions.safetensors"
    save_file(tensors, str(tensor_path))
    report["directions_sha256"] = sha256_file(tensor_path)
    (args.output_dir / "fisher_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "candidate_count": len(tensors)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
