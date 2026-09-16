#!/usr/bin/env python3
"""Build automatically massive-coordinate-masked rank-1 directions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from swift_abliteration.direction_study import (
    bootstrap_masked_direction_stability,
    coordinate_masked_direction,
    cosine_similarity,
    massive_activation_coordinate_mask,
    normalized_average,
    standardized_separation,
)
from swift_abliteration.gpu_support import sha256_file
from swift_abliteration.metrics import summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-directions")
    parser.add_argument("--log-robust-z-threshold", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    from safetensors.numpy import load_file, save_file

    activations = load_file(args.capture)
    references = load_file(args.reference_directions) if args.reference_directions else {}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    layer_pattern = re.compile(r"standard_direction_harmful_layer_(\d+)$")
    layers = sorted(
        int(match.group(1))
        for key in activations
        if (match := layer_pattern.fullmatch(key)) is not None
    )
    if not layers:
        raise RuntimeError("The capture contains no candidate-layer activations.")

    directions: dict[str, np.ndarray] = {}
    report = {
        "schema_version": 1,
        "estimator": "automatically detected massive-coordinate masked mean difference",
        "capture": args.capture,
        "capture_sha256": sha256_file(Path(args.capture)),
        "log_robust_z_threshold": args.log_robust_z_threshold,
        "layers": {},
        "uses_final_test": False,
        "checkpoint_saved": False,
    }
    for layer in layers:
        suffix = f"_layer_{layer}"
        source_arrays = {
            source: {
                split: activations[f"{source}_direction_{split}{suffix}"]
                for split in ("harmful", "harmless")
            }
            for source in ("standard", "matched")
        }
        mask, mask_details = massive_activation_coordinate_mask(
            [
                source_arrays[source][split]
                for source in ("standard", "matched")
                for split in ("harmful", "harmless")
            ],
            args.log_robust_z_threshold,
        )
        layer_directions = {}
        source_report = {}
        for source in ("standard", "matched"):
            harmful = source_arrays[source]["harmful"]
            harmless = source_arrays[source]["harmless"]
            direction = coordinate_masked_direction(harmful, harmless, mask)
            key = f"{source}_layer_{layer}_massive_masked"
            directions[key] = direction.astype(np.float32)
            layer_directions[source] = direction
            stability = bootstrap_masked_direction_stability(
                harmful,
                harmless,
                mask,
                direction,
                samples=args.bootstrap_samples,
            )
            eval_harmful = activations[f"{source}_evaluation_harmful{suffix}"]
            eval_harmless = activations[f"{source}_evaluation_harmless{suffix}"]
            source_report[source] = {
                "bootstrap_cosine": summarize(stability),
                "heldout_standardized_separation": standardized_separation(
                    eval_harmful, eval_harmless, direction
                ),
            }
            for reference_name in ("plain", "winsor_995"):
                reference_key = f"{source}_layer_{layer}_{reference_name}"
                if reference_key in references:
                    source_report[source][f"cosine_to_{reference_name}"] = (
                        cosine_similarity(direction, references[reference_key])
                    )
        consensus = normalized_average(
            [layer_directions["standard"], layer_directions["matched"]]
        )
        directions[f"consensus_layer_{layer}_massive_masked"] = consensus.astype(
            np.float32
        )
        report["layers"][str(layer)] = {
            "mask": mask_details,
            "cross_source_cosine": cosine_similarity(
                layer_directions["standard"], layer_directions["matched"]
            ),
            "sources": source_report,
        }

    direction_path = output / "massive_masked_directions.safetensors"
    save_file(directions, str(direction_path))
    report["direction_count"] = len(directions)
    report["directions_sha256"] = sha256_file(direction_path)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "direction_count": len(directions),
                "output_dir": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
