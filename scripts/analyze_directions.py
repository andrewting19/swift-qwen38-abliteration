#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from swift_abliteration.direction_study import (
    bootstrap_cosine_stability,
    cosine_similarity,
    standardized_separation,
    winsorized_direction,
)
from swift_abliteration.gpu_support import sha256_file
from swift_abliteration.math_core import refusal_direction
from swift_abliteration.metrics import summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from safetensors.numpy import load_file, save_file

    activations = load_file(args.capture)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    directions = {}
    report = {"sources": {}, "cross_source_plain_cosine": {}}
    layer_pattern = re.compile(r"standard_direction_harmful_layer_(\d+)$")
    layers = sorted(
        int(match.group(1))
        for key in activations
        if (match := layer_pattern.fullmatch(key)) is not None
    )
    if not layers:
        raise RuntimeError("The capture contains no candidate-layer activations.")
    for source in ("standard", "matched"):
        report["sources"][source] = {}
        for layer_index in layers:
            suffix = f"_layer_{layer_index}"
            harmful = activations[f"{source}_direction_harmful{suffix}"]
            harmless = activations[f"{source}_direction_harmless{suffix}"]
            eval_harmful = activations[f"{source}_evaluation_harmful{suffix}"]
            eval_harmless = activations[f"{source}_evaluation_harmless{suffix}"]
            plain = refusal_direction(harmful, harmless)
            candidates = {"plain": (plain, None, 0.0)}
            for quantile in (0.990, 0.995, 0.999):
                direction, threshold, changed = winsorized_direction(
                    harmful, harmless, quantile
                )
                candidates[f"winsor_{int(quantile * 1000):03d}"] = (
                    direction,
                    threshold,
                    changed,
                )
            layer_report = {}
            for name, (direction, threshold, changed) in candidates.items():
                key = f"{source}_layer_{layer_index}_{name}"
                directions[key] = direction.astype(np.float32)
                stability = bootstrap_cosine_stability(
                    harmful, harmless, direction, samples=500
                )
                layer_report[name] = {
                    "threshold": threshold,
                    "changed_fraction": changed,
                    "cosine_to_plain": cosine_similarity(direction, plain),
                    "bootstrap_cosine": summarize(stability),
                    "heldout_standardized_separation": standardized_separation(
                        eval_harmful, eval_harmless, direction
                    ),
                }
            report["sources"][source][str(layer_index)] = layer_report
    report["cross_source_plain_cosine"] = {
        str(layer_index): cosine_similarity(
            directions[f"standard_layer_{layer_index}_plain"],
            directions[f"matched_layer_{layer_index}_plain"],
        )
        for layer_index in layers
    }
    directions_path = output / "directions.safetensors"
    save_file(directions, str(directions_path))
    report["directions_sha256"] = sha256_file(directions_path)
    (output / "direction_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "direction_count": len(directions),
                "output_dir": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
