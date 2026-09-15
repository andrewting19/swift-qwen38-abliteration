#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.numpy import load_file, save_file

from swift_abliteration.direction_study import (
    cosine_similarity,
    orthogonalize_direction,
    winsorized_direction,
)
from swift_abliteration.gpu_support import sha256_file, write_json


SOURCES = ("standard", "matched")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build source-specific residual directions from an iterative capture."
    )
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--basis-rank", type=int, required=True)
    parser.add_argument("--capture-rank", type=int, required=True)
    parser.add_argument("--winsor-quantile", type=float, default=0.995)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.basis_rank <= 0 or args.capture_rank <= args.basis_rank:
        raise ValueError("capture-rank must be greater than the positive basis-rank.")

    source_directions = load_file(args.directions)
    activations = load_file(args.activations)
    basis_keys = [
        f"iterative_direction_{index}" for index in range(1, args.basis_rank + 1)
    ]
    missing_basis = [key for key in basis_keys if key not in source_directions]
    if missing_basis:
        raise KeyError(f"Missing basis directions: {missing_basis}")
    basis = []
    for key in basis_keys:
        value, _ = orthogonalize_direction(source_directions[key], basis)
        basis.append(value)

    output_tensors = {
        key: value for key, value in zip(basis_keys, basis, strict=True)
    }
    report = {
        "basis_rank": args.basis_rank,
        "capture_rank": args.capture_rank,
        "winsor_quantile": args.winsor_quantile,
        "sources": {},
    }
    residuals = {}
    raw = {}
    for source in SOURCES:
        harmful_key = f"rank{args.capture_rank}_before_{source}_harmful"
        harmless_key = f"rank{args.capture_rank}_before_{source}_harmless"
        if harmful_key not in activations or harmless_key not in activations:
            raise KeyError(f"Missing activation pair for {source}.")
        candidate, threshold, changed = winsorized_direction(
            activations[harmful_key],
            activations[harmless_key],
            args.winsor_quantile,
        )
        residual, residual_norm = orthogonalize_direction(candidate, basis)
        raw[source] = candidate
        residuals[source] = residual
        key = f"rank{args.capture_rank}_{source}_residual"
        output_tensors[key] = residual
        report["sources"][source] = {
            "output_key": key,
            "winsor_threshold": threshold,
            "changed_fraction": changed,
            "orthogonal_residual_norm": residual_norm,
            "cosine_to_basis": [
                cosine_similarity(candidate, direction) for direction in basis
            ],
        }

    report["raw_standard_matched_cosine"] = cosine_similarity(
        raw["standard"], raw["matched"]
    )
    report["residual_standard_matched_cosine"] = cosine_similarity(
        residuals["standard"], residuals["matched"]
    )
    joint_basis = list(basis)
    for index, source in enumerate(("matched", "standard"), start=1):
        direction, residual_norm = orthogonalize_direction(raw[source], joint_basis)
        joint_basis.append(direction)
        output_tensors[f"rank{args.capture_rank}_joint_{index}"] = direction
        report[f"joint_{index}"] = {
            "source": source,
            "orthogonal_residual_norm": residual_norm,
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "branch_directions.safetensors"
    save_file(output_tensors, output_path)
    report.update(
        {
            "source_directions": str(args.directions.resolve()),
            "source_directions_sha256": sha256_file(args.directions),
            "source_activations": str(args.activations.resolve()),
            "source_activations_sha256": sha256_file(args.activations),
            "output_sha256": sha256_file(output_path),
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
        }
    )
    write_json(args.output_dir / "branch_report.json", report)
    print(json.dumps({"status": "complete", "output": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
