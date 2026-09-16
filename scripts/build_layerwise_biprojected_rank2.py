#!/usr/bin/env python3
"""Build one target-layer biprojected rank-2 candidate from local directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_projected_position_candidates import (
    orthonormal_rows,
    project_away_from_mean,
)
from swift_abliteration.gpu_support import sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directions", type=Path, required=True)
    parser.add_argument("--harmless-means", type=Path, required=True)
    parser.add_argument("--harmless-means-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-position", type=int, default=-12)
    parser.add_argument("--second-position", type=int, default=-13)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    import torch
    from safetensors.torch import load_file, save_file

    directions = load_file(str(args.source_directions), device="cpu")
    means = load_file(str(args.harmless_means), device="cpu")
    means_report = json.loads(args.harmless_means_report.read_text(encoding="utf-8"))
    layers = [int(value) for value in means_report["layers"]]
    source_keys = [
        f"original_position_minus_{abs(args.first_position)}_layer_32",
        f"original_position_minus_{abs(args.second_position)}_layer_32",
    ]
    missing = [key for key in source_keys if key not in directions]
    if missing:
        raise KeyError(f"Missing source directions: {missing}")
    source_rows = [directions[key].float().numpy() for key in source_keys]
    original_basis = orthonormal_rows(source_rows)

    layerwise = []
    diagnostics = {}
    for layer in layers:
        mean_keys = [
            f"harmless_position_minus_{abs(args.first_position)}_layer_{layer}_masked_mean",
            f"harmless_position_minus_{abs(args.second_position)}_layer_{layer}_masked_mean",
        ]
        missing_means = [key for key in mean_keys if key not in means]
        if missing_means:
            raise KeyError(f"Missing harmless means: {missing_means}")
        harmless_rows = [means[key].float().numpy() for key in mean_keys]
        projected_rows = [
            project_away_from_mean(direction, harmless)
            for direction, harmless in zip(source_rows, harmless_rows, strict=True)
        ]
        basis = orthonormal_rows(projected_rows)
        layerwise.append(basis)
        principal_cosines = np.linalg.svd(
            original_basis @ basis.T, compute_uv=False
        )
        diagnostics[str(layer)] = {
            "source_to_projected_cosine": [
                float(source @ projected)
                for source, projected in zip(
                    source_rows, projected_rows, strict=True
                )
            ],
            "projected_row_pair_cosine": float(projected_rows[0] @ projected_rows[1]),
            "original_subspace_principal_cosines": [
                float(value) for value in principal_cosines
            ],
        }

    tensor = np.stack(layerwise).astype(np.float32)
    args.output_dir.mkdir(parents=True)
    tensor_path = args.output_dir / "layerwise_biprojected_rank2.safetensors"
    save_file(
        {"target_layer_biprojected_rank2": torch.from_numpy(tensor).contiguous()},
        str(tensor_path),
    )
    report = {
        "schema_version": 1,
        "method": "project each source row away from the matching target-layer harmless mean, then QR",
        "source_positions": [args.first_position, args.second_position],
        "layers": layers,
        "shape": list(tensor.shape),
        "diagnostics": diagnostics,
        "artifacts": {
            "source_directions_sha256": sha256_file(args.source_directions),
            "harmless_means_sha256": sha256_file(args.harmless_means),
            "harmless_means_report_sha256": sha256_file(args.harmless_means_report),
            "output_sha256": sha256_file(tensor_path),
        },
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_report": False,
    }
    write_json(args.output_dir / "layerwise_biprojected_rank2_report.json", report)
    print(
        json.dumps(
            {
                "status": "complete",
                "shape": list(tensor.shape),
                "raw_text_printed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
