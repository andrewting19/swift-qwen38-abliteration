#!/usr/bin/env python3
"""Build projected rank-1 and rank-2 candidates from saved safe activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from swift_abliteration.gpu_support import sha256_file, write_json


def project_away_from_mean(direction: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Remove the unit mean component from a direction and normalize the result."""
    value = np.asarray(direction, dtype=np.float64)
    anchor = np.asarray(mean, dtype=np.float64)
    anchor_norm = np.linalg.norm(anchor)
    if not np.isfinite(anchor_norm) or anchor_norm <= 0.0:
        raise ValueError("The harmless mean has zero or invalid length.")
    anchor = anchor / anchor_norm
    projected = value - float(value @ anchor) * anchor
    projected_norm = np.linalg.norm(projected)
    if not np.isfinite(projected_norm) or projected_norm <= 1e-12:
        raise ValueError("Projection removed the complete direction.")
    return (projected / projected_norm).astype(np.float32)


def orthonormal_rows(vectors: list[np.ndarray]) -> np.ndarray:
    """Return a QR-orthonormal row basis for the supplied vectors."""
    values = np.stack(vectors).astype(np.float64)
    basis = np.linalg.qr(values.T, mode="reduced")[0].T
    return basis.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--position", type=int, action="append", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    from safetensors.torch import load_file, save_file
    import torch

    positions = list(dict.fromkeys(args.position))
    if len(positions) != 2 or any(position >= 0 for position in positions):
        raise ValueError("Supply exactly two distinct negative prompt positions.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    position_values = [int(value) for value in manifest["positions"]]
    activations = load_file(str(args.activations), device="cpu")
    directions = load_file(str(args.directions), device="cpu")
    harmless_key = f"harmless_resid_pre_layer_{args.layer}"
    if harmless_key not in activations:
        raise KeyError(f"Missing activation tensor: {harmless_key}")

    original: list[np.ndarray] = []
    projected: list[np.ndarray] = []
    diagnostics: dict[str, dict] = {}
    output: dict[str, torch.Tensor] = {}
    for position in positions:
        candidate_key = (
            f"behavior_filtered_position_minus_{abs(position)}_"
            f"layer_{args.layer}_massive_masked"
        )
        if candidate_key not in directions:
            raise KeyError(f"Missing direction tensor: {candidate_key}")
        position_index = position_values.index(position)
        report = manifest["candidates"][candidate_key]
        masked_indices = [
            int(value)
            for value in report["massive_activation_mask"]["selected_indices"]
        ]
        harmless_mean = (
            activations[harmless_key][:, position_index, :]
            .float()
            .mean(dim=0)
            .numpy()
        )
        harmless_mean[masked_indices] = 0.0
        direction = directions[candidate_key].float().numpy()
        direction = direction / np.linalg.norm(direction)
        refined = project_away_from_mean(direction, harmless_mean)
        harmless_unit = harmless_mean / np.linalg.norm(harmless_mean)
        original.append(direction.astype(np.float32))
        projected.append(refined)
        output[f"original_position_minus_{abs(position)}_layer_{args.layer}"] = (
            torch.from_numpy(direction.astype(np.float32))
        )
        output[f"projected_position_minus_{abs(position)}_layer_{args.layer}"] = (
            torch.from_numpy(refined)
        )
        diagnostics[str(position)] = {
            "source_key": candidate_key,
            "masked_indices": masked_indices,
            "direction_harmless_mean_cosine": float(direction @ harmless_unit),
            "projected_harmless_mean_cosine": float(refined @ harmless_unit),
            "original_projected_cosine": float(direction @ refined),
        }

    original_basis = orthonormal_rows(original)
    projected_basis = orthonormal_rows(projected)
    output["original_complementary_rank2"] = torch.from_numpy(
        original_basis.copy()
    ).contiguous()
    output["projected_complementary_rank2"] = torch.from_numpy(
        projected_basis.copy()
    ).contiguous()

    args.output_dir.mkdir(parents=True)
    tensor_path = args.output_dir / "projected_candidates.safetensors"
    save_file(output, str(tensor_path))
    report = {
        "schema_version": 1,
        "method": "project each local direction away from its masked harmless mean, then QR",
        "layer": args.layer,
        "positions": positions,
        "candidate_keys": sorted(output),
        "diagnostics": diagnostics,
        "pair_cosine_before_projection": float(original[0] @ original[1]),
        "pair_cosine_after_projection": float(projected[0] @ projected[1]),
        "artifacts": {
            "activations_sha256": sha256_file(args.activations),
            "directions_sha256": sha256_file(args.directions),
            "manifest_sha256": sha256_file(args.manifest),
            "output_sha256": sha256_file(tensor_path),
        },
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_report": False,
    }
    write_json(args.output_dir / "projected_candidates_report.json", report)
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_count": len(output),
                "positions": positions,
                "raw_text_printed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
