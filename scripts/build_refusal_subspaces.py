#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from swift_abliteration.gpu_support import sha256_file


def orthonormal_rows(vectors: np.ndarray, rank: int) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not 0 < rank <= min(values.shape):
        raise ValueError("Invalid vectors or requested rank.")
    q, r = np.linalg.qr(values.T, mode="reduced")
    diagonal = np.abs(np.diag(r))
    if np.count_nonzero(diagonal > 1e-6) < rank:
        raise ValueError("Vectors do not contain the requested independent rank.")
    return q[:, :rank].T.astype(np.float32)


def matched_svd_basis(
    harmful: np.ndarray,
    harmless: np.ndarray,
    rank: int,
    quantile: float = 0.995,
) -> tuple[np.ndarray, dict]:
    harmful = np.asarray(harmful, dtype=np.float32)
    harmless = np.asarray(harmless, dtype=np.float32)
    if harmful.shape != harmless.shape or harmful.ndim != 2:
        raise ValueError("Matched harmful and harmless activations must align.")
    threshold = float(
        np.quantile(np.abs(np.concatenate([harmful, harmless], axis=0)), quantile)
    )
    clipped_harmful = np.clip(harmful, -threshold, threshold)
    clipped_harmless = np.clip(harmless, -threshold, threshold)
    differences = clipped_harmful - clipped_harmless
    mean = differences.mean(axis=0)
    mean /= np.linalg.norm(mean)
    centered = differences - differences.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    candidates = np.concatenate([mean[None, :], vh], axis=0)
    basis = orthonormal_rows(candidates, rank)
    energy = singular_values**2
    return basis, {
        "rank": rank,
        "winsor_quantile": quantile,
        "winsor_threshold": threshold,
        "centered_svd_energy_fraction": float(
            energy[: max(0, rank - 1)].sum() / energy.sum()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build small refusal subspaces from saved training activations."
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[24, 32, 38, 44, 52])
    args = parser.parse_args()

    from safetensors.numpy import load_file, save_file

    activations = load_file(str(args.capture))
    directions = load_file(str(args.directions))
    tensors = {}
    report = {
        "uses_final_test": False,
        "source_capture_sha256": sha256_file(args.capture),
        "source_directions_sha256": sha256_file(args.directions),
        "layers": {},
    }
    for layer in args.layers:
        source_vectors = np.stack(
            [
                directions[f"standard_layer_{layer}_winsor_995"],
                directions[f"matched_layer_{layer}_winsor_995"],
            ]
        )
        source_basis = orthonormal_rows(source_vectors, 2)
        source_key = f"source_span_layer_{layer}_rank2"
        tensors[source_key] = source_basis

        harmful = activations[f"matched_direction_harmful_layer_{layer}"]
        harmless = activations[f"matched_direction_harmless_layer_{layer}"]
        layer_report = {
            "source_span_rank2_gram_max_error": float(
                np.max(np.abs(source_basis @ source_basis.T - np.eye(2)))
            ),
            "matched_svd": {},
        }
        for rank in (2, 4, 8):
            basis, details = matched_svd_basis(harmful, harmless, rank)
            key = f"matched_svd_layer_{layer}_rank{rank}"
            tensors[key] = basis
            details["gram_max_error"] = float(
                np.max(np.abs(basis @ basis.T - np.eye(rank)))
            )
            layer_report["matched_svd"][str(rank)] = details
        report["layers"][str(layer)] = layer_report

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tensor_path = args.output_dir / "refusal_subspaces.safetensors"
    save_file(tensors, str(tensor_path))
    report["subspaces_sha256"] = sha256_file(tensor_path)
    (args.output_dir / "subspace_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "complete", "tensor_count": len(tensors)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
