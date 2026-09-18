#!/usr/bin/env python3
"""Build a behavior-filtered rank-2 refusal basis independently at every layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.derive_behavior_filtered_multilayer import row_ids_sha256, select_rows
from swift_abliteration.config import load_config
from swift_abliteration.direction_study import (
    coordinate_masked_direction,
    massive_activation_coordinate_mask,
)
from swift_abliteration.gpu_support import (
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.live_model import (
    capture_last_positions_resid_pre_multi,
    validate_live_model,
)


def orthonormal_rows(vectors: list[np.ndarray]) -> np.ndarray:
    """Return a QR-orthonormal row basis for the supplied vectors."""
    values = np.stack(vectors).astype(np.float64)
    basis = np.linalg.qr(values.T, mode="reduced")[0].T
    return basis.astype(np.float32)


def split_half_cosine(
    harmful: np.ndarray, harmless: np.ndarray, mask: np.ndarray
) -> float:
    midpoint_harmful = len(harmful) // 2
    midpoint_harmless = len(harmless) // 2
    first = coordinate_masked_direction(
        harmful[:midpoint_harmful], harmless[:midpoint_harmless], mask
    )
    second = coordinate_masked_direction(
        harmful[midpoint_harmful:], harmless[midpoint_harmless:], mask
    )
    return float(first @ second)


def build_layer_basis(
    harmful_by_position: list[np.ndarray],
    harmless_by_position: list[np.ndarray],
    masks: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    directions = [
        coordinate_masked_direction(harmful, harmless, mask)
        for harmful, harmless, mask in zip(
            harmful_by_position, harmless_by_position, masks, strict=True
        )
    ]
    return orthonormal_rows(directions), directions


def persistent_coordinate_mask(
    masks: list[np.ndarray], frequency: float
) -> tuple[np.ndarray, int]:
    if not masks or not 0.0 < frequency <= 1.0:
        raise ValueError("Masks and a frequency in (0, 1] are required.")
    values = [np.asarray(mask, dtype=bool) for mask in masks]
    if any(mask.shape != values[0].shape for mask in values[1:]):
        raise ValueError("All coordinate masks must have the same shape.")
    required = int(np.ceil(frequency * len(values)))
    return np.stack(values).sum(axis=0) >= required, required


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--position", type=int, action="append", required=True)
    parser.add_argument("--selected-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-robust-z-threshold", type=float, default=10.0)
    parser.add_argument("--persistent-mask-frequency", type=float, default=0.9)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    positions = sorted(set(args.position))
    if len(positions) != 2 or any(position >= 0 for position in positions):
        raise ValueError("Exactly two distinct negative positions are required.")
    if min(args.selected_count, args.batch_size) <= 0:
        raise ValueError("Selected count and batch size must be positive.")
    if not 0.0 < args.persistent_mask_frequency <= 1.0:
        raise ValueError("Persistent mask frequency must be in (0, 1].")
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    source_paths = {
        "harmful": args.response_dir / "harmful.jsonl",
        "harmless": args.response_dir / "harmless.jsonl",
    }
    selected = {
        "harmful": select_rows(
            source_paths["harmful"], refused=True, count=args.selected_count
        ),
        "harmless": select_rows(
            source_paths["harmless"], refused=False, count=args.selected_count
        ),
    }
    cfg = load_config(args.config)
    layers = list(range(cfg.model.num_layers))
    position_count = abs(min(positions))

    processor = AutoProcessor.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    validate_live_model(model, cfg)
    captured = {}
    for group, rows in selected.items():
        captured[group] = capture_last_positions_resid_pre_multi(
            model,
            processor,
            [row["prompt"] for row in rows],
            layers,
            position_count,
            args.system_prompt,
            args.batch_size,
        )

    detected_masks = {}
    detector_reports = {}
    for layer in layers:
        values = {
            group: torch.stack(captured[group][layer]).float().cpu().numpy()
            for group in ("harmful", "harmless")
        }
        for position in positions:
            position_index = position + position_count
            harmful = values["harmful"][:, position_index, :]
            harmless = values["harmless"][:, position_index, :]
            mask, mask_report = massive_activation_coordinate_mask(
                [harmful, harmless], args.log_robust_z_threshold
            )
            detected_masks[(layer, position)] = mask
            detector_reports[(layer, position)] = mask_report

    persistent_mask, required_detections = persistent_coordinate_mask(
        list(detected_masks.values()), args.persistent_mask_frequency
    )
    if not bool(persistent_mask.any()):
        raise ValueError("No coordinate passes the persistent massive-mask rule.")

    bases = []
    diagnostics = {}
    previous_basis = None
    for layer in layers:
        values = {
            group: torch.stack(captured[group][layer]).float().cpu().numpy()
            for group in ("harmful", "harmless")
        }
        harmful_by_position = []
        harmless_by_position = []
        masks = []
        position_reports = []
        for position in positions:
            position_index = position + position_count
            harmful = values["harmful"][:, position_index, :]
            harmless = values["harmless"][:, position_index, :]
            harmful_by_position.append(harmful)
            harmless_by_position.append(harmless)
            masks.append(persistent_mask)
            position_reports.append(
                {
                    "position": position,
                    "local_massive_activation_detection": detector_reports[
                        (layer, position)
                    ],
                    "split_half_cosine": split_half_cosine(
                        harmful, harmless, persistent_mask
                    ),
                }
            )
        basis, raw_directions = build_layer_basis(
            harmful_by_position, harmless_by_position, masks
        )
        bases.append(basis)
        diagnostics[str(layer)] = {
            "positions": position_reports,
            "raw_direction_cosine": float(raw_directions[0] @ raw_directions[1]),
            "previous_layer_principal_cosines": None
            if previous_basis is None
            else [
                float(value)
                for value in np.linalg.svd(
                    previous_basis @ basis.T, compute_uv=False
                )
            ],
        }
        previous_basis = basis

    tensor = np.stack(bases).astype(np.float32)
    args.output_dir.mkdir(parents=True)
    tensor_path = args.output_dir / "layerwise_behavior_rank2.safetensors"
    save_file(
        {"layerwise_behavior_rank2": torch.from_numpy(tensor).contiguous()},
        str(tensor_path),
    )
    report = {
        "schema_version": 1,
        "experiment": "behavior_filtered_layer_local_rank2",
        "method": "masked harmful-minus-harmless direction at each target layer and position, followed by QR",
        "config": args.config,
        "model": {"id": cfg.model.id, "revision": cfg.model.revision},
        "layers": layers,
        "positions": positions,
        "shape": list(tensor.shape),
        "capture_site": "transformer_block_resid_pre",
        "persistent_massive_activation_mask": {
            "frequency": args.persistent_mask_frequency,
            "required_detections": required_detections,
            "detector_count": len(detected_masks),
            "selected_count": int(persistent_mask.sum()),
            "selected_indices": np.flatnonzero(persistent_mask).tolist(),
        },
        "selection": {
            group: {
                "count": len(rows),
                "id_sha256": row_ids_sha256(rows),
                "source_sha256": sha256_file(source_paths[group]),
            }
            for group, rows in selected.items()
        },
        "diagnostics": diagnostics,
        "tensor_sha256": sha256_file(tensor_path),
        "uses_evaluation_rows": False,
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_report": False,
        "system": system_record(),
    }
    write_json(args.output_dir / "layerwise_behavior_rank2_report.json", report)
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
