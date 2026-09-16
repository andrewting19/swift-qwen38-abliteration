#!/usr/bin/env python3
"""Capture masked harmless means at selected prompt positions for every layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.derive_behavior_filtered_multilayer import row_ids_sha256, select_rows
from swift_abliteration.config import load_config
from swift_abliteration.direction_study import massive_activation_coordinate_mask
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--position", type=int, action="append", required=True)
    parser.add_argument("--selected-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-robust-z-threshold", type=float, default=10.0)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    positions = sorted(set(args.position))
    if not positions or any(position >= 0 for position in positions):
        raise ValueError("Positions must be distinct negative offsets.")
    if min(args.selected_count, args.batch_size) <= 0:
        raise ValueError("Selected count and batch size must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    harmless_path = args.response_dir / "harmless.jsonl"
    rows = select_rows(harmless_path, refused=False, count=args.selected_count)
    cfg = load_config(args.config)
    layers = list(range(cfg.model.num_layers))
    position_count = abs(min(positions))
    args.output_dir.mkdir(parents=True)

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
    captured = capture_last_positions_resid_pre_multi(
        model,
        processor,
        [row["prompt"] for row in rows],
        layers,
        position_count,
        args.system_prompt,
        args.batch_size,
    )

    tensors = {}
    reports = {}
    for layer in layers:
        values = torch.stack(captured[layer]).float().cpu().numpy()
        for position in positions:
            position_index = position + position_count
            position_values = values[:, position_index, :]
            mask, mask_report = massive_activation_coordinate_mask(
                [position_values], args.log_robust_z_threshold
            )
            mean = position_values.mean(axis=0)
            mean[mask] = 0.0
            norm = float((mean @ mean) ** 0.5)
            if not norm > 0.0:
                raise ValueError(
                    f"Harmless mean has zero length at layer {layer}, position {position}."
                )
            key = f"harmless_position_minus_{abs(position)}_layer_{layer}_masked_mean"
            tensors[key] = torch.from_numpy((mean / norm).astype("float32"))
            reports[key] = {
                "layer": layer,
                "position": position,
                "massive_activation_mask": mask_report,
            }

    tensor_path = args.output_dir / "harmless_layer_means.safetensors"
    save_file(tensors, str(tensor_path))
    write_json(
        args.output_dir / "harmless_layer_means_report.json",
        {
            "schema_version": 1,
            "experiment": "behavior_filtered_harmless_layer_means",
            "config": args.config,
            "model": {"id": cfg.model.id, "revision": cfg.model.revision},
            "layers": layers,
            "positions": positions,
            "capture_site": "transformer_block_resid_pre",
            "selected_count": len(rows),
            "selection_id_sha256": row_ids_sha256(rows),
            "source_sha256": sha256_file(harmless_path),
            "tensor_sha256": sha256_file(tensor_path),
            "tensors": reports,
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_report": False,
            "system": system_record(),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "tensor_count": len(tensors),
                "raw_text_printed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
