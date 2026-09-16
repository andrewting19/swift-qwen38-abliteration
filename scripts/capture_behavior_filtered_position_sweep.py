#!/usr/bin/env python3
"""Extract rank-1 directions from several prompt-suffix positions and layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.derive_behavior_filtered_multilayer import row_ids_sha256, select_rows
from swift_abliteration.config import load_config
from swift_abliteration.direction_study import (
    bootstrap_masked_direction_stability,
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
from swift_abliteration.metrics import summarize


DEFAULT_LAYERS = (24, 32, 38, 44, 52)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--position-count", type=int, default=16)
    parser.add_argument("--selected-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--log-robust-z-threshold", type=float, default=10.0)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    if min(
        args.position_count,
        args.selected_count,
        args.batch_size,
        args.bootstrap_samples,
    ) <= 0:
        raise ValueError("Counts and batch size must be positive.")

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    layers = sorted(set(args.layer or DEFAULT_LAYERS))
    selected = {
        "harmful": select_rows(
            args.response_dir / "harmful.jsonl",
            refused=True,
            count=args.selected_count,
        ),
        "harmless": select_rows(
            args.response_dir / "harmless.jsonl",
            refused=False,
            count=args.selected_count,
        ),
    }
    cfg = load_config(args.config)
    if any(layer < 0 or layer >= cfg.model.num_layers for layer in layers):
        raise ValueError("A capture layer is outside the language model.")
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

    captured = {}
    for name, rows in selected.items():
        captured[name] = capture_last_positions_resid_pre_multi(
            model,
            processor,
            [row["prompt"] for row in rows],
            layers,
            args.position_count,
            args.system_prompt,
            args.batch_size,
        )

    directions = {}
    activation_tensors = {}
    candidate_reports = {}
    for layer in layers:
        harmful = torch.stack(captured["harmful"][layer]).float().cpu().numpy()
        harmless = torch.stack(captured["harmless"][layer]).float().cpu().numpy()
        activation_tensors[f"harmful_resid_pre_layer_{layer}"] = torch.from_numpy(
            harmful
        )
        activation_tensors[f"harmless_resid_pre_layer_{layer}"] = torch.from_numpy(
            harmless
        )
        for position_index in range(args.position_count):
            position = position_index - args.position_count
            harmful_position = harmful[:, position_index, :]
            harmless_position = harmless[:, position_index, :]
            mask, mask_report = massive_activation_coordinate_mask(
                [harmful_position, harmless_position],
                args.log_robust_z_threshold,
            )
            direction = coordinate_masked_direction(
                harmful_position,
                harmless_position,
                mask,
            )
            stability = bootstrap_masked_direction_stability(
                harmful_position,
                harmless_position,
                mask,
                direction,
                samples=args.bootstrap_samples,
            )
            key = (
                f"behavior_filtered_position_minus_{abs(position)}_"
                f"layer_{layer}_massive_masked"
            )
            directions[key] = torch.from_numpy(direction)
            candidate_reports[key] = {
                "position": position,
                "position_order": "negative offset from final prompt token",
                "layer": layer,
                "capture_site": "transformer_block_resid_pre",
                "massive_activation_mask": mask_report,
                "bootstrap_cosine": summarize(stability),
            }

    direction_path = args.output_dir / "directions.safetensors"
    activation_path = args.output_dir / "activations.safetensors"
    save_file(directions, str(direction_path))
    save_file(activation_tensors, str(activation_path))
    manifest = {
        "schema_version": 1,
        "experiment": "behavior_filtered_prompt_suffix_position_sweep",
        "method": "massive-coordinate-masked difference of means",
        "config": args.config,
        "model": {"id": cfg.model.id, "revision": cfg.model.revision},
        "layers": layers,
        "positions": list(range(-args.position_count, 0)),
        "capture_site": "transformer_block_resid_pre",
        "candidate_count": len(directions),
        "candidates": candidate_reports,
        "selection": {
            name: {
                "count": len(rows),
                "id_sha256": row_ids_sha256(rows),
                "source_sha256": sha256_file(args.response_dir / f"{name}.jsonl"),
            }
            for name, rows in selected.items()
        },
        "artifacts": {
            "directions_sha256": sha256_file(direction_path),
            "activations_sha256": sha256_file(activation_path),
        },
        "uses_evaluation_rows_for_direction": False,
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_manifest": False,
        "system": system_record(),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_count": len(directions),
                "layers": layers,
                "positions": list(range(-args.position_count, 0)),
                "raw_text_printed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
