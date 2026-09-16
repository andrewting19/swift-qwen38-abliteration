#!/usr/bin/env python3
"""Re-capture a fixed behavior-filtered prompt set at several layers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
    capture_last_token_activations_multi,
    validate_live_model,
)
from swift_abliteration.metrics import summarize
from swift_abliteration.refusal_heuristics import xstest_refusal


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def select_rows(path: Path, *, refused: bool, count: int) -> list[dict]:
    rows = [row for row in read_rows(path) if xstest_refusal(row["response"]) == refused]
    if len(rows) < count:
        raise RuntimeError(f"Only {len(rows)} rows pass the behavior filter: {path}")
    return rows[:count]


def row_ids_sha256(rows: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps([str(row["id"]) for row in rows], separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--selected-count", type=int, default=128)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--log-robust-z-threshold", type=float, default=10.0)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    layers = sorted(set(args.layer))
    if args.selected_count <= 0 or not layers:
        raise ValueError("Selected count and layers must be positive.")

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    selected = {
        "harmful": select_rows(
            args.response_dir / "harmful.jsonl", refused=True, count=args.selected_count
        ),
        "harmless": select_rows(
            args.response_dir / "harmless.jsonl", refused=False, count=args.selected_count
        ),
    }
    cfg = load_config(args.config)
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
        captured[name] = capture_last_token_activations_multi(
            model,
            processor,
            [row["prompt"] for row in rows],
            layers,
            args.system_prompt,
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    directions = {}
    activation_tensors = {}
    layer_reports = {}
    for layer in layers:
        harmful = torch.stack(captured["harmful"][layer]).float().cpu().numpy()
        harmless = torch.stack(captured["harmless"][layer]).float().cpu().numpy()
        mask, mask_report = massive_activation_coordinate_mask(
            [harmful, harmless], args.log_robust_z_threshold
        )
        direction = coordinate_masked_direction(harmful, harmless, mask)
        stability = bootstrap_masked_direction_stability(
            harmful, harmless, mask, direction, samples=500
        )
        directions[f"behavior_filtered_layer_{layer}_massive_masked"] = (
            torch.from_numpy(direction)
        )
        activation_tensors[f"harmful_layer_{layer}"] = torch.from_numpy(harmful)
        activation_tensors[f"harmless_layer_{layer}"] = torch.from_numpy(harmless)
        layer_reports[str(layer)] = {
            "massive_activation_mask": mask_report,
            "bootstrap_cosine": summarize(stability),
        }

    direction_path = output / "behavior_filtered_multilayer_directions.safetensors"
    activation_path = output / "behavior_filtered_multilayer_activations.safetensors"
    save_file(directions, str(direction_path))
    save_file(activation_tensors, str(activation_path))
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "method": "base-behavior-filtered massive-coordinate-masked mean difference",
            "layers": layer_reports,
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
            "uses_evaluation_rows": False,
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_manifest": False,
            "system": system_record(),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "layers": layers,
                "selected_count_per_group": args.selected_count,
                "raw_text_printed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
