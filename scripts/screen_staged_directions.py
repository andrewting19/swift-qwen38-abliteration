#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.screen_directions import GROUPS, run_arm
from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    read_prompt_jsonl,
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.live_model import (
    apply_layerwise_runtime_edit,
    validate_live_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen a global primary edit followed by a tunable secondary edit."
    )
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--primary-key", action="append", required=True)
    parser.add_argument("--primary-alpha", type=float, default=1.0)
    parser.add_argument("--primary-first-layer", type=int, default=0)
    parser.add_argument("--primary-last-layer", type=int, default=63)
    parser.add_argument("--primary-layer-stride", type=int, default=1)
    parser.add_argument(
        "--primary-no-embedding",
        action="store_true",
        help="Do not apply the primary subspace to the token embedding.",
    )
    parser.add_argument("--secondary-key", action="append", required=True)
    parser.add_argument("--secondary-alpha", type=float, required=True)
    parser.add_argument("--secondary-first-layer", type=int, default=0)
    parser.add_argument("--secondary-last-layer", type=int, default=63)
    parser.add_argument("--secondary-embedding", action="store_true")
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-limit", type=int, default=16)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if not 0.0 <= args.primary_alpha <= 1.0:
        raise ValueError("primary-alpha must be between 0 and 1.")
    if not 0.0 <= args.secondary_alpha <= 1.0:
        raise ValueError("secondary-alpha must be between 0 and 1.")
    if not 0 <= args.primary_first_layer <= args.primary_last_layer:
        raise ValueError("The primary layer range is invalid.")
    if args.primary_layer_stride <= 0:
        raise ValueError("The primary layer stride must be positive.")
    if not 0 <= args.secondary_first_layer <= args.secondary_last_layer:
        raise ValueError("The secondary layer range is invalid.")

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    tensors = load_file(str(args.directions), device="cpu")
    keys = [*args.primary_key, *args.secondary_key]
    missing = [key for key in keys if key not in tensors]
    if missing:
        raise KeyError(f"Missing direction keys: {missing}")
    primary = torch.stack([tensors[key] for key in args.primary_key])
    secondary = torch.stack([tensors[key] for key in args.secondary_key])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    prompt_groups = {}
    prompt_sources = {}
    for name, path in GROUPS.items():
        values, digest = read_prompt_jsonl(path)
        prompt_groups[name] = values[: args.group_limit]
        prompt_sources[name] = {
            "path": path,
            "sha256": digest,
            "source_count": len(values),
            "used_count": len(prompt_groups[name]),
        }

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
    primary_layers = list(
        range(
            args.primary_first_layer,
            args.primary_last_layer + 1,
            args.primary_layer_stride,
        )
    )
    primary_record = apply_layerwise_runtime_edit(
        model,
        cfg,
        {index: primary for index in primary_layers},
        args.primary_alpha,
        embedding_direction=None if args.primary_no_embedding else primary,
    )
    secondary_layers = list(
        range(args.secondary_first_layer, args.secondary_last_layer + 1)
    )
    secondary_record = apply_layerwise_runtime_edit(
        model,
        cfg,
        {index: secondary for index in secondary_layers},
        args.secondary_alpha,
        embedding_direction=secondary if args.secondary_embedding else None,
    )
    arm = run_arm(
        model,
        processor,
        prompt_groups,
        args.output_dir / args.arm_name,
        args.max_new_tokens,
        args.system_prompt,
        args.batch_size,
    )
    arm["primary_intervention"] = primary_record
    arm["secondary_intervention"] = secondary_record
    write_json(
        args.output_dir / "screen_manifest.json",
        {
            "config": args.config,
            "direction_file": str(args.directions.resolve()),
            "direction_file_sha256": sha256_file(args.directions),
            "primary_keys": args.primary_key,
            "primary_alpha": args.primary_alpha,
            "primary_layers": primary_layers,
            "primary_layer_stride": args.primary_layer_stride,
            "primary_embedding": not args.primary_no_embedding,
            "secondary_keys": args.secondary_key,
            "secondary_alpha": args.secondary_alpha,
            "secondary_layers": secondary_layers,
            "secondary_embedding": args.secondary_embedding,
            "prompt_sources": prompt_sources,
            "arm": {args.arm_name: arm},
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_manifest": False,
            "system": system_record(),
        },
    )
    print(json.dumps({"status": "complete", "arm": args.arm_name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
