#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.screen_directions import GROUPS, generate_group
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
    capture_last_token_logits,
    validate_live_model,
)
from swift_abliteration.metrics import forward_kl_from_logits, summarize


def incremental_alpha(current: float, target: float) -> float:
    """Return beta where (I-beta P)(I-current P) = I-target P."""
    if not 0.0 <= current < 1.0:
        raise ValueError("Current alpha must be in [0, 1).")
    if not current < target <= 1.0:
        raise ValueError("Target alpha must be greater than current and at most 1.")
    return (target - current) / (1.0 - current)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen increasing projection strengths with one model load."
    )
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--direction-key", action="append", required=True)
    parser.add_argument("--alpha", action="append", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-limit", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    alphas = args.alpha
    if alphas != sorted(set(alphas)) or not alphas or not 0.0 < alphas[0] <= alphas[-1] <= 1.0:
        raise ValueError("Alpha values must be unique, increasing, and in (0, 1].")

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    tensors = load_file(str(args.directions), device="cpu")
    missing = [key for key in args.direction_key if key not in tensors]
    if missing:
        raise KeyError(f"Missing direction keys: {missing}")
    directions = torch.stack([tensors[key] for key in args.direction_key])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected_groups = ("standard_harmful", "matched_harmful", "standard_harmless")
    prompts = {}
    prompt_sources = {}
    for name in selected_groups:
        path = GROUPS[name]
        values, digest = read_prompt_jsonl(path)
        prompts[name] = values[: args.group_limit]
        prompt_sources[name] = {
            "path": path,
            "sha256": digest,
            "source_count": len(values),
            "used_count": len(prompts[name]),
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
    base_logits = torch.stack(
        capture_last_token_logits(
            model, processor, prompts["standard_harmless"], args.system_prompt
        )
    ).numpy()

    current = 0.0
    arms = {}
    for target in alphas:
        step = incremental_alpha(current, target)
        intervention = apply_layerwise_runtime_edit(
            model,
            cfg,
            {index: directions for index in range(cfg.model.num_layers)},
            step,
            embedding_direction=None,
        )
        arm_name = f"alpha_{target:.3f}".replace(".", "_")
        arm_dir = args.output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=False)
        response_hashes = {}
        tokenizer = getattr(processor, "tokenizer", processor)
        for group in ("standard_harmful", "matched_harmful"):
            response_hashes[group] = generate_group(
                model,
                tokenizer,
                prompts[group],
                arm_dir / f"{group}.jsonl",
                args.max_new_tokens,
                args.system_prompt,
                args.batch_size,
            )
        candidate_logits = torch.stack(
            capture_last_token_logits(
                model, processor, prompts["standard_harmless"], args.system_prompt
            )
        ).numpy()
        np.savez_compressed(arm_dir / "standard_harmless_logits.npz", logits=candidate_logits)
        kl_report = {
            "metric": "last_prompt_token_forward_kl_nats",
            "summary": summarize(forward_kl_from_logits(base_logits, candidate_logits)),
        }
        write_json(arm_dir / "kl.json", kl_report)
        arms[arm_name] = {
            "target_alpha": target,
            "incremental_alpha": step,
            "response_sha256": response_hashes,
            "logits_sha256": sha256_file(arm_dir / "standard_harmless_logits.npz"),
            "kl": kl_report["summary"],
            "intervention": intervention,
        }
        current = target
        print(json.dumps({"arm": arm_name, "kl": kl_report["summary"]}, sort_keys=True))

    write_json(
        args.output_dir / "screen_manifest.json",
        {
            "config": args.config,
            "direction_file": str(args.directions.resolve()),
            "direction_file_sha256": sha256_file(args.directions),
            "direction_keys": args.direction_key,
            "alphas": alphas,
            "method": "sequential_equivalent_projection_steps",
            "prompt_sources": prompt_sources,
            "arms": arms,
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_manifest": False,
            "system": system_record(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
