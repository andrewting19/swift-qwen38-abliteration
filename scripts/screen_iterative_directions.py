#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.screen_directions import GROUPS, run_arm
from swift_abliteration.config import load_config
from swift_abliteration.direction_study import (
    cosine_similarity,
    normalized_average,
    winsorized_direction,
)
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
    capture_last_token_activations_multi,
    validate_live_model,
)


DIRECTION_GROUPS = {
    "standard_harmful": "data/prepared/direction_harmful.jsonl",
    "standard_harmless": "data/prepared/direction_harmless.jsonl",
    "matched_harmful": "data/prepared/matched/direction_harmful.jsonl",
    "matched_harmless": "data/prepared/matched/direction_harmless.jsonl",
}


def orthogonalize(candidate: np.ndarray, basis: list[np.ndarray]) -> tuple[np.ndarray, float]:
    value = np.asarray(candidate, dtype=np.float32).copy()
    for direction in basis:
        value -= float(value @ direction) * direction
    residual_norm = float(np.linalg.norm(value))
    if not np.isfinite(residual_norm) or residual_norm <= 1e-6:
        raise ValueError("The new direction is contained in the existing basis.")
    return (value / residual_norm).astype(np.float32), residual_norm


def rows_to_numpy(values) -> np.ndarray:
    return np.stack([value.numpy() for value in values]).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure and screen sequential refusal directions on an edited model."
    )
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--primary-layer", type=int, default=52)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--winsor-quantile", type=float, default=0.995)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-limit", type=int, default=16)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if args.max_rank < 2:
        raise ValueError("max-rank must be at least 2.")
    if args.group_limit <= 0:
        raise ValueError("group-limit must be positive.")

    import torch
    from safetensors.numpy import save_file
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    if not cfg.edit.include_embedding:
        raise ValueError("The iterative screen requires the configured embedding edit.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_tensors = load_file(str(args.directions), device="cpu")
    primary_keys = [
        f"standard_layer_{args.primary_layer}_winsor_995",
        f"matched_layer_{args.primary_layer}_winsor_995",
    ]
    missing = [key for key in primary_keys if key not in source_tensors]
    if missing:
        raise KeyError(f"Missing primary directions: {missing}")
    primary = normalized_average(
        [source_tensors[key].numpy() for key in primary_keys]
    ).astype(np.float32)

    evaluation_groups = {}
    evaluation_sources = {}
    for name, path in GROUPS.items():
        values, digest = read_prompt_jsonl(path)
        evaluation_groups[name] = values[: args.group_limit]
        evaluation_sources[name] = {
            "path": path,
            "sha256": digest,
            "source_count": len(values),
            "used_count": len(evaluation_groups[name]),
        }
    direction_prompts = {}
    direction_sources = {}
    for name, path in DIRECTION_GROUPS.items():
        values, digest = read_prompt_jsonl(path)
        direction_prompts[name] = values
        direction_sources[name] = {
            "path": path,
            "sha256": digest,
            "count": len(values),
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
    all_layers = list(range(cfg.edit.first_layer, cfg.edit.last_layer + 1))

    basis = [primary]
    saved_directions = {"iterative_direction_1": primary}
    saved_activations = {}
    arms = {}
    direction_quality = {
        "1": {
            "source": "normalized standard-plus-matched winsor-995 consensus",
            "primary_keys": primary_keys,
        }
    }

    edit_record = apply_layerwise_runtime_edit(
        model,
        cfg,
        {index: torch.from_numpy(primary) for index in all_layers},
        1.0,
        embedding_direction=torch.from_numpy(primary),
    )
    arms["iterative_rank1"] = run_arm(
        model,
        processor,
        evaluation_groups,
        args.output_dir / "iterative_rank1",
        args.max_new_tokens,
        args.system_prompt,
        args.batch_size,
    )
    arms["iterative_rank1"]["intervention"] = edit_record

    for rank in range(2, args.max_rank + 1):
        captured = {}
        for name, prompts in direction_prompts.items():
            values = capture_last_token_activations_multi(
                model,
                processor,
                prompts,
                [args.primary_layer],
                args.system_prompt,
            )[args.primary_layer]
            array = rows_to_numpy(values)
            captured[name] = array
            saved_activations[f"rank{rank}_before_{name}"] = array

        standard, standard_threshold, _ = winsorized_direction(
            captured["standard_harmful"],
            captured["standard_harmless"],
            args.winsor_quantile,
        )
        matched, matched_threshold, _ = winsorized_direction(
            captured["matched_harmful"],
            captured["matched_harmless"],
            args.winsor_quantile,
        )
        consensus = normalized_average([standard, matched])
        next_direction, residual_norm = orthogonalize(consensus, basis)
        direction_quality[str(rank)] = {
            "source": "edited-model standard-plus-matched winsor-995 consensus",
            "standard_matched_cosine": cosine_similarity(standard, matched),
            "cosine_to_previous_basis": [
                cosine_similarity(consensus, direction) for direction in basis
            ],
            "orthogonal_residual_norm": residual_norm,
            "standard_winsor_threshold": standard_threshold,
            "matched_winsor_threshold": matched_threshold,
        }
        basis.append(next_direction)
        saved_directions[f"iterative_direction_{rank}"] = next_direction
        current = torch.from_numpy(next_direction)
        edit_record = apply_layerwise_runtime_edit(
            model,
            cfg,
            {index: current for index in all_layers},
            1.0,
            embedding_direction=current,
        )
        arm = f"iterative_rank{rank}"
        arms[arm] = run_arm(
            model,
            processor,
            evaluation_groups,
            args.output_dir / arm,
            args.max_new_tokens,
            args.system_prompt,
            args.batch_size,
        )
        arms[arm]["intervention"] = edit_record

    directions_path = args.output_dir / "iterative_directions.safetensors"
    activations_path = args.output_dir / "iterative_training_activations.safetensors"
    save_file(saved_directions, str(directions_path))
    save_file(saved_activations, str(activations_path))
    manifest = {
        "config": args.config,
        "source_directions": str(args.directions.resolve()),
        "source_directions_sha256": sha256_file(args.directions),
        "primary_layer": args.primary_layer,
        "max_rank": args.max_rank,
        "winsor_quantile": args.winsor_quantile,
        "direction_sources": direction_sources,
        "evaluation_sources": evaluation_sources,
        "direction_quality": direction_quality,
        "basis_gram_max_error": float(
            np.max(np.abs(np.stack(basis) @ np.stack(basis).T - np.eye(len(basis))))
        ),
        "directions_sha256": sha256_file(directions_path),
        "training_activations_sha256": sha256_file(activations_path),
        "arms": arms,
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_manifest": False,
        "system": system_record(),
    }
    write_json(args.output_dir / "screen_manifest.json", manifest)
    print(json.dumps({"status": "complete", "arm_count": len(arms)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
