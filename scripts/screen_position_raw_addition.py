#!/usr/bin/env python3
"""Test raw mean-difference activation addition for saved position directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.screen_generation_subspaces import read_prompt_slice
from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.intervention import activation_addition_input_hook
from swift_abliteration.live_model import capture_last_token_logits, validate_live_model
from swift_abliteration.metrics import summarize
from swift_abliteration.refusal_scoring import (
    DEFAULT_REFUSAL_OPENERS,
    refusal_scores_from_logits,
    resolve_single_token_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--base-arm-metadata", type=Path, required=True)
    parser.add_argument("--safe-path", default="data/prepared/xstest-safe/selection.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--safe-limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coefficient", type=float, action="append")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    return parser.parse_args()


def raw_masked_difference(activations, candidate: dict):
    layer = int(candidate["layer"])
    position_count = int(activations[f"harmful_resid_pre_layer_{layer}"].shape[1])
    position_index = int(candidate["position"]) + position_count
    if not 0 <= position_index < position_count:
        raise ValueError("Candidate position is outside the saved activation tensor.")
    harmful = activations[f"harmful_resid_pre_layer_{layer}"][:, position_index]
    harmless = activations[f"harmless_resid_pre_layer_{layer}"][:, position_index]
    direction = harmful.float().mean(0) - harmless.float().mean(0)
    direction[candidate["massive_activation_mask"]["selected_indices"]] = 0
    return direction


def score_summary(values: np.ndarray) -> dict:
    return {
        **summarize(values),
        "positive_count": int((values > 0).sum()),
        "positive_rate": float((values > 0).mean()),
    }


def main() -> int:
    args = parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    coefficients = args.coefficient or [1.0]
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    if args.safe_limit <= 0 or args.batch_size <= 0:
        raise ValueError("Limits and batch size must be positive.")
    if any(value == 0 for value in coefficients):
        raise ValueError("Addition coefficients must be nonzero.")
    args.output_dir.mkdir(parents=True)

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    manifest = json.loads(args.direction_manifest.read_text(encoding="utf-8"))
    base_arm = json.loads(args.base_arm_metadata.read_text(encoding="utf-8"))
    directions = load_file(str(args.directions), device="cpu")
    activations = load_file(str(args.activations), device="cpu")
    candidate_keys = sorted(manifest["candidates"])

    safe_rows, safe_sha256 = read_prompt_slice(args.safe_path, 0, args.safe_limit)
    safe_labels = base_arm["groups"]["xstest_safe"]["opening_refusal_labels"]
    if len(safe_labels) != len(safe_rows):
        raise ValueError("Base safe labels do not match the safe prompt count.")
    answerable_indices = [
        index for index, refused in enumerate(safe_labels) if not bool(refused)
    ]
    safe_prompts = [safe_rows[index]["text"] for index in answerable_indices]

    processor = AutoProcessor.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    tokenizer = getattr(processor, "tokenizer", processor)
    refusal_token_ids = resolve_single_token_ids(tokenizer, DEFAULT_REFUSAL_OPENERS)
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
            model,
            processor,
            safe_prompts,
            args.system_prompt,
            args.batch_size,
        )
    )
    base_scores = refusal_scores_from_logits(base_logits, refusal_token_ids).float().numpy()
    results = {}
    for ordinal, key in enumerate(candidate_keys, start=1):
        candidate = manifest["candidates"][key]
        raw_direction = raw_masked_difference(activations, candidate)
        raw_norm = float(raw_direction.norm())
        normalized = raw_direction / raw_direction.norm().clamp_min(1e-12)
        cosine = float(torch.dot(normalized, directions[key].float()))
        if cosine < 0.999:
            raise RuntimeError(f"Raw and saved directions disagree for {key}: {cosine}")
        coefficient_results = {}
        for coefficient in coefficients:
            with activation_addition_input_hook(
                model,
                raw_direction,
                int(candidate["layer"]),
                coefficient,
            ) as intervention:
                added_logits = torch.stack(
                    capture_last_token_logits(
                        model,
                        processor,
                        safe_prompts,
                        args.system_prompt,
                        args.batch_size,
                    )
                )
            added_scores = (
                refusal_scores_from_logits(added_logits, refusal_token_ids)
                .float()
                .numpy()
            )
            coefficient_results[str(coefficient)] = {
                "mean_score_increase": float(added_scores.mean() - base_scores.mean()),
                "added_scores": score_summary(added_scores),
                "intervention": intervention,
            }
        results[key] = {
            "position": int(candidate["position"]),
            "source_layer": int(candidate["layer"]),
            "raw_masked_mean_difference_norm": raw_norm,
            "cosine_with_saved_unit_direction": cosine,
            "coefficients": coefficient_results,
        }
        print(
            json.dumps(
                {
                    "candidate": ordinal,
                    "candidate_count": len(candidate_keys),
                    "key": key,
                    "raw_norm": raw_norm,
                    "maximum_added_mean_score": max(
                        value["added_scores"]["mean"]
                        for value in coefficient_results.values()
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ranked_by_coefficient = {}
    for coefficient in coefficients:
        label = str(coefficient)
        ranked_by_coefficient[label] = sorted(
            candidate_keys,
            key=lambda key: results[key]["coefficients"][label]["added_scores"][
                "mean"
            ],
            reverse=True,
        )
    report = {
        "experiment": "raw_mean_difference_refusal_addition",
        "method_note": (
            "The reference implementation adds the unnormalized mean-difference "
            "vector. Projection-based removal is scale invariant, but activation "
            "addition is not."
        ),
        "config": args.config,
        "direction_file_sha256": sha256_file(args.directions),
        "activation_file_sha256": sha256_file(args.activations),
        "direction_manifest_sha256": sha256_file(args.direction_manifest),
        "base_arm_metadata_sha256": sha256_file(args.base_arm_metadata),
        "safe_source": {
            "path": args.safe_path,
            "sha256": safe_sha256,
            "requested_count": len(safe_rows),
            "base_answerable_count": len(answerable_indices),
            "base_answerable_indices": answerable_indices,
        },
        "refusal_openers": list(DEFAULT_REFUSAL_OPENERS),
        "base_scores": score_summary(base_scores),
        "coefficients": coefficients,
        "candidate_count": len(candidate_keys),
        "results": results,
        "ranked_by_coefficient": ranked_by_coefficient,
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_report": False,
        "system": system_record(),
    }
    write_json(args.output_dir / "raw_addition_report.json", report)
    print(json.dumps({"status": "complete", "candidate_count": len(candidate_keys)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
