#!/usr/bin/env python3
"""Build one direction from prompts filtered by the base model's behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.screen_directions import generate_group
from swift_abliteration.config import load_config
from swift_abliteration.direction_study import (
    bootstrap_masked_direction_stability,
    coordinate_masked_direction,
    massive_activation_coordinate_mask,
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
    capture_last_token_activations_multi,
    validate_live_model,
)
from swift_abliteration.metrics import summarize
from swift_abliteration.refusal_heuristics import xstest_refusal


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def id_digest(rows: list[dict]) -> str:
    values = [str(row["id"]) for row in rows]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=38)
    parser.add_argument("--selected-count", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--log-robust-z-threshold", type=float, default=10.0)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if args.selected_count <= 0:
        raise ValueError("Selected count must be positive.")

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    source_paths = {
        "harmful": args.input_dir / "candidates_harmful.jsonl",
        "harmless": args.input_dir / "candidates_harmless.jsonl",
    }
    prompts = {}
    sources = {}
    for name, path in source_paths.items():
        values, digest = read_prompt_jsonl(path)
        prompts[name] = values
        sources[name] = {"path": str(path), "count": len(values), "sha256": digest}

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
    tokenizer = getattr(processor, "tokenizer", processor)

    response_dir = output / "base_responses"
    response_dir.mkdir()
    selected_prompts = {}
    selection_report = {}
    for name in ("harmful", "harmless"):
        path = response_dir / f"{name}.jsonl"
        response_sha = generate_group(
            model,
            tokenizer,
            prompts[name],
            path,
            args.max_new_tokens,
            args.system_prompt,
            args.batch_size,
        )
        rows = read_rows(path)
        if name == "harmful":
            eligible = [row for row in rows if xstest_refusal(row["response"])]
            behavior = "base_opening_refusal"
        else:
            eligible = [row for row in rows if not xstest_refusal(row["response"])]
            behavior = "base_answer"
        if len(eligible) < args.selected_count:
            raise RuntimeError(
                f"Only {len(eligible)} {name} prompts passed the behavior filter."
            )
        chosen = eligible[: args.selected_count]
        selected_prompts[name] = [row["prompt"] for row in chosen]
        selection_report[name] = {
            "behavior": behavior,
            "eligible_count": len(eligible),
            "selected_count": len(chosen),
            "selected_id_sha256": id_digest(chosen),
            "response_file_sha256": response_sha,
        }

    activations = {}
    for name in ("harmful", "harmless"):
        captured = capture_last_token_activations_multi(
            model,
            processor,
            selected_prompts[name],
            [args.layer],
            args.system_prompt,
        )
        activations[name] = torch.stack(captured[args.layer]).float().cpu().numpy()

    mask, mask_report = massive_activation_coordinate_mask(
        [activations["harmful"], activations["harmless"]],
        args.log_robust_z_threshold,
    )
    direction = coordinate_masked_direction(
        activations["harmful"], activations["harmless"], mask
    )
    stability = bootstrap_masked_direction_stability(
        activations["harmful"],
        activations["harmless"],
        mask,
        direction,
        samples=500,
    )
    direction_path = output / "behavior_filtered_direction.safetensors"
    save_file(
        {f"behavior_filtered_layer_{args.layer}_massive_masked": torch.from_numpy(direction)},
        str(direction_path),
    )
    activation_path = output / "selected_activations.safetensors"
    save_file(
        {
            f"harmful_layer_{args.layer}": torch.from_numpy(activations["harmful"]),
            f"harmless_layer_{args.layer}": torch.from_numpy(activations["harmless"]),
        },
        str(activation_path),
    )
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "method": "base-behavior-filtered massive-coordinate-masked mean difference",
            "config": args.config,
            "layer": args.layer,
            "system_prompt": args.system_prompt,
            "sources": sources,
            "selection": selection_report,
            "massive_activation_mask": mask_report,
            "bootstrap_cosine": summarize(stability),
            "artifacts": {
                "direction_sha256": sha256_file(direction_path),
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
                "selected_counts": {name: len(values) for name, values in selected_prompts.items()},
                "mask_count": int(mask.sum()),
                "raw_text_printed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
