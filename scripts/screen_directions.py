#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    read_prompt_jsonl,
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.intervention import weight_equivalent_ablation_hooks
from swift_abliteration.live_model import (
    apply_runtime_edit,
    capture_last_token_logits,
    render_prompt,
    text_backbone,
    validate_live_model,
)

GROUPS = {
    "standard_harmful": "data/prepared/evaluation_harmful.jsonl",
    "matched_harmful": "data/prepared/matched/evaluation_harmful.jsonl",
    "standard_harmless": "data/prepared/evaluation_harmless.jsonl",
    "matched_harmless": "data/prepared/matched/evaluation_harmless.jsonl",
    "xstest_safe": "data/prepared/xstest-safe/validation.jsonl",
}


def direction_layer(key: str) -> int:
    match = re.search(r"_layer_(\d+)_", key)
    if match is None:
        raise ValueError(f"Direction key does not contain a layer number: {key}")
    return int(match.group(1))


def generate_group(
    model,
    tokenizer,
    prompts,
    path: Path,
    max_new_tokens: int,
    system_prompt: str,
    batch_size: int,
) -> str:
    import torch

    device = text_backbone(model).embed_tokens.weight.device
    digest = hashlib.sha256()
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        with path.open("w", encoding="utf-8") as handle:
            for start in range(0, len(prompts), batch_size):
                prompt_batch = prompts[start : start + batch_size]
                rendered = [
                    render_prompt(tokenizer, prompt, system_prompt)
                    for prompt in prompt_batch
                ]
                batch = tokenizer(rendered, return_tensors="pt", padding=True)
                batch = {name: value.to(device) for name, value in batch.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **batch,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                    )
                input_width = batch["input_ids"].shape[1]
                for offset, prompt in enumerate(prompt_batch):
                    new_tokens = generated[offset, input_width:]
                    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    line = (
                        json.dumps(
                            {
                                "id": start + offset,
                                "prompt": prompt,
                                "response": response,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    handle.write(line)
                    digest.update(line.encode())
                handle.flush()
    finally:
        tokenizer.padding_side = previous_padding_side
    return digest.hexdigest()


def run_arm(
    model,
    processor,
    prompts: dict[str, list[str]],
    output: Path,
    max_new_tokens: int,
    system_prompt: str,
    batch_size: int,
) -> dict:
    import numpy as np
    import torch

    output.mkdir(parents=True, exist_ok=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    response_hashes = {
        name: generate_group(
            model,
            tokenizer,
            values,
            output / f"{name}.jsonl",
            max_new_tokens,
            system_prompt,
            batch_size,
        )
        for name, values in prompts.items()
    }
    logits_path = output / "standard_harmless_logits.npz"
    logits = torch.stack(
        capture_last_token_logits(
            model, processor, prompts["standard_harmless"], system_prompt
        )
    ).numpy()
    np.savez_compressed(logits_path, logits=logits)
    return {
        "response_sha256": response_hashes,
        "logits_sha256": sha256_file(logits_path),
        "logit_shape": list(logits.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--directions", required=True)
    parser.add_argument("--direction-key", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--intervention-mode",
        choices=("module-hooks", "in-memory-weight-edit"),
        default="module-hooks",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Run only the requested reversible candidate arms.",
    )
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    prompt_groups = {}
    prompt_sources = {}
    for name, path in GROUPS.items():
        values, digest = read_prompt_jsonl(path)
        prompt_groups[name] = values
        prompt_sources[name] = {"path": path, "count": len(values), "sha256": digest}
    directions = load_file(args.directions, device="cpu")
    missing = [key for key in args.direction_key if key not in directions]
    if missing:
        raise ValueError(f"Direction keys are not present: {missing}")

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
    intervention_layers = list(range(cfg.edit.first_layer, cfg.edit.last_layer + 1))
    if args.intervention_mode == "in-memory-weight-edit" and (
        not args.skip_base or len(args.direction_key) != 1
    ):
        raise RuntimeError(
            "An in-memory weight edit requires --skip-base and exactly one direction."
        )
    arms = {}
    if not args.skip_base:
        arms["base"] = run_arm(
            model,
            processor,
            prompt_groups,
            output / "base",
            args.max_new_tokens,
            args.system_prompt,
            args.batch_size,
        )
    if args.intervention_mode == "in-memory-weight-edit":
        key = args.direction_key[0]
        measured_layer = direction_layer(key)
        intervention_record = apply_runtime_edit(
            model,
            cfg,
            directions[key],
            args.alpha,
            generation_uses_mtp=False,
        )
        arms[key] = run_arm(
            model,
            processor,
            prompt_groups,
            output / key,
            args.max_new_tokens,
            args.system_prompt,
            args.batch_size,
        )
        arms[key]["direction_layer"] = measured_layer
        arms[key]["intervention_layers"] = intervention_layers
        arms[key]["intervention"] = intervention_record
    for key in args.direction_key if args.intervention_mode == "module-hooks" else ():
        measured_layer = direction_layer(key)
        with weight_equivalent_ablation_hooks(
            model,
            cfg,
            directions[key],
            args.alpha,
            generation_uses_mtp=False,
        ) as intervention_record:
            arms[key] = run_arm(
                model,
                processor,
                prompt_groups,
                output / key,
                args.max_new_tokens,
                args.system_prompt,
                args.batch_size,
            )
        arms[key]["direction_layer"] = measured_layer
        arms[key]["intervention_layers"] = intervention_layers
        arms[key]["intervention"] = intervention_record
        arms[key]["alpha"] = args.alpha
    write_json(
        output / "screen_manifest.json",
        {
            "config": args.config,
            "direction_file": str(Path(args.directions).resolve()),
            "direction_file_sha256": sha256_file(args.directions),
            "prompt_sources": prompt_sources,
            "system_prompt": args.system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "intervention_mode": args.intervention_mode,
            "base_arm_included": not args.skip_base,
            "intervention_layers": intervention_layers,
            "arms": arms,
            "system": system_record(),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_count": len(args.direction_key),
                "output_dir": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
