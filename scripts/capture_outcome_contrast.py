#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.screen_directions import generate_group
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
    capture_last_token_activations_multi,
    validate_live_model,
)


GROUPS = {
    "standard_harmful": "data/prepared/direction_harmful.jsonl",
    "matched_harmful": "data/prepared/matched/direction_harmful.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture activations and responses for an outcome-matched contrast."
    )
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--direction-key", action="append", required=True)
    parser.add_argument("--capture-layer", type=int, default=52)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()

    import torch
    from safetensors.numpy import save_file
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    tensors = load_file(str(args.directions), device="cpu")
    missing = [key for key in args.direction_key if key not in tensors]
    if missing:
        raise KeyError(f"Missing direction keys: {missing}")
    basis = torch.stack([tensors[key] for key in args.direction_key])
    prompts = {}
    sources = {}
    for name, path in GROUPS.items():
        values, digest = read_prompt_jsonl(path)
        prompts[name] = values
        sources[name] = {"path": path, "count": len(values), "sha256": digest}

    args.output_dir.mkdir(parents=True, exist_ok=False)
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
    layers = list(range(cfg.edit.first_layer, cfg.edit.last_layer + 1))
    intervention = apply_layerwise_runtime_edit(
        model,
        cfg,
        {index: basis for index in layers},
        1.0,
        embedding_direction=basis,
    )

    activations = {}
    response_hashes = {}
    tokenizer = getattr(processor, "tokenizer", processor)
    for name, values in prompts.items():
        captured = capture_last_token_activations_multi(
            model,
            processor,
            values,
            [args.capture_layer],
            args.system_prompt,
        )[args.capture_layer]
        activations[name] = np.stack([value.numpy() for value in captured]).astype(
            np.float32
        )
        response_hashes[name] = generate_group(
            model,
            tokenizer,
            values,
            args.output_dir / f"{name}.jsonl",
            args.max_new_tokens,
            args.system_prompt,
            args.batch_size,
        )

    activations_path = args.output_dir / "activations.safetensors"
    save_file(activations, activations_path)
    write_json(
        args.output_dir / "capture_manifest.json",
        {
            "config": args.config,
            "direction_file": str(args.directions.resolve()),
            "direction_file_sha256": sha256_file(args.directions),
            "direction_keys": args.direction_key,
            "capture_layer": args.capture_layer,
            "sources": sources,
            "response_sha256": response_hashes,
            "activations_sha256": sha256_file(activations_path),
            "intervention": intervention,
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_manifest": False,
            "system": system_record(),
        },
    )
    print(json.dumps({"status": "complete", "group_count": len(prompts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
