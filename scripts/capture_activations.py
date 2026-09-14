#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import asdict
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
from swift_abliteration.live_model import (
    capture_last_token_activations_multi,
    capture_last_token_logits,
    validate_live_model,
)

GROUPS = {
    "standard_direction_harmful": "data/prepared/direction_harmful.jsonl",
    "standard_direction_harmless": "data/prepared/direction_harmless.jsonl",
    "standard_evaluation_harmful": "data/prepared/evaluation_harmful.jsonl",
    "standard_evaluation_harmless": "data/prepared/evaluation_harmless.jsonl",
    "matched_direction_harmful": "data/prepared/matched/direction_harmful.jsonl",
    "matched_direction_harmless": "data/prepared/matched/direction_harmless.jsonl",
    "matched_evaluation_harmful": "data/prepared/matched/evaluation_harmful.jsonl",
    "matched_evaluation_harmless": "data/prepared/matched/evaluation_harmless.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument(
        "--candidate-config", default="configs/direction_candidates.toml"
    )
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    with Path(args.candidate_config).open("rb") as handle:
        candidate_cfg = tomllib.load(handle)
    direction_layers = [
        int(index) for index in candidate_cfg["study"]["direction_layers"]
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
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
    tensors = {}
    sources = {}
    prompt_cache = {}
    for name, path in GROUPS.items():
        prompts, digest = read_prompt_jsonl(path)
        prompt_cache[name] = prompts
        sources[name] = {"path": path, "count": len(prompts), "sha256": digest}
        layer_results = capture_last_token_activations_multi(
            model, processor, prompts, direction_layers, args.system_prompt
        )
        for layer_index, values in layer_results.items():
            tensors[f"{name}_layer_{layer_index}"] = torch.stack(values).contiguous()
    save_file(tensors, str(output / "activations.safetensors"))
    reference_logits = torch.stack(
        capture_last_token_logits(
            model,
            processor,
            prompt_cache["standard_evaluation_harmless"],
            args.system_prompt,
        )
    ).numpy()
    import numpy as np

    np.savez_compressed(
        output / "base_harmless_last_token_logits.npz", logits=reference_logits
    )
    capture_path = output / "activations.safetensors"
    logits_path = output / "base_harmless_last_token_logits.npz"
    write_json(
        output / "capture_manifest.json",
        {
            "config": asdict(cfg),
            "system": system_record(),
            "sources": sources,
            "direction_layers": direction_layers,
            "system_prompt": args.system_prompt,
            "activation_shapes": {
                name: list(tensor.shape) for name, tensor in tensors.items()
            },
            "logit_shape": list(reference_logits.shape),
            "artifact_sha256": {
                "activations.safetensors": sha256_file(capture_path),
                "base_harmless_last_token_logits.npz": sha256_file(logits_path),
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output),
                "prompt_counts": {k: v["count"] for k, v in sources.items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
