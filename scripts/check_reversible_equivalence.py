#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.intervention import (
    planned_runtime_writers,
    weight_equivalent_ablation_hooks,
)
from swift_abliteration.live_model import (
    mtp_root,
    render_prompt,
    text_backbone,
    validate_live_model,
)


def projected_output_weight(weight, direction, alpha):
    import torch

    r = direction.to(device=weight.device, dtype=torch.float32)
    r = r / torch.linalg.vector_norm(r)
    value = weight.float()
    value = value - alpha * r[:, None] * (r @ value)[None, :]
    return value.to(dtype=weight.dtype)


def compare(observed, expected) -> dict[str, float]:
    import torch

    difference = (observed.float() - expected.float()).abs()
    left = observed.float().reshape(-1)
    right = expected.float().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=0)
    return {
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "cosine_similarity": float(cosine.item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--directions", required=True)
    parser.add_argument("--direction-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()

    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    directions = load_file(args.directions, device="cpu")
    if args.direction_key not in directions:
        raise ValueError(f"Direction key is not present: {args.direction_key}")
    direction = directions[args.direction_key]
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    validate_live_model(model, cfg)
    processor = AutoProcessor.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    targets = planned_runtime_writers(model, cfg, generation_uses_mtp=False)
    bias_modules = [
        target.name
        for target in targets
        if target.kind == "linear" and getattr(target.module, "bias", None) is not None
    ]
    selected_names = {
        "model.language_model.embed_tokens",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.3.self_attn.o_proj",
    }
    checks = []
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    with torch.inference_mode(), weight_equivalent_ablation_hooks(
        model, cfg, direction, args.alpha, generation_uses_mtp=False
    ) as intervention:
        for target in targets:
            if target.name not in selected_names:
                continue
            if target.kind == "embedding":
                token_ids = torch.tensor(
                    [[0, 1, cfg.model.vocab_size - 1]],
                    device=target.module.weight.device,
                )
                observed = target.module(token_ids)
                rows = target.module.weight[token_ids].float()
                r = direction.to(device=rows.device, dtype=torch.float32)
                r = r / torch.linalg.vector_norm(r)
                expected = (rows - args.alpha * (rows @ r).unsqueeze(-1) * r).to(
                    dtype=target.module.weight.dtype
                )
            else:
                sample = torch.randn(
                    (1, 2, target.module.in_features),
                    generator=generator,
                    dtype=torch.float32,
                ).to(
                    device=target.module.weight.device,
                    dtype=target.module.weight.dtype,
                )
                observed = target.module(sample)
                projected_weight = projected_output_weight(
                    target.module.weight, direction, args.alpha
                )
                expected = functional.linear(sample, projected_weight, target.module.bias)
                del projected_weight
            metrics = compare(observed, expected)
            if (
                metrics["cosine_similarity"] < 0.999
                or metrics["maximum_absolute_error"] > 0.125
            ):
                raise RuntimeError(
                    f"Live equivalence check failed for {target.name}: {metrics}"
                )
            checks.append({"module": target.name, **metrics})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mtp_available = True
    mtp_calls = 0
    try:
        mtp_module = mtp_root(model)
    except RuntimeError:
        mtp_available = False
        mtp_module = None
    handle = None
    if mtp_module is not None:

        def count_mtp(_module, _inputs):
            nonlocal mtp_calls
            mtp_calls += 1

        handle = mtp_module.register_forward_pre_hook(count_mtp)
    try:
        tokenizer = getattr(processor, "tokenizer", processor)
        rendered = render_prompt(tokenizer, "Hello", "You are a helpful assistant.")
        batch = tokenizer([rendered], return_tensors="pt")
        device = text_backbone(model).embed_tokens.weight.device
        batch = {name: value.to(device) for name, value in batch.items()}
        with torch.inference_mode():
            model.generate(**batch, do_sample=False, max_new_tokens=1)
    finally:
        if handle is not None:
            handle.remove()
    if mtp_calls:
        raise RuntimeError("The active generation path executed MTP unexpectedly.")

    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    embeddings_tied = bool(
        output_embedding is not None and output_embedding.weight is input_embedding.weight
    )
    result = {
        "status": "passed",
        "config": args.config,
        "direction_file_sha256": sha256_file(args.directions),
        "direction_key": args.direction_key,
        "runtime_writer_count": len(targets),
        "bias_module_count": len(bias_modules),
        "bias_modules": bias_modules,
        "input_output_embeddings_tied": embeddings_tied,
        "mtp_available": mtp_available,
        "mtp_calls_during_one_token_generation": mtp_calls,
        "intervention": intervention,
        "sample_checks": checks,
        "system": system_record(),
    }
    write_json(Path(args.output), result)
    print(json.dumps({"status": "passed", "writer_count": len(targets), "checks": checks}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
