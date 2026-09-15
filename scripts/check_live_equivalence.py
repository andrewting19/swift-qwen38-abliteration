#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np

from scripts.screen_directions import GROUPS
from scripts.screen_layerwise import resolve_embedding_direction, resolve_plan
from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    read_prompt_jsonl,
    require_acknowledgement,
    require_large_gpu,
    system_record,
    write_json,
)
from swift_abliteration.intervention import (
    layerwise_weight_equivalent_ablation_hooks,
)
from swift_abliteration.live_model import (
    apply_layerwise_runtime_edit,
    capture_last_token_logits,
    validate_live_model,
)
from swift_abliteration.metrics import forward_kl_from_logits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare live output hooks with an in-memory weight projection."
    )
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--plan-config", required=True)
    parser.add_argument("--plan-name", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    with Path(args.plan_config).open("rb") as handle:
        plan_cfg = tomllib.load(handle)
    plans = [plan for plan in plan_cfg["plans"] if plan["name"] == args.plan_name]
    if len(plans) != 1:
        raise ValueError("The plan name must select exactly one plan.")
    plan = plans[0]
    anchors = [int(value) for value in plan_cfg["study"]["anchor_layers"]]
    estimator = str(plan_cfg["study"]["estimator"])
    tensors = load_file(args.directions, device="cpu")
    directions, direction_keys = resolve_plan(plan, anchors, estimator, tensors)
    embedding_direction, embedding_keys = resolve_embedding_direction(
        plan, anchors, estimator, tensors
    )

    prompt_values = []
    prompt_sources = {}
    for group in ("standard_harmful", "standard_harmless"):
        values, digest = read_prompt_jsonl(GROUPS[group])
        prompt_values.append(values[0])
        prompt_sources[group] = {"sha256": digest, "selected_id": 0}
    selected_prompt_sha256 = hashlib.sha256(
        json.dumps(prompt_values, ensure_ascii=False).encode()
    ).hexdigest()

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

    def capture() -> np.ndarray:
        return torch.stack(
            capture_last_token_logits(
                model, processor, prompt_values, args.system_prompt
            )
        ).numpy()

    base = capture()
    with layerwise_weight_equivalent_ablation_hooks(
        model,
        cfg,
        directions,
        1.0,
        embedding_direction=embedding_direction,
    ) as hook_record:
        hooked = capture()
    edit_record = apply_layerwise_runtime_edit(
        model,
        cfg,
        directions,
        1.0,
        embedding_direction=embedding_direction,
    )
    edited = capture()
    report = {
        "plan": args.plan_name,
        "prompt_count": len(prompt_values),
        "prompt_sources": prompt_sources,
        "selected_prompt_sha256": selected_prompt_sha256,
        "direction_keys_by_target_layer": direction_keys,
        "embedding_direction_keys": embedding_keys,
        "hook_record": hook_record,
        "edit_record": edit_record,
        "base_to_hook_kl": forward_kl_from_logits(base, hooked).tolist(),
        "base_to_in_memory_kl": forward_kl_from_logits(base, edited).tolist(),
        "hook_to_in_memory_kl": forward_kl_from_logits(hooked, edited).tolist(),
        "hook_vs_edit_max_abs_logit": float(np.max(np.abs(hooked - edited))),
        "hook_vs_edit_mean_abs_logit": float(np.mean(np.abs(hooked - edited))),
        "same_argmax_per_prompt": (hooked.argmax(-1) == edited.argmax(-1)).tolist(),
        "raw_text_in_report": False,
        "system": system_record(),
    }
    write_json(args.output, report)
    print(json.dumps({"status": "complete", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
