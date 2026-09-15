#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from swift_abliteration.config import load_config
from swift_abliteration.direction_study import normalized_average
from swift_abliteration.gpu_support import (
    read_prompt_jsonl,
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.intervention import (
    layerwise_weight_equivalent_ablation_hooks,
)
from swift_abliteration.live_model import validate_live_model
from swift_abliteration.live_model import apply_layerwise_runtime_edit

from scripts.screen_directions import GROUPS, run_arm


def nearest_anchor(layer: int, anchors: list[int]) -> int:
    return min(anchors, key=lambda anchor: (abs(anchor - layer), anchor))


def select_plans(plans: list[dict], requested: set[str]) -> list[dict]:
    selected = [
        plan
        for plan in plans
        if not requested or str(plan["name"]) in requested
    ]
    selected_names = {str(plan["name"]) for plan in selected}
    if requested and requested != selected_names:
        raise ValueError(
            f"Requested plans are not defined: {sorted(requested - selected_names)}"
        )
    return selected


def plan_target_layers(plan: dict) -> list[int]:
    if "target_layers" in plan:
        values = [int(value) for value in plan["target_layers"]]
    elif "target_first" in plan and "target_last" in plan:
        first = int(plan["target_first"])
        last = int(plan["target_last"])
        if last < first:
            raise ValueError("Plan target_last must not be less than target_first.")
        values = list(range(first, last + 1))
    else:
        raise ValueError("A plan needs target_layers or a target range.")
    if not values or len(values) != len(set(values)):
        raise ValueError("Plan target layers must be nonempty and unique.")
    return values


def resolve_plan(
    plan: dict,
    anchors: list[int],
    estimator: str,
    tensors: dict,
):
    import torch

    source = str(plan["source"])
    directions = {}
    direction_keys = {}
    fixed_anchor = plan.get("fixed_anchor")
    if fixed_anchor is not None and int(fixed_anchor) not in anchors:
        raise ValueError("Plan fixed_anchor must be one of the study anchors.")
    for target in plan_target_layers(plan):
        anchor = int(fixed_anchor) if fixed_anchor is not None else nearest_anchor(target, anchors)
        if source == "consensus":
            keys = [
                f"standard_layer_{anchor}_{estimator}",
                f"matched_layer_{anchor}_{estimator}",
            ]
            value = torch.from_numpy(
                normalized_average([tensors[key].numpy() for key in keys])
            )
            direction_keys[str(target)] = keys
        elif source in {"standard", "matched"}:
            key = f"{source}_layer_{anchor}_{estimator}"
            keys = [key]
            value = tensors[key]
            direction_keys[str(target)] = keys
        elif source == "artifact":
            key = str(plan["key_template"]).format(anchor=anchor)
            keys = [key]
            value = tensors[key]
            direction_keys[str(target)] = keys
        else:
            raise ValueError(f"Unsupported direction source: {source}")
        directions[target] = value
    return directions, direction_keys


def resolve_embedding_direction(
    plan: dict,
    anchors: list[int],
    estimator: str,
    tensors: dict,
):
    """Resolve an optional weight-equivalent embedding projection for a plan."""
    if "embedding_anchor" not in plan:
        return None, []
    anchor = int(plan["embedding_anchor"])
    if anchor not in anchors:
        raise ValueError("Plan embedding_anchor must be one of the study anchors.")
    embedding_plan = {
        "source": plan.get("embedding_source", plan["source"]),
        "target_layers": [anchor],
        "fixed_anchor": anchor,
    }
    if "embedding_key_template" in plan:
        embedding_plan["key_template"] = plan["embedding_key_template"]
    elif "key_template" in plan:
        embedding_plan["key_template"] = plan["key_template"]
    directions, keys = resolve_plan(
        embedding_plan, anchors, estimator, tensors
    )
    return directions[anchor], keys[str(anchor)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen weight-equivalent layer-specific refusal directions."
    )
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--plan-config", default="configs/layerwise_screen.toml")
    parser.add_argument("--directions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-limit", type=int)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--plan-name", action="append")
    parser.add_argument(
        "--intervention-mode",
        choices=("module-hooks", "in-memory-weight-edit"),
        default="module-hooks",
    )
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if args.group_limit is not None and args.group_limit <= 0:
        raise ValueError("Group limit must be positive.")

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    with Path(args.plan_config).open("rb") as handle:
        plan_cfg = tomllib.load(handle)
    anchors = [int(value) for value in plan_cfg["study"]["anchor_layers"]]
    estimator = str(plan_cfg["study"]["estimator"])
    requested = set(args.plan_name or [])
    plans = select_plans(plan_cfg["plans"], requested)
    if args.intervention_mode == "in-memory-weight-edit" and (
        not args.skip_base or len(plans) != 1
    ):
        raise RuntimeError(
            "An in-memory layerwise edit requires --skip-base and one selected plan."
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    prompt_groups = {}
    prompt_sources = {}
    for name, path in GROUPS.items():
        values, digest = read_prompt_jsonl(path)
        used = values[: args.group_limit] if args.group_limit else values
        prompt_groups[name] = used
        prompt_sources[name] = {
            "path": path,
            "source_count": len(values),
            "used_count": len(used),
            "sha256": digest,
        }

    direction_path = Path(args.directions)
    tensors = load_file(str(direction_path), device="cpu")
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
    for plan in plans:
        name = str(plan["name"])
        directions, direction_keys = resolve_plan(
            plan, anchors, estimator, tensors
        )
        embedding_direction, embedding_direction_keys = resolve_embedding_direction(
            plan, anchors, estimator, tensors
        )
        alpha = float(plan.get("alpha", 1.0))
        if args.intervention_mode == "in-memory-weight-edit":
            intervention = apply_layerwise_runtime_edit(
                model,
                cfg,
                directions,
                alpha,
                embedding_direction=embedding_direction,
            )
            arms[name] = run_arm(
                model,
                processor,
                prompt_groups,
                output / name,
                args.max_new_tokens,
                args.system_prompt,
                args.batch_size,
            )
        else:
            with layerwise_weight_equivalent_ablation_hooks(
                model,
                cfg,
                directions,
                alpha,
                embedding_direction=embedding_direction,
            ) as intervention:
                arms[name] = run_arm(
                    model,
                    processor,
                    prompt_groups,
                    output / name,
                    args.max_new_tokens,
                    args.system_prompt,
                    args.batch_size,
                )
        arms[name]["direction_source"] = plan["source"]
        arms[name]["direction_keys_by_target_layer"] = direction_keys
        arms[name]["embedding_direction_keys"] = embedding_direction_keys
        arms[name]["intervention"] = intervention

    write_json(
        output / "screen_manifest.json",
        {
            "config": args.config,
            "plan_config": args.plan_config,
            "direction_file": str(direction_path.resolve()),
            "direction_file_sha256": sha256_file(direction_path),
            "prompt_sources": prompt_sources,
            "system_prompt": args.system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "group_limit": args.group_limit,
            "base_arm_included": not args.skip_base,
            "intervention_mode": args.intervention_mode,
            "anchor_layers": anchors,
            "estimator": estimator,
            "arms": arms,
            "uses_final_test": False,
            "checkpoint_saved": False,
            "system": system_record(),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_count": len(plans),
                "output_dir": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
