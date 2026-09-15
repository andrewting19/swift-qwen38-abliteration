#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    read_prompt_jsonl,
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.intervention import (
    activation_addition_input_hook,
    weight_equivalent_ablation_hooks,
)
from swift_abliteration.live_model import capture_last_token_logits, validate_live_model
from swift_abliteration.metrics import forward_kl_from_logits, summarize
from swift_abliteration.refusal_scoring import (
    DEFAULT_REFUSAL_OPENERS,
    refusal_scores_from_logits,
    resolve_single_token_ids,
)


GROUPS = {
    "standard_harmful": "data/prepared/evaluation_harmful.jsonl",
    "matched_harmful": "data/prepared/matched/evaluation_harmful.jsonl",
    "standard_harmless": "data/prepared/evaluation_harmless.jsonl",
    "matched_harmless": "data/prepared/matched/evaluation_harmless.jsonl",
}


def direction_layer(key: str) -> int:
    match = re.search(r"_layer_(\d+)_", key)
    if match is None:
        raise ValueError(f"Direction key does not contain a layer: {key}")
    return int(match.group(1))


def score_summary(values: np.ndarray) -> dict:
    return {
        **summarize(values),
        "positive_count": int((values > 0).sum()),
        "positive_rate": float((values > 0).mean()),
    }


def split_logits(values: list, lengths: list[int]) -> list:
    output = []
    start = 0
    for length in lengths:
        output.append(values[start : start + length])
        start += length
    if start != len(values):
        raise ValueError("Logit split lengths do not match the combined values.")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Causally screen rank-1 candidates with short next-token forwards."
    )
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--direction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-group", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--maximum-mean-kl", type=float, default=0.10)
    parser.add_argument("--finalist-count", type=int, default=4)
    parser.add_argument("--pilot-output-dir", type=Path)
    parser.add_argument("--pilot-max-new-tokens", type=int, default=32)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    direction_report_path = args.direction_dir / "direction_report.json"
    direction_report = json.loads(direction_report_path.read_text(encoding="utf-8"))
    candidate_keys = direction_report["shortlist"]
    directions_path = args.direction_dir / "rank1_directions.safetensors"
    directions = load_file(str(directions_path), device="cpu")
    missing = [key for key in candidate_keys if key not in directions]
    if missing:
        raise KeyError(f"Shortlisted directions are missing: {missing}")

    prompts = {}
    prompt_sources = {}
    for name, path in GROUPS.items():
        values, digest = read_prompt_jsonl(path)
        values = values[: args.limit_per_group]
        prompts[name] = values
        prompt_sources[name] = {"path": path, "sha256": digest, "count": len(values)}

    base_scores = {}
    for name in GROUPS:
        split = "evaluation"
        source, kind = name.split("_", maxsplit=1)
        capture_name = f"{source}_{split}_{kind}"
        with np.load(args.capture_dir / "groups" / f"{capture_name}.npz") as values:
            base_scores[name] = values["refusal_scores"][: args.limit_per_group].copy()
            if name == "standard_harmless":
                base_harmless_logits = values["first_step_logits"][
                    : args.limit_per_group
                ].copy()

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

    removal_groups = ("standard_harmful", "matched_harmful", "standard_harmless")
    addition_groups = ("standard_harmless", "matched_harmless")
    removal_prompts = [value for name in removal_groups for value in prompts[name]]
    addition_prompts = [value for name in addition_groups for value in prompts[name]]
    results = {}
    for key in candidate_keys:
        direction = directions[key]
        layer = direction_layer(key)
        with weight_equivalent_ablation_hooks(
            model, cfg, direction, 1.0, generation_uses_mtp=False
        ) as removal_record:
            combined = capture_last_token_logits(
                model,
                processor,
                removal_prompts,
                args.system_prompt,
                args.batch_size,
            )
        split = split_logits(combined, [len(prompts[name]) for name in removal_groups])
        removal_logits = {
            name: torch.stack(values) for name, values in zip(removal_groups, split)
        }
        removal_scores = {
            name: refusal_scores_from_logits(values, refusal_token_ids).float().numpy()
            for name, values in removal_logits.items()
        }
        kl_values = forward_kl_from_logits(
            base_harmless_logits, removal_logits["standard_harmless"].numpy()
        )

        with activation_addition_input_hook(model, direction, layer, 1.0) as add_record:
            combined = capture_last_token_logits(
                model,
                processor,
                addition_prompts,
                args.system_prompt,
                args.batch_size,
            )
        split = split_logits(combined, [len(prompts[name]) for name in addition_groups])
        addition_scores = {
            name: refusal_scores_from_logits(torch.stack(values), refusal_token_ids)
            .float()
            .numpy()
            for name, values in zip(addition_groups, split)
        }

        removal_improvement = {
            name: float(base_scores[name].mean() - removal_scores[name].mean())
            for name in ("standard_harmful", "matched_harmful")
        }
        addition_improvement = {
            name: float(addition_scores[name].mean() - base_scores[name].mean())
            for name in addition_groups
        }
        eligible = (
            float(kl_values.mean()) <= args.maximum_mean_kl
            and min(removal_improvement.values()) > 0
            and min(addition_improvement.values()) > 0
        )
        results[key] = {
            "layer": layer,
            "position": direction_report["candidates"][key]["position"],
            "eligible": eligible,
            "removal_improvement": removal_improvement,
            "addition_improvement": addition_improvement,
            "removal_scores": {
                name: score_summary(values) for name, values in removal_scores.items()
            },
            "addition_scores": {
                name: score_summary(values) for name, values in addition_scores.items()
            },
            "mean_harmful_score_reduction": float(
                np.mean(list(removal_improvement.values()))
            ),
            "minimum_harmful_score_reduction": min(removal_improvement.values()),
            "minimum_harmless_addition_increase": min(addition_improvement.values()),
            "harmless_kl": summarize(kl_values),
            "removal_intervention": removal_record,
            "addition_intervention": add_record,
        }

    ranked = sorted(
        (key for key, result in results.items() if result["eligible"]),
        key=lambda key: (
            results[key]["minimum_harmful_score_reduction"],
            results[key]["minimum_harmless_addition_increase"],
            -results[key]["harmless_kl"]["mean"],
        ),
        reverse=True,
    )
    finalists = ranked[: args.finalist_count]
    write_json(
        args.output_dir / "proxy_report.json",
        {
            "capture_manifest_sha256": sha256_file(
                args.capture_dir / "capture_manifest.json"
            ),
            "direction_report_sha256": sha256_file(direction_report_path),
            "direction_file_sha256": sha256_file(directions_path),
            "prompt_sources": prompt_sources,
            "limit_per_group": args.limit_per_group,
            "maximum_mean_kl": args.maximum_mean_kl,
            "results": results,
            "finalists": finalists,
            "stop_reason": None
            if finalists
            else "No rank-1 candidate passed proxy gates.",
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_report": False,
            "system": system_record(),
        },
    )
    if finalists and args.pilot_output_dir is not None:
        if args.pilot_output_dir.exists():
            raise FileExistsError(
                f"Pilot output already exists: {args.pilot_output_dir}"
            )
        args.pilot_output_dir.mkdir(parents=True)
        from scripts.screen_directions import run_arm

        def run_pilot_arm(arm: str, batch_size: int) -> tuple[dict, int]:
            attempt = args.pilot_output_dir / f".{arm}.batch{batch_size}.attempt"
            try:
                result = run_arm(
                    model,
                    processor,
                    prompts,
                    attempt,
                    args.pilot_max_new_tokens,
                    args.system_prompt,
                    batch_size,
                )
            except torch.cuda.OutOfMemoryError:
                failed = args.pilot_output_dir / f"{arm}.batch{batch_size}.oom"
                attempt.rename(failed)
                torch.cuda.empty_cache()
                fallback = max(1, batch_size // 2)
                retry = args.pilot_output_dir / f".{arm}.batch{fallback}.attempt"
                result = run_arm(
                    model,
                    processor,
                    prompts,
                    retry,
                    args.pilot_max_new_tokens,
                    args.system_prompt,
                    fallback,
                )
                retry.rename(args.pilot_output_dir / arm)
                return result, fallback
            attempt.rename(args.pilot_output_dir / arm)
            return result, batch_size

        base_result, effective_batch_size = run_pilot_arm("base", args.batch_size)
        pilot_arms = {"base": {**base_result, "batch_size": effective_batch_size}}
        for key in finalists:
            with weight_equivalent_ablation_hooks(
                model, cfg, directions[key], 1.0, generation_uses_mtp=False
            ) as intervention:
                result, used_batch_size = run_pilot_arm(key, effective_batch_size)
                pilot_arms[key] = {**result, "batch_size": used_batch_size}
            pilot_arms[key]["intervention"] = intervention
        write_json(
            args.pilot_output_dir / "screen_manifest.json",
            {
                "source": "rank1_proxy_finalists",
                "proxy_report_sha256": sha256_file(
                    args.output_dir / "proxy_report.json"
                ),
                "direction_file": str(directions_path.resolve()),
                "direction_file_sha256": sha256_file(directions_path),
                "prompt_sources": prompt_sources,
                "max_new_tokens": args.pilot_max_new_tokens,
                "requested_batch_size": args.batch_size,
                "arms": pilot_arms,
                "uses_final_test": False,
                "checkpoint_saved": False,
                "openai_models_used": False,
                "remote_api_used": False,
                "system": system_record(),
            },
        )
    print(json.dumps({"status": "complete", "finalists": finalists}))
    return 0 if finalists else 2


if __name__ == "__main__":
    raise SystemExit(main())
