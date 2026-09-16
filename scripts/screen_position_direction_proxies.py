#!/usr/bin/env python3
"""Screen saved position directions with the original causal proxy tests."""

from __future__ import annotations

import argparse
import json
import re
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
from swift_abliteration.intervention import (
    activation_addition_input_hook,
    reference_activation_ablation_hooks,
)
from swift_abliteration.live_model import capture_last_token_logits, validate_live_model
from swift_abliteration.metrics import forward_kl_from_logits, summarize
from swift_abliteration.refusal_scoring import (
    DEFAULT_REFUSAL_OPENERS,
    refusal_scores_from_logits,
    resolve_single_token_ids,
)

GROUP_PATHS = {
    "standard_harmful": "data/prepared/evaluation_harmful.jsonl",
    "matched_harmful": "data/prepared/matched/evaluation_harmful.jsonl",
    "xstest_safe": "data/prepared/xstest-safe/selection.jsonl",
}
HARMFUL_GROUPS = ("standard_harmful", "matched_harmful")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--weight-screen-report", type=Path, required=True)
    parser.add_argument("--base-arm-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--standard-harmful-path", default=GROUP_PATHS["standard_harmful"])
    parser.add_argument("--matched-harmful-path", default=GROUP_PATHS["matched_harmful"])
    parser.add_argument("--safe-path", default=GROUP_PATHS["xstest_safe"])
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--maximum-reference-kl", type=float, default=0.10)
    parser.add_argument("--finalist-count", type=int, default=8)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    return parser.parse_args()


def split_tensor(values, lengths: list[int]):
    output = []
    start = 0
    for length in lengths:
        output.append(values[start : start + length])
        start += length
    if start != len(values):
        raise ValueError("Combined logits do not match the requested split lengths.")
    return output


def score_summary(values: np.ndarray) -> dict:
    return {
        **summarize(values),
        "positive_count": int((values > 0).sum()),
        "positive_rate": float((values > 0).mean()),
    }


def layer_from_key(key: str) -> int:
    match = re.search(r"_layer_(\d+)_", key)
    if match is None:
        raise ValueError(f"Candidate key has no layer: {key}")
    return int(match.group(1))


def main() -> int:
    args = parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if args.limit <= 0 or args.batch_size <= 0 or args.finalist_count <= 0:
        raise ValueError("Limits and batch size must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    candidate_dir = args.output_dir / "candidates"
    candidate_dir.mkdir()

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    manifest = json.loads(args.direction_manifest.read_text(encoding="utf-8"))
    weight_report = json.loads(args.weight_screen_report.read_text(encoding="utf-8"))
    base_arm = json.loads(args.base_arm_metadata.read_text(encoding="utf-8"))
    directions = load_file(str(args.directions), device="cpu")
    candidate_keys = sorted(manifest["candidates"])
    if set(candidate_keys) != set(directions):
        raise ValueError("Direction tensors and the direction manifest do not match.")

    group_paths = {
        "standard_harmful": args.standard_harmful_path,
        "matched_harmful": args.matched_harmful_path,
        "xstest_safe": args.safe_path,
    }
    groups = {}
    prompt_sources = {}
    for name, path in group_paths.items():
        rows, digest = read_prompt_slice(path, 0, args.limit)
        groups[name] = rows
        prompt_sources[name] = {"path": path, "sha256": digest, "count": len(rows)}

    safe_labels = base_arm["groups"]["xstest_safe"]["opening_refusal_labels"]
    if len(safe_labels) != len(groups["xstest_safe"]):
        raise ValueError("Base safe labels do not match the safe prompt count.")
    answerable_safe_indices = [
        index for index, refused in enumerate(safe_labels) if not bool(refused)
    ]
    if not answerable_safe_indices:
        raise ValueError("No base-answerable safe prompts are available for addition.")

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

    ordered_groups = (*HARMFUL_GROUPS, "xstest_safe")
    combined_prompts = [
        row["text"] for name in ordered_groups for row in groups[name]
    ]
    lengths = [len(groups[name]) for name in ordered_groups]
    base_combined = torch.stack(
        capture_last_token_logits(
            model,
            processor,
            combined_prompts,
            args.system_prompt,
            args.batch_size,
        )
    )
    base_split = split_tensor(base_combined, lengths)
    base_logits = dict(zip(ordered_groups, base_split, strict=True))
    base_scores = {
        name: refusal_scores_from_logits(base_logits[name], refusal_token_ids)
        .float()
        .numpy()
        for name in ordered_groups
    }
    safe_prompts = [
        groups["xstest_safe"][index]["text"] for index in answerable_safe_indices
    ]
    base_answerable_safe_scores = base_scores["xstest_safe"][answerable_safe_indices]

    results = {}
    for ordinal, key in enumerate(candidate_keys, start=1):
        direction = directions[key]
        source_layer = layer_from_key(key)
        with reference_activation_ablation_hooks(model, direction) as ablation_record:
            ablated_combined = torch.stack(
                capture_last_token_logits(
                    model,
                    processor,
                    combined_prompts,
                    args.system_prompt,
                    args.batch_size,
                )
            )
        ablated_split = split_tensor(ablated_combined, lengths)
        ablated_logits = dict(zip(ordered_groups, ablated_split, strict=True))
        ablated_scores = {
            name: refusal_scores_from_logits(ablated_logits[name], refusal_token_ids)
            .float()
            .numpy()
            for name in ordered_groups
        }
        reference_kl = forward_kl_from_logits(
            base_logits["xstest_safe"].numpy(),
            ablated_logits["xstest_safe"].numpy(),
        )

        with activation_addition_input_hook(
            model, direction, source_layer, 1.0
        ) as addition_record:
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
            refusal_scores_from_logits(added_logits, refusal_token_ids).float().numpy()
        )

        removal_improvement = {
            name: float(base_scores[name].mean() - ablated_scores[name].mean())
            for name in HARMFUL_GROUPS
        }
        addition_improvement = float(
            added_scores.mean() - base_answerable_safe_scores.mean()
        )
        weight_result = weight_report["results"][key]
        weight_comparison = weight_result["comparison"]
        minimum_weight_removal = min(
            weight_comparison["harmful"][name]["opening_refusal"][
                "removed_fraction_of_base_refusals"
            ]
            for name in HARMFUL_GROUPS
        )
        weight_kl = float(weight_comparison["harmless_kl_base_answered"]["mean"])
        source_layer_allowed = source_layer < int(cfg.model.num_layers * 0.8)
        official_proxy_eligible = (
            float(reference_kl.mean()) <= args.maximum_reference_kl
            and min(removal_improvement.values()) > 0
            and float(added_scores.mean()) >= 0
            and source_layer_allowed
        )
        bidirectional_proxy_eligible = (
            official_proxy_eligible and addition_improvement > 0
        )
        result = {
            "position": manifest["candidates"][key]["position"],
            "source_layer": source_layer,
            "source_layer_allowed_by_reference_pruning": source_layer_allowed,
            "official_proxy_eligible": official_proxy_eligible,
            "bidirectional_proxy_eligible": bidirectional_proxy_eligible,
            "reference_activation_ablation": {
                "removal_improvement": removal_improvement,
                "scores": {
                    name: score_summary(ablated_scores[name])
                    for name in ordered_groups
                },
                "harmless_kl": summarize(reference_kl),
                "intervention": ablation_record,
            },
            "activation_addition": {
                "base_answerable_safe_count": len(answerable_safe_indices),
                "mean_score_increase": addition_improvement,
                "base_scores": score_summary(base_answerable_safe_scores),
                "added_scores": score_summary(added_scores),
                "intervention": addition_record,
            },
            "weight_equivalent_generation_screen": {
                "minimum_opening_refusal_removal": minimum_weight_removal,
                "base_answered_safe_kl_mean": weight_kl,
            },
        }
        results[key] = result
        write_json(candidate_dir / f"{key}.json", result)
        print(
            json.dumps(
                {
                    "candidate": ordinal,
                    "candidate_count": len(candidate_keys),
                    "key": key,
                    "official_proxy_eligible": official_proxy_eligible,
                    "bidirectional_proxy_eligible": bidirectional_proxy_eligible,
                    "minimum_weight_removal": minimum_weight_removal,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ranked = sorted(
        candidate_keys,
        key=lambda key: (
            results[key]["bidirectional_proxy_eligible"],
            results[key]["official_proxy_eligible"],
            results[key]["weight_equivalent_generation_screen"][
                "minimum_opening_refusal_removal"
            ],
            min(
                results[key]["reference_activation_ablation"][
                    "removal_improvement"
                ].values()
            ),
            results[key]["activation_addition"]["mean_score_increase"],
            -results[key]["weight_equivalent_generation_screen"][
                "base_answered_safe_kl_mean"
            ],
        ),
        reverse=True,
    )
    finalists = ranked[: args.finalist_count]
    report = {
        "experiment": "reference_causal_proxy_screen_for_position_directions",
        "method_note": (
            "The activation-ablation arm reproduces the original candidate-selection "
            "proxy. It is not a weight-equivalent model edit. Finalists require a "
            "separate weight-equivalent generation test."
        ),
        "config": args.config,
        "direction_file_sha256": sha256_file(args.directions),
        "direction_manifest_sha256": sha256_file(args.direction_manifest),
        "weight_screen_report_sha256": sha256_file(args.weight_screen_report),
        "base_arm_metadata_sha256": sha256_file(args.base_arm_metadata),
        "prompt_sources": prompt_sources,
        "answerable_safe_indices": answerable_safe_indices,
        "refusal_openers": list(DEFAULT_REFUSAL_OPENERS),
        "candidate_count": len(candidate_keys),
        "results": results,
        "ranked": ranked,
        "finalists": finalists,
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_report": False,
        "system": system_record(),
    }
    write_json(args.output_dir / "proxy_report.json", report)
    print(json.dumps({"status": "complete", "finalists": finalists}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
