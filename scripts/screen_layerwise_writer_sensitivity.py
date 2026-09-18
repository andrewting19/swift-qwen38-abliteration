#!/usr/bin/env python3
"""Rank individual residual writers by refusal effect versus harmless KL cost."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.screen_generation_subspaces import read_prompt_slice, validate_basis
from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.intervention import (
    layerwise_weight_equivalent_ablation_hooks,
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


@dataclass(frozen=True)
class Writer:
    name: str
    component: str
    layer: int | None


def split_tensor(values, lengths: list[int]):
    output = []
    start = 0
    for length in lengths:
        output.append(values[start : start + length])
        start += length
    if start != len(values):
        raise ValueError("Combined logits do not match group lengths.")
    return output


def retained_fraction(full_effect: float, candidate_effect: float) -> float | None:
    if not np.isfinite(full_effect) or abs(full_effect) <= 1e-9:
        return None
    return float(candidate_effect / full_effect)


def restoration_priority(kl_recovered: float, minimum_retained: float) -> float:
    """Prefer KL recovery that loses little of the full refusal-score effect."""
    if not np.isfinite(kl_recovered) or not np.isfinite(minimum_retained):
        return float("-inf")
    if kl_recovered <= 0.0:
        return float(kl_recovered)
    effect_loss = max(1.0 - minimum_retained, 1e-3)
    return float(kl_recovered / effect_loss)


def cumulative_counts(total: int) -> list[int]:
    requested = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
    return sorted({value for value in requested if value <= total})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--base-arm-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if min(args.limit, args.batch_size) <= 0:
        raise ValueError("Limits and batch size must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    candidates = load_file(str(args.candidate_file), device="cpu")
    if args.candidate_key not in candidates:
        raise KeyError(f"Missing candidate: {args.candidate_key}")
    directions = candidates[args.candidate_key]
    if directions.ndim != 3 or directions.shape[0] != cfg.model.num_layers:
        raise ValueError(
            "The writer sensitivity screen requires [layers, rank, hidden]."
        )
    ranks = [validate_basis(directions[layer]) for layer in range(len(directions))]
    if len(set(ranks)) != 1:
        raise ValueError(f"Candidate rank changes by layer: {ranks}")

    base_arm = json.loads(args.base_arm_metadata.read_text(encoding="utf-8"))
    if "base" in base_arm:
        base_arm = base_arm["base"]
    groups = {}
    prompt_sources = {}
    for name, path in GROUP_PATHS.items():
        rows, digest = read_prompt_slice(path, 0, args.limit)
        groups[name] = rows
        prompt_sources[name] = {"path": path, "sha256": digest, "count": len(rows)}
    safe_labels = base_arm["groups"]["xstest_safe"]["opening_refusal_labels"]
    if len(safe_labels) < args.limit:
        raise ValueError("Base safe labels do not cover the requested prompt count.")
    answerable_safe = np.flatnonzero(~np.asarray(safe_labels[: args.limit], dtype=bool))
    if not len(answerable_safe):
        raise ValueError("No base-answerable safe prompt is available.")

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
    lengths = [len(groups[name]) for name in ordered_groups]
    prompts = [row["text"] for name in ordered_groups for row in groups[name]]

    def capture():
        combined = torch.stack(
            capture_last_token_logits(
                model,
                processor,
                prompts,
                args.system_prompt,
                args.batch_size,
            )
        )
        return dict(zip(ordered_groups, split_tensor(combined, lengths), strict=True))

    base_logits = capture()
    base_scores = {
        name: refusal_scores_from_logits(base_logits[name], refusal_token_ids)
        .float()
        .numpy()
        for name in HARMFUL_GROUPS
    }
    all_layers = list(range(cfg.edit.first_layer, cfg.edit.last_layer + 1))
    assignments = {layer: directions[layer] for layer in all_layers}

    def capture_edit(
        attention_layers: set[int],
        mlp_layers: set[int],
        include_embedding: bool,
    ):
        with layerwise_weight_equivalent_ablation_hooks(
            model,
            cfg,
            assignments,
            alpha=1.0,
            embedding_direction=(
                directions[cfg.edit.first_layer]
                if include_embedding and cfg.edit.include_embedding
                else None
            ),
            attention_layers=attention_layers,
            mlp_layers=mlp_layers,
        ) as intervention:
            logits = capture()
        return logits, intervention

    def evaluate(logits, full_kl_mean: float | None, full_effects: dict | None):
        safe_kl = forward_kl_from_logits(
            base_logits["xstest_safe"].numpy(),
            logits["xstest_safe"].numpy(),
        )[answerable_safe]
        effects = {}
        retained = {}
        for name in HARMFUL_GROUPS:
            scores = (
                refusal_scores_from_logits(logits[name], refusal_token_ids)
                .float()
                .numpy()
            )
            effect = float(base_scores[name].mean() - scores.mean())
            effects[name] = effect
            if full_effects is not None:
                retained[name] = retained_fraction(full_effects[name], effect)
        valid_retained = [value for value in retained.values() if value is not None]
        result = {
            "harmless_kl": summarize(safe_kl),
            "refusal_score_effect": effects,
            "refusal_score_effect_retained": retained,
            "minimum_effect_retained": min(valid_retained)
            if valid_retained
            else None,
        }
        if full_kl_mean is not None:
            result["harmless_kl_recovered_from_full"] = float(
                full_kl_mean - safe_kl.mean()
            )
        return result

    full_logits, full_intervention = capture_edit(set(all_layers), set(all_layers), True)
    full_metrics = evaluate(full_logits, None, None)
    full_kl_mean = float(full_metrics["harmless_kl"]["mean"])
    full_effects = full_metrics["refusal_score_effect"]

    writers = []
    if cfg.edit.include_embedding:
        writers.append(Writer("model.language_model.embed_tokens", "embedding", None))
    for layer in all_layers:
        writers.append(Writer(f"layer_{layer}_attention", "attention", layer))
        writers.append(Writer(f"layer_{layer}_mlp", "mlp", layer))

    individual = {}
    for index, writer in enumerate(writers, start=1):
        attention = set(all_layers)
        mlp = set(all_layers)
        include_embedding = True
        if writer.component == "attention":
            attention.remove(int(writer.layer))
        elif writer.component == "mlp":
            mlp.remove(int(writer.layer))
        else:
            include_embedding = False
        logits, intervention = capture_edit(attention, mlp, include_embedding)
        metrics = evaluate(logits, full_kl_mean, full_effects)
        retained = metrics["minimum_effect_retained"]
        recovered = metrics["harmless_kl_recovered_from_full"]
        metrics.update(
            {
                "writer": {
                    "name": writer.name,
                    "component": writer.component,
                    "layer": writer.layer,
                },
                "restoration_priority": restoration_priority(recovered, retained),
                "active_module_count": intervention["module_count"],
            }
        )
        individual[writer.name] = metrics
        if index % 8 == 0 or index == len(writers):
            print(
                json.dumps(
                    {
                        "status": "writer_sensitivity",
                        "completed": index,
                        "total": len(writers),
                    },
                    sort_keys=True,
                )
            )

    ranked_writers = sorted(
        individual,
        key=lambda name: individual[name]["restoration_priority"],
        reverse=True,
    )
    writer_lookup = {writer.name: writer for writer in writers}
    cumulative = {}
    for count in cumulative_counts(len(writers)):
        restored = ranked_writers[:count]
        restored_attention = {
            int(writer_lookup[name].layer)
            for name in restored
            if writer_lookup[name].component == "attention"
        }
        restored_mlp = {
            int(writer_lookup[name].layer)
            for name in restored
            if writer_lookup[name].component == "mlp"
        }
        restore_embedding = any(
            writer_lookup[name].component == "embedding" for name in restored
        )
        logits, intervention = capture_edit(
            set(all_layers).difference(restored_attention),
            set(all_layers).difference(restored_mlp),
            not restore_embedding,
        )
        metrics = evaluate(logits, full_kl_mean, full_effects)
        metrics.update(
            {
                "restored_writer_count": count,
                "restored_writers": restored,
                "active_module_count": intervention["module_count"],
                "attention_layers": intervention["attention_layers"],
                "mlp_layers": intervention["mlp_layers"],
                "embedding_included": intervention["embedding_included"],
            }
        )
        cumulative[f"restore_top_{count}"] = metrics
        print(
            json.dumps(
                {
                    "arm": f"restore_top_{count}",
                    "mean_kl": metrics["harmless_kl"]["mean"],
                    "minimum_effect_retained": metrics["minimum_effect_retained"],
                },
                sort_keys=True,
            )
        )

    pareto_order = sorted(
        cumulative,
        key=lambda name: (
            cumulative[name]["harmless_kl"]["mean"] <= 0.1,
            cumulative[name]["minimum_effect_retained"],
            -cumulative[name]["harmless_kl"]["mean"],
        ),
        reverse=True,
    )
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "layerwise_writer_sensitivity_report.json"
    write_json(
        report_path,
        {
            "schema_version": 1,
            "experiment": "layerwise_rank2_leave_one_writer_out_and_cumulative_restore",
            "config": args.config,
            "candidate_file": str(args.candidate_file),
            "candidate_file_sha256": sha256_file(args.candidate_file),
            "candidate_key": args.candidate_key,
            "rank": ranks[0],
            "prompt_sources": prompt_sources,
            "limit_per_group": args.limit,
            "answerable_safe_count": len(answerable_safe),
            "full_edit": {**full_metrics, "intervention": full_intervention},
            "individual_writer_restoration": individual,
            "ranked_writers_to_restore": ranked_writers,
            "cumulative_restoration": cumulative,
            "ranked_cumulative_arms": pareto_order,
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_report": False,
            "system": system_record(),
        },
    )
    print(
        json.dumps(
            {"status": "complete", "report": str(report_path), "ranked": pareto_order},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
