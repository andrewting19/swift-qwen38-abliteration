#!/usr/bin/env python3
"""Measure rank-2 refusal and KL sensitivity when one layer band is restored."""

from __future__ import annotations

import argparse
import json
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
    """Return retained effect when the full-edit denominator is meaningful."""
    if not np.isfinite(full_effect) or abs(full_effect) <= 1e-9:
        return None
    return float(candidate_effect / full_effect)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--base-arm-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--band-size", type=int, default=8)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if min(args.limit, args.batch_size, args.band_size) <= 0:
        raise ValueError("Limits, batch size, and band size must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    candidates = load_file(str(args.candidate_file), device="cpu")
    if args.candidate_key not in candidates:
        raise KeyError(f"Missing candidate: {args.candidate_key}")
    direction = candidates[args.candidate_key]
    rank = validate_basis(direction)
    if rank != 2:
        raise ValueError("The layer-band sensitivity scan requires rank 2.")
    base_arm = json.loads(args.base_arm_metadata.read_text(encoding="utf-8"))

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

    def capture_edit(active_layers: list[int]):
        with layerwise_weight_equivalent_ablation_hooks(
            model,
            cfg,
            {layer: direction for layer in active_layers},
            alpha=1.0,
            embedding_direction=direction if cfg.edit.include_embedding else None,
        ) as intervention:
            logits = capture()
        return logits, intervention

    full_logits, full_intervention = capture_edit(all_layers)
    full_safe_kl = forward_kl_from_logits(
        base_logits["xstest_safe"].numpy(),
        full_logits["xstest_safe"].numpy(),
    )[answerable_safe]
    full_effects = {}
    for name in HARMFUL_GROUPS:
        full_scores = (
            refusal_scores_from_logits(full_logits[name], refusal_token_ids)
            .float()
            .numpy()
        )
        full_effects[name] = float(base_scores[name].mean() - full_scores.mean())

    bands = [
        all_layers[start : start + args.band_size]
        for start in range(0, len(all_layers), args.band_size)
    ]
    results = {}
    for band in bands:
        active = [layer for layer in all_layers if layer not in set(band)]
        logits, intervention = capture_edit(active)
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
            retained[name] = retained_fraction(full_effects[name], effect)
        key = f"restore_layers_{band[0]}_{band[-1]}"
        results[key] = {
            "restored_layers": band,
            "active_layer_count": len(active),
            "harmless_kl": summarize(safe_kl),
            "harmless_kl_recovered_from_full": float(
                full_safe_kl.mean() - safe_kl.mean()
            ),
            "refusal_score_effect": effects,
            "refusal_score_effect_retained": retained,
            "minimum_effect_retained": min(
                value for value in retained.values() if value is not None
            ),
            "intervention": intervention,
        }
        print(
            json.dumps(
                {
                    "arm": key,
                    "mean_kl": float(safe_kl.mean()),
                    "minimum_effect_retained": results[key][
                        "minimum_effect_retained"
                    ],
                },
                sort_keys=True,
            )
        )

    ranked = sorted(
        results,
        key=lambda key: (
            results[key]["minimum_effect_retained"] >= 0.8,
            results[key]["harmless_kl_recovered_from_full"],
            results[key]["minimum_effect_retained"],
        ),
        reverse=True,
    )
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "rank2_layer_band_sensitivity_report.json"
    write_json(
        report_path,
        {
            "schema_version": 1,
            "experiment": "rank2_leave_one_layer_band_out_sensitivity",
            "config": args.config,
            "candidate_file": str(args.candidate_file),
            "candidate_file_sha256": sha256_file(args.candidate_file),
            "candidate_key": args.candidate_key,
            "rank": rank,
            "prompt_sources": prompt_sources,
            "limit_per_group": args.limit,
            "band_size": args.band_size,
            "full_edit": {
                "harmless_kl": summarize(full_safe_kl),
                "refusal_score_effect": full_effects,
                "intervention": full_intervention,
            },
            "results": results,
            "ranked": ranked,
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_report": False,
            "system": system_record(),
        },
    )
    print(json.dumps({"status": "complete", "report": str(report_path), "ranked": ranked}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
