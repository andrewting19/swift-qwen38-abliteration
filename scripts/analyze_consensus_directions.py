#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from swift_abliteration.direction_study import (
    bootstrap_consensus_stability,
    cosine_similarity,
    normalized_average,
    standardized_separation,
)
from swift_abliteration.gpu_support import sha256_file
from swift_abliteration.metrics import summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=38)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3819)
    args = parser.parse_args()

    from safetensors.numpy import load_file, save_file

    activations = load_file(args.capture)
    source_directions = load_file(args.directions)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    layer = args.layer
    sources = ("standard", "matched")
    estimators = {
        "plain": None,
        "winsor_995": 0.995,
    }
    groups = {
        source: (
            activations[f"{source}_direction_harmful_layer_{layer}"],
            activations[f"{source}_direction_harmless_layer_{layer}"],
        )
        for source in sources
    }
    evaluation = {
        source: (
            activations[f"{source}_evaluation_harmful_layer_{layer}"],
            activations[f"{source}_evaluation_harmless_layer_{layer}"],
        )
        for source in sources
    }

    consensus_directions: dict[str, np.ndarray] = {}
    report: dict[str, object] = {
        "layer": layer,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "uses_final_test": False,
        "candidates": {},
    }
    for estimator, winsor_quantile in estimators.items():
        constituent_keys = [
            f"{source}_layer_{layer}_{estimator}" for source in sources
        ]
        constituents = [source_directions[key] for key in constituent_keys]
        consensus = normalized_average(constituents)
        key = f"consensus_layer_{layer}_{estimator}"
        consensus_directions[key] = consensus.astype(np.float32)
        stability = bootstrap_consensus_stability(
            [groups[source] for source in sources],
            consensus,
            samples=args.bootstrap_samples,
            seed=args.seed,
            winsor_quantile=winsor_quantile,
        )
        report["candidates"][key] = {
            "constituent_keys": constituent_keys,
            "constituent_cosine": cosine_similarity(*constituents),
            "cosine_to_constituents": {
                source: cosine_similarity(consensus, direction)
                for source, direction in zip(sources, constituents, strict=True)
            },
            "bootstrap_cosine": summarize(stability),
            "heldout_standardized_separation": {
                source: standardized_separation(
                    evaluation[source][0], evaluation[source][1], consensus
                )
                for source in sources
            },
        }

    plain_key = f"consensus_layer_{layer}_plain"
    winsor_key = f"consensus_layer_{layer}_winsor_995"
    report["cross_candidate_cosine"] = cosine_similarity(
        consensus_directions[plain_key], consensus_directions[winsor_key]
    )
    for key, candidate in report["candidates"].items():
        candidate["cosine_to_other_source_estimators"] = {
            source_key: cosine_similarity(consensus_directions[key], direction)
            for source_key, direction in source_directions.items()
            if source_key.endswith((f"layer_{layer}_plain", f"layer_{layer}_winsor_995"))
        }

    directions_path = output / "consensus_directions.safetensors"
    save_file(consensus_directions, str(directions_path))
    report["directions_sha256"] = sha256_file(directions_path)
    report_path = output / "consensus_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_count": len(consensus_directions),
                "output_dir": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
