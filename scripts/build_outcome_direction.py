#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.numpy import load_file, save_file

from swift_abliteration.direction_study import (
    cosine_similarity,
    normalized_average,
    orthogonalize_direction,
    winsorized_direction,
)
from swift_abliteration.gpu_support import sha256_file, write_json


GROUPS = ("standard_harmful", "matched_harmful")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a refusal direction from refused versus answered harmful prompts."
    )
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--basis-key", action="append", required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--judgments-dir", type=Path, required=True)
    parser.add_argument("--winsor-quantile", type=float, default=0.995)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_directions = load_file(args.directions)
    activations = load_file(args.activations)
    missing = [key for key in args.basis_key if key not in source_directions]
    if missing:
        raise KeyError(f"Missing basis directions: {missing}")
    basis = []
    for key in args.basis_key:
        direction, _ = orthogonalize_direction(source_directions[key], basis)
        basis.append(direction)

    source_candidates = []
    source_report = {}
    for group in GROUPS:
        rows = read_jsonl(args.judgments_dir / f"{group}.jsonl")
        labels = {row["id"]: bool(row["judgment"]["response_refusal"]) for row in rows}
        matrix = activations[group]
        if sorted(labels) != list(range(len(matrix))):
            raise ValueError(f"Judgment IDs do not match activations for {group}.")
        refused = matrix[[labels[index] for index in range(len(matrix))]]
        answered = matrix[[not labels[index] for index in range(len(matrix))]]
        if min(len(refused), len(answered)) < 2:
            raise ValueError(f"Outcome contrast is too small for {group}.")
        candidate, threshold, changed = winsorized_direction(
            refused, answered, args.winsor_quantile
        )
        source_candidates.append(candidate)
        source_report[group] = {
            "refused_count": len(refused),
            "answered_count": len(answered),
            "winsor_threshold": threshold,
            "changed_fraction": changed,
        }

    consensus = normalized_average(source_candidates)
    outcome, residual_norm = orthogonalize_direction(consensus, basis)
    output_tensors = {
        **{
            f"basis_direction_{index}": direction
            for index, direction in enumerate(basis, start=1)
        },
        "outcome_refusal_direction": outcome,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "outcome_directions.safetensors"
    save_file(output_tensors, output_path)
    write_json(
        args.output_dir / "outcome_report.json",
        {
            "basis_keys": args.basis_key,
            "sources": source_report,
            "source_direction_cosine": cosine_similarity(
                source_candidates[0], source_candidates[1]
            ),
            "outcome_residual_norm": residual_norm,
            "cosine_to_basis": [
                cosine_similarity(consensus, direction) for direction in basis
            ],
            "directions_sha256": sha256_file(args.directions),
            "activations_sha256": sha256_file(args.activations),
            "output_sha256": sha256_file(output_path),
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_manifest": False,
        },
    )
    print(json.dumps({"status": "complete", "output": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
