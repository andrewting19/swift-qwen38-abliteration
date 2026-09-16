#!/usr/bin/env python3
"""Prepare larger direction-training pools without touching held-out rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tomllib
from pathlib import Path

from scripts.prepare_data import load_pinned_rows, text_from_row, write_jsonl


def used_indices(source: dict) -> set[int]:
    return {
        int(index)
        for purpose in ("direction", "evaluation", "final_test")
        for index in source[f"{purpose}_indices"]
    }


def eligible_indices(label: str, source: dict, rows: list[dict]) -> list[int]:
    excluded = used_indices(source)
    values = []
    for index, row in enumerate(rows):
        if index in excluded:
            continue
        if label == "harmless" and str(row.get("input", "")).strip():
            continue
        try:
            text_from_row(row, source["text_columns"])
        except ValueError:
            continue
        values.append(index)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", default="data/splits.toml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--seed", type=int, default=3819)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("Count must be positive.")

    with Path(args.split_file).open("rb") as handle:
        spec = tomllib.load(handle)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "requested_count_per_group": args.count,
        "uses_evaluation_rows": False,
        "uses_final_test_rows": False,
        "harmless_requires_empty_input": True,
        "groups": {},
    }
    for offset, label in enumerate(("harmful", "harmless")):
        source = spec[label]
        rows = load_pinned_rows(source)
        eligible = eligible_indices(label, source, rows)
        if len(eligible) < args.count:
            raise RuntimeError(f"Only {len(eligible)} eligible {label} rows.")
        generator = random.Random(args.seed + offset)
        indices = sorted(generator.sample(eligible, args.count))
        records = [
            {
                "source_index": index,
                "text": text_from_row(rows[index], source["text_columns"]),
            }
            for index in indices
        ]
        path = output / f"candidates_{label}.jsonl"
        digest = write_jsonl(path, records)
        manifest["groups"][label] = {
            "count": len(records),
            "dataset": source["dataset"],
            "revision": source["revision"],
            "source_sha256": source["source_sha256"],
            "source_index_sha256": hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode()
            ).hexdigest(),
            "output_sha256": digest,
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output),
                "count_per_group": args.count,
                "raw_text_printed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
