#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tomllib
from pathlib import Path


def write_json(path: Path, rows: list[dict]) -> str:
    content = json.dumps(
        {"questions": rows}, ensure_ascii=False, sort_keys=True
    ).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="benchmarks/sources.toml")
    parser.add_argument("--output-dir", default="benchmarks/data")
    parser.add_argument("--seed", type=int, default=3819)
    args = parser.parse_args()
    from datasets import load_dataset

    with Path(args.sources).open("rb") as handle:
        spec = tomllib.load(handle)["datasets"]["wmdp"]
    dataset = load_dataset(
        spec["id"], "wmdp-cyber", split="test", revision=spec["revision"]
    )
    rows = [
        {
            "source_index": index,
            "question": row["question"],
            "choices": row["choices"],
            "answer": int(row["answer"]),
        }
        for index, row in enumerate(dataset)
    ]
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    quick = [rows[index] for index in indices[: min(256, len(indices))]]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    quick_path = output / "WMDP-Cyber-256.json"
    full_path = output / "WMDP-Cyber-full.json"
    report = {
        "dataset": spec["id"],
        "revision": spec["revision"],
        "seed": args.seed,
        "files": {
            "quick": {
                "path": str(quick_path),
                "count": len(quick),
                "sha256": write_json(quick_path, quick),
            },
            "full": {
                "path": str(full_path),
                "count": len(rows),
                "sha256": write_json(full_path, rows),
            },
        },
    }
    manifest_path = output / "wmdp_manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
