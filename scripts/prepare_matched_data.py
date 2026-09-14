#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from urllib.request import urlopen


def write_jsonl(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", default="data/matched_splits.toml")
    parser.add_argument("--output-dir", default="data/prepared/matched")
    args = parser.parse_args()
    with Path(args.split_file).open("rb") as handle:
        spec = tomllib.load(handle)
    with urlopen(spec["source_url"], timeout=60) as response:
        content = response.read()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != spec["source_sha256"]:
        raise ValueError(f"Matched source hash mismatch: {actual_hash}")
    source = json.loads(content)
    pairs = source["pairs"]
    if len(pairs) != spec["pair_count"]:
        raise ValueError(f"Expected {spec['pair_count']} pairs, found {len(pairs)}.")

    direction_ids = spec["direction_pair_indices"]
    evaluation_ids = spec["evaluation_pair_indices"]
    if set(direction_ids) & set(evaluation_ids):
        raise ValueError("Direction and evaluation pair IDs overlap.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": spec["dataset"],
        "revision": spec["revision"],
        "source_sha256": spec["source_sha256"],
        "source_metadata": source.get("metadata", {}),
        "files": {},
    }
    score_report = {}
    for purpose, pair_ids in (
        ("direction", direction_ids),
        ("evaluation", evaluation_ids),
    ):
        chosen = [pairs[index] for index in pair_ids]
        score_report[purpose] = {
            "count": len(chosen),
            "minimum": min(float(pair["score"]) for pair in chosen),
            "maximum": max(float(pair["score"]) for pair in chosen),
            "mean": sum(float(pair["score"]) for pair in chosen) / len(chosen),
        }
        for label in ("harmful", "harmless"):
            rows = [
                {
                    "pair_index": pair_index,
                    "source_index": pair[f"{label}_index"],
                    "text": pair[label],
                }
                for pair_index, pair in zip(pair_ids, chosen, strict=True)
            ]
            path = output / f"{purpose}_{label}.jsonl"
            manifest["files"][path.name] = {
                "count": len(rows),
                "sha256": write_jsonl(path, rows),
            }
    manifest["semantic_match_scores"] = score_report
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Do not print prompts. The command output is safe for logs.
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
