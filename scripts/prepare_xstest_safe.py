#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from swift_abliteration.gpu_support import sha256_file, write_json


SOURCE_COMMIT = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
SOURCE_URL = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/"
    f"{SOURCE_COMMIT}/xstest_prompts.csv"
)
SOURCE_SHA256 = "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"
EXPECTED_SAFE_COUNT = 250
PARTITION_SIZES = {
    "selection": 64,
    "validation": 64,
    "final_test": 64,
    "audit_remainder": 58,
}


def stable_seed(seed: int, name: str) -> int:
    payload = f"{seed}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stratified_order(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    types = sorted(by_type)
    random.Random(stable_seed(seed, "type-order")).shuffle(types)
    for name, values in by_type.items():
        random.Random(stable_seed(seed, name)).shuffle(values)
    ordered = []
    depth = max(len(values) for values in by_type.values())
    for index in range(depth):
        for name in types:
            if index < len(by_type[name]):
                ordered.append(by_type[name][index])
    return ordered


def parse_safe_rows(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    required = {"id", "prompt", "type", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("The XSTest source columns do not match the pinned schema.")
    safe = [row for row in rows if row["label"] == "safe"]
    if len(safe) != EXPECTED_SAFE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAFE_COUNT} safe rows, found {len(safe)}."
        )
    if any(not row["prompt"].strip() for row in safe):
        raise ValueError("A safe XSTest row has an empty prompt.")
    if len({row["id"] for row in safe}) != len(safe):
        raise ValueError("Safe XSTest IDs are not unique.")
    return safe


def write_jsonl_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare safe-only, disjoint XSTest evaluation partitions."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3819)
    args = parser.parse_args()

    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"XSTest source hash mismatch: {digest}")
    safe = parse_safe_rows(payload)
    ordered = stratified_order(safe, args.seed)

    cursor = 0
    records = {}
    all_ids: set[str] = set()
    for partition, size in PARTITION_SIZES.items():
        selected = ordered[cursor : cursor + size]
        cursor += size
        ids = {row["id"] for row in selected}
        if all_ids.intersection(ids):
            raise RuntimeError("XSTest partitions overlap.")
        all_ids.update(ids)
        output_rows = [
            {
                "id": f"xstest-{row['id']}",
                "text": row["prompt"],
                "type": row["type"],
                "source_id": row["id"],
            }
            for row in selected
        ]
        path = args.output_dir / f"{partition}.jsonl"
        write_jsonl_atomic(path, output_rows)
        records[partition] = {
            "path": str(path),
            "count": len(output_rows),
            "sha256": sha256_file(path),
            "type_counts": dict(
                sorted(Counter(row["type"] for row in selected).items())
            ),
        }
    if cursor != len(ordered) or len(all_ids) != EXPECTED_SAFE_COUNT:
        raise RuntimeError("XSTest partition sizes do not cover every safe row once.")

    write_json(
        args.output_dir / "manifest.json",
        {
            "dataset": "XSTest",
            "purpose": "safe prompts designed to test exaggerated safety refusal",
            "license": "CC-BY-4.0",
            "source_repository": "https://github.com/paul-rottger/xstest",
            "source_commit": SOURCE_COMMIT,
            "source_url": SOURCE_URL,
            "source_sha256": SOURCE_SHA256,
            "seed": args.seed,
            "partition_method": "seeded within-type shuffle, then balanced round-robin",
            "safe_rows_only": True,
            "unsafe_rows_written": False,
            "fully_answerable_by_dataset_design": True,
            "records": records,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "safe_count": len(safe),
                "partitions": PARTITION_SIZES,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
