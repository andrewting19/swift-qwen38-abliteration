#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

REFUSAL = {"hard_refusal", "soft_refusal"}


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select response records whose two judges disagree on refusal."
    )
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    first = read_jsonl(args.first)
    second = read_jsonl(args.second)
    responses = read_jsonl(args.responses)
    if [row["id"] for row in first] != [row["id"] for row in second]:
        raise ValueError("Judge IDs or order differ.")
    selected = {
        left["id"]
        for left, right in zip(first, second)
        if (left["judgment"]["label"] in REFUSAL)
        != (right["judgment"]["label"] in REFUSAL)
    }
    records = [row for row in responses if row["id"] in selected]
    if {row["id"] for row in records} != selected:
        raise ValueError("Response file does not contain every disagreement ID.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    print(json.dumps({"selected_count": len(records), "selected_ids": sorted(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
