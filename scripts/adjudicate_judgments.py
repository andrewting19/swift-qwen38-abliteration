#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REFUSAL = {"hard_refusal", "soft_refusal"}


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve binary refusal disagreements with a tie-break judgment."
    )
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--tie-break", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    first = read_jsonl(args.first)
    second = read_jsonl(args.second)
    ties = {row["id"]: row for row in read_jsonl(args.tie_break)}
    if [row["id"] for row in first] != [row["id"] for row in second]:
        raise ValueError("Judge IDs or order differ.")
    rendered: list[str] = []
    tie_count = 0
    for left, right in zip(first, second):
        left_refusal = left["judgment"]["label"] in REFUSAL
        right_refusal = right["judgment"]["label"] in REFUSAL
        if left_refusal == right_refusal:
            chosen = left
            method = "two_pass_binary_agreement"
        else:
            if left["id"] not in ties:
                raise ValueError(f"Missing tie-break judgment for id {left['id']}.")
            chosen = ties[left["id"]]
            method = "xhigh_tie_break"
            tie_count += 1
        row = {
            "id": left["id"],
            "judgment": chosen["judgment"],
            "adjudication": {
                "method": method,
                "first_label": left["judgment"]["label"],
                "second_label": right["judgment"]["label"],
            },
        }
        rendered.append(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    payload = "".join(rendered)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "count": len(rendered),
                "tie_break_count": tie_count,
                "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
