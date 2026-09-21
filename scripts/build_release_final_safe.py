#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = (
    ("standard_harmless", Path("data/prepared/final_test_harmless.jsonl")),
    ("matched_harmless", Path("data/prepared/matched/final_test_harmless.jsonl")),
    ("xstest_safe", Path("data/prepared/xstest-safe/final_test.jsonl")),
)


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine sealed final safe groups without printing prompt text."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    combined = []
    sources = {}
    for group, path in SOURCES:
        rows = read_rows(path)
        if len(rows) != 64:
            raise ValueError(f"Expected 64 rows in {path}, found {len(rows)}.")
        sources[group] = {
            "path": str(path),
            "count": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for index, row in enumerate(rows):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing text in {path} row {index}.")
            combined.append(
                {
                    "id": f"{group}:{index}",
                    "source_group": group,
                    "source_index": index,
                    "text": text,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in combined
    )
    args.output.write_text(payload, encoding="utf-8")
    report = {
        "output": str(args.output),
        "count": len(combined),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "sources": sources,
        "prompt_text_printed": False,
    }
    report_path = args.output.with_suffix(".manifest.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
