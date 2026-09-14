#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tomllib
from pathlib import Path
from urllib.request import urlopen


def text_from_row(row: dict, columns: list[str]) -> str:
    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"No usable text in columns {columns}; available columns: {sorted(row)}"
    )


def write_jsonl(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def load_pinned_rows(source: dict) -> list[dict]:
    with urlopen(source["source_url"], timeout=60) as response:
        content = response.read()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != source["source_sha256"]:
        raise ValueError(
            f"Source hash mismatch for {source['dataset']}: "
            f"expected {source['source_sha256']}, found {actual_hash}"
        )
    if source["source_format"] == "csv":
        return list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    if source["source_format"] == "json":
        rows = json.loads(content)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list from {source['dataset']}.")
        return rows
    raise ValueError(f"Unsupported source format: {source['source_format']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", default="data/splits.toml")
    parser.add_argument("--output-dir", default="data/prepared")
    args = parser.parse_args()
    with Path(args.split_file).open("rb") as handle:
        spec = tomllib.load(handle)
    output = Path(args.output_dir)
    manifest = {"split_file": args.split_file, "files": {}}
    review_groups: list[tuple[str, list[dict]]] = []
    for label in ("harmful", "harmless"):
        source = spec[label]
        dataset = load_pinned_rows(source)
        for purpose in ("direction", "evaluation"):
            indices = source[f"{purpose}_indices"]
            records = [
                {
                    "source_index": index,
                    "text": text_from_row(dataset[index], source["text_columns"]),
                }
                for index in indices
            ]
            path = output / f"{purpose}_{label}.jsonl"
            review_groups.append((f"{purpose.title()} — {label.title()}", records))
            manifest["files"][str(path)] = {
                "count": len(records),
                "sha256": write_jsonl(path, records),
                "dataset": source["dataset"],
                "revision": source["revision"],
                "source_url": source["source_url"],
                "source_sha256": source["source_sha256"],
            }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    review_lines = [
        "# Exact Prompt Sets",
        "",
        "This file is generated from pinned sources. It is excluded from Git because it contains raw harmful requests.",
        "",
    ]
    for heading, records in review_groups:
        review_lines.extend([f"## {heading}", ""])
        for number, record in enumerate(records, start=1):
            review_lines.extend(
                [
                    f"{number}. Source row `{record['source_index']}`",
                    "",
                    f"   {record['text']}",
                    "",
                ]
            )
    (output / "PROMPT_SETS.md").write_text("\n".join(review_lines), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
