#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib


def text_from_row(row: dict, columns: list[str]) -> str:
    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"No usable text in columns {columns}; available columns: {sorted(row)}")


def write_jsonl(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", default="data/splits.toml")
    parser.add_argument("--output-dir", default="data/prepared")
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install GPU extras first: pip install -e '.[gpu]'") from exc

    with Path(args.split_file).open("rb") as handle:
        spec = tomllib.load(handle)
    output = Path(args.output_dir)
    manifest = {"split_file": args.split_file, "files": {}}
    for label in ("harmful", "harmless"):
        source = spec[label]
        dataset = load_dataset(source["dataset"], revision=source["revision"], split=source["split"])
        for purpose in ("direction", "evaluation"):
            indices = source[f"{purpose}_indices"]
            records = [
                {"source_index": index, "text": text_from_row(dataset[index], source["text_columns"])}
                for index in indices
            ]
            path = output / f"{purpose}_{label}.jsonl"
            manifest["files"][str(path)] = {
                "count": len(records),
                "sha256": write_jsonl(path, records),
                "dataset": source["dataset"],
                "revision": source["revision"],
            }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
