#!/usr/bin/env python3
"""Verify that a GGUF contains the requested exact tensor quantization map."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


def read_exact_type_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, tensor_type = line.rsplit("=", 1)
        match = re.fullmatch(r"\^(.+)\$", pattern)
        if not match:
            raise ValueError(f"line {line_number} is not an exact anchored pattern")
        name = re.sub(r"\\(.)", r"\1", match.group(1))
        result[name] = tensor_type.upper()
    return result


def compare_types(
    requested: dict[str, str], actual: dict[str, str]
) -> list[tuple[str, str, str | None]]:
    return [
        (name, expected, actual.get(name))
        for name, expected in requested.items()
        if actual.get(name) != expected
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("type_map", type=Path)
    parser.add_argument("--expected-tensors", type=int, default=866)
    parser.add_argument("--gguf-python-path", type=Path)
    args = parser.parse_args()

    if args.gguf_python_path:
        sys.path.insert(0, str(args.gguf_python_path))
    from gguf import GGUFReader

    reader = GGUFReader(args.gguf, "r")
    actual = {tensor.name: tensor.tensor_type.name for tensor in reader.tensors}
    requested = read_exact_type_map(args.type_map)
    errors = compare_types(requested, actual)
    counts = Counter(actual.values())

    print(f"gguf={args.gguf}")
    print(f"tensors={len(actual)}")
    print(f"mapped={len(requested)}")
    print("types=" + ",".join(f"{kind}:{counts[kind]}" for kind in sorted(counts)))
    if len(actual) != args.expected_tensors:
        print(f"error=expected {args.expected_tensors} tensors", file=sys.stderr)
        return 1
    if errors:
        for name, expected, found in errors[:20]:
            print(f"mismatch={name}: expected {expected}, found {found}", file=sys.stderr)
        print(f"error={len(errors)} tensor type mismatches", file=sys.stderr)
        return 1
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
