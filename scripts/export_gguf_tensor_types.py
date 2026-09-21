#!/usr/bin/env python3
"""Export exact per-tensor quantization types from a reference GGUF file."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


def exact_pattern(name: str) -> str:
    """Return a regex that matches only one GGUF tensor name."""

    return f"^{re.escape(name)}$"


def render_type_map(tensors: list[tuple[str, str]]) -> str:
    """Render llama-quantize tensor-type-file entries."""

    return "\n".join(
        f"{exact_pattern(name)}={tensor_type}"
        for name, tensor_type in tensors
        if tensor_type != "F32"
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--gguf-python-path",
        type=Path,
        help="Path that contains the gguf Python package",
    )
    args = parser.parse_args()

    if args.gguf_python_path:
        sys.path.insert(0, str(args.gguf_python_path))

    try:
        from gguf import GGUFReader
    except ImportError as error:
        parser.error(f"cannot import gguf: {error}")

    reader = GGUFReader(args.reference, "r")
    tensors = [(tensor.name, tensor.tensor_type.name) for tensor in reader.tensors]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_type_map(tensors), encoding="utf-8")

    counts = Counter(tensor_type for _, tensor_type in tensors)
    print(f"reference={args.reference}")
    print(f"tensors={len(tensors)}")
    print(f"mapped={sum(count for kind, count in counts.items() if kind != 'F32')}")
    print("types=" + ",".join(f"{kind}:{counts[kind]}" for kind in sorted(counts)))
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
