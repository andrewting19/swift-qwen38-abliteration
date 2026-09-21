#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a deterministic SHA-256 manifest for an artifact tree."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-name", default="SHA256SUMS")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output_name
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != output
    )
    payload = "".join(
        f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in files
    )
    output.write_text(payload, encoding="utf-8")
    print(f"{output}: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
