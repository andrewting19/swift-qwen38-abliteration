#!/usr/bin/env python3
"""Deduplicate saved harmful generations for one-pass local judging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


GROUPS = ("standard_harmful", "matched_harmful")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_jsonl(path: Path) -> tuple[int, str]:
    count = 0
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not {"id", "prompt", "response"}.issubset(row):
                raise ValueError(f"Missing required fields: {path}")
            count += 1
            ids.append(str(row["id"]))
    if not count:
        raise ValueError(f"Empty JSONL file: {path}")
    ids_sha256 = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return count, ids_sha256


def discover(run_root: Path) -> dict:
    unique: dict[tuple[str, str], dict] = {}
    alias_count = 0
    for raw_root in sorted(path for path in run_root.rglob("raw") if path.is_dir()):
        for arm in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            paths = {group: arm / f"{group}.jsonl" for group in GROUPS}
            if not all(path.is_file() for path in paths.values()):
                continue
            hashes = tuple(sha256_file(paths[group]) for group in GROUPS)
            metadata = {group: inspect_jsonl(paths[group]) for group in GROUPS}
            key = hashlib.sha256("".join(hashes).encode()).hexdigest()[:20]
            item = unique.setdefault(
                hashes,
                {
                    "key": f"candidate_{key}",
                    "aliases": [],
                    "groups": {
                        group: {
                            "source": str(paths[group].relative_to(run_root)),
                            "sha256": hashes[index],
                            "count": metadata[group][0],
                            "ids_sha256": metadata[group][1],
                        }
                        for index, group in enumerate(GROUPS)
                    },
                },
            )
            item["aliases"].append(str(arm.relative_to(run_root)))
            alias_count += 1
    items = sorted(unique.values(), key=lambda item: item["key"])
    if not items:
        raise ValueError(f"No complete harmful response pairs found below {run_root}")
    for item in items:
        item["aliases"].sort()
    source_fingerprint = hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "raw_text_in_manifest": False,
        "groups": list(GROUPS),
        "source_fingerprint": source_fingerprint,
        "alias_count": alias_count,
        "unique_candidate_count": len(items),
        "judgment_count": sum(
            group["count"] for item in items for group in item["groups"].values()
        ),
        "items": items,
    }


def materialize(run_root: Path, output_root: Path, manifest: dict) -> None:
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"Output exists without manifest: {output_root}")
        existing = json.loads(manifest_path.read_text())
        if existing.get("source_fingerprint") != manifest["source_fingerprint"]:
            raise FileExistsError(f"Output has a different source fingerprint: {output_root}")
        return
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        for item in manifest["items"]:
            arm_root = staging / item["key"]
            arm_root.mkdir(parents=True)
            for group, group_data in item["groups"].items():
                shutil.copyfile(run_root / group_data["source"], arm_root / f"{group}.jsonl")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    manifest = discover(run_root)
    materialize(run_root, output_root, manifest)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "alias_count": manifest["alias_count"],
                "unique_candidate_count": manifest["unique_candidate_count"],
                "judgment_count": manifest["judgment_count"],
                "source_fingerprint": manifest["source_fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
