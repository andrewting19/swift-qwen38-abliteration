#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from swift_abliteration.architecture import (
    select_weight_names,
    validate_public_metadata,
)
from swift_abliteration.checkpoint_edit import edit_shard_tensors
from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)


def copy_non_weight_files(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix == ".safetensors":
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target, follow_symlinks=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--direction-key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    cfg = load_config(args.config)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    directions = load_file(args.directions, device="cpu")
    if args.direction_key not in directions:
        raise ValueError(f"Direction key not found: {args.direction_key}")
    direction = directions[args.direction_key]

    source = Path(snapshot_download(repo_id=cfg.model.id, revision=cfg.model.revision))
    model_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    validation = validate_public_metadata(cfg, model_config, index)
    validation["full_weights_downloaded"] = True
    text_config = model_config.get("text_config", model_config)
    planned = set(
        select_weight_names(
            cfg,
            text_config["layer_types"],
            set(index["weight_map"]),
        )
    )
    copy_non_weight_files(source, destination)
    edited = []
    shard_hashes = {}
    shard_names = sorted(set(index["weight_map"].values()))
    for shard_name in shard_names:
        source_shard = source / shard_name
        tensors = load_file(source_shard, device="cpu")
        with safe_open(source_shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        edited.extend(edit_shard_tensors(tensors, planned, direction, cfg.edit.alpha))
        target_shard = destination / shard_name
        target_shard.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, target_shard, metadata=metadata)
        shard_hashes[shard_name] = sha256_file(target_shard)
    if set(edited) != planned:
        missing = sorted(planned - set(edited))
        raise RuntimeError(f"Not all planned tensors were edited: {missing}")
    write_json(
        destination / "abliteration_manifest.json",
        {
            "config": asdict(cfg),
            "direction_key": args.direction_key,
            "direction_file": str(Path(args.directions).resolve()),
            "direction_file_sha256": sha256_file(args.directions),
            "source_snapshot": str(source),
            "edited_tensor_count": len(edited),
            "edited_tensors": sorted(edited),
            "checkpoint_shard_sha256": shard_hashes,
            "public_metadata_validation": validation,
            "system": system_record(),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "edited_tensor_count": len(edited),
                "output_dir": str(destination),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
