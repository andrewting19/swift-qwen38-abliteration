#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from swift_abliteration.architecture import select_weight_names, validate_public_metadata
from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import sha256_file, write_json
from swift_abliteration.torch_ops import project_output_weight_


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a saved abliterated checkpoint against its pinned base."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--direction-key", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    cfg = load_config(args.config)
    checkpoint = args.checkpoint.resolve()
    manifest = json.loads(
        (checkpoint / "abliteration_manifest.json").read_text(encoding="utf-8")
    )
    directions = load_file(args.directions, device="cpu")
    direction = directions[args.direction_key].float()
    basis = torch.linalg.qr(direction.transpose(0, 1), mode="reduced").Q.transpose(0, 1)

    base = Path(snapshot_download(repo_id=cfg.model.id, revision=cfg.model.revision))
    model_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    metadata = validate_public_metadata(cfg, model_config, index)
    layer_types = model_config.get("text_config", model_config)["layer_types"]
    planned = set(select_weight_names(cfg, layer_types, set(index["weight_map"])))
    manifest_edited = set(manifest["edited_tensors"])
    if manifest_edited != planned:
        raise RuntimeError("Manifest edited tensors differ from the configured plan.")
    if manifest["direction_key"] != args.direction_key:
        raise RuntimeError("Manifest direction key differs from the requested key.")
    if manifest["direction_file_sha256"] != sha256_file(args.directions):
        raise RuntimeError("Manifest direction hash differs from the supplied file.")

    shard_names = sorted(set(index["weight_map"].values()))
    all_tensor_count = 0
    edited_count = 0
    unedited_count = 0
    nonfinite_count = 0
    max_projection_abs = 0.0
    max_projection_relative = 0.0
    shard_hashes = {}

    for shard_name in shard_names:
        candidate_path = checkpoint / shard_name
        expected_hash = manifest["checkpoint_shard_sha256"][shard_name]
        actual_hash = sha256_file(candidate_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Shard hash mismatch: {shard_name}")
        shard_hashes[shard_name] = actual_hash

        base_tensors = load_file(base / shard_name, device="cpu")
        candidate_tensors = load_file(candidate_path, device="cpu")
        if set(base_tensors) != set(candidate_tensors):
            raise RuntimeError(f"Tensor names differ in shard: {shard_name}")
        for name in sorted(candidate_tensors):
            source = base_tensors[name]
            saved = candidate_tensors[name]
            all_tensor_count += 1
            if source.shape != saved.shape or source.dtype != saved.dtype:
                raise RuntimeError(f"Tensor metadata differs: {name}")
            if saved.is_floating_point() and not torch.isfinite(saved).all():
                nonfinite_count += 1
            if name not in planned:
                if not torch.equal(source, saved):
                    raise RuntimeError(f"Unplanned tensor changed: {name}")
                unedited_count += 1
                continue

            expected = source.clone()
            project_output_weight_(expected, direction, cfg.edit.alpha)
            if not torch.equal(expected, saved):
                raise RuntimeError(f"Saved tensor differs from exact edit: {name}")
            residual = basis @ saved.float()
            max_projection_abs = max(
                max_projection_abs, float(residual.abs().max().item())
            )
            denominator = float(saved.float().norm().item())
            if denominator:
                max_projection_relative = max(
                    max_projection_relative,
                    float(residual.norm().item()) / denominator,
                )
            edited_count += 1

    if nonfinite_count:
        raise RuntimeError(f"Found {nonfinite_count} tensors with non-finite values.")
    forbidden = sorted(
        name
        for name in planned
        if "embed_tokens" in name
        or "mtp" in name.lower()
        or "visual" in name.lower()
        or "vision" in name.lower()
    )
    if forbidden:
        raise RuntimeError(f"Forbidden release tensors were edited: {forbidden}")

    result = {
        "status": "pass",
        "checkpoint": str(checkpoint),
        "base_model": cfg.model.id,
        "base_revision": cfg.model.revision,
        "direction_key": args.direction_key,
        "direction_file_sha256": sha256_file(args.directions),
        "rank": int(basis.shape[0]),
        "all_tensor_count": all_tensor_count,
        "edited_tensor_count": edited_count,
        "unedited_tensor_count": unedited_count,
        "nonfinite_tensor_count": nonfinite_count,
        "maximum_projection_residual_absolute": max_projection_abs,
        "maximum_projection_residual_relative": max_projection_relative,
        "checkpoint_shard_sha256": shard_hashes,
        "public_metadata_validation": metadata,
        "embedding_edited": False,
        "mtp_edited": False,
        "vision_edited": False,
        "raw_prompt_text_used": False,
    }
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "status",
        "rank",
        "edited_tensor_count",
        "unedited_tensor_count",
        "maximum_projection_residual_absolute",
        "maximum_projection_residual_relative",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
