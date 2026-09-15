#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path

import numpy as np

from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    read_prompt_jsonl,
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.live_model import (
    capture_prompt_and_first_output_activations_multi,
    validate_live_model,
)
from swift_abliteration.refusal_scoring import (
    DEFAULT_REFUSAL_OPENERS,
    refusal_scores_from_logits,
    resolve_single_token_ids,
)


GROUPS = {
    "standard_direction_harmful": "data/prepared/direction_harmful.jsonl",
    "standard_direction_harmless": "data/prepared/direction_harmless.jsonl",
    "standard_evaluation_harmful": "data/prepared/evaluation_harmful.jsonl",
    "standard_evaluation_harmless": "data/prepared/evaluation_harmless.jsonl",
    "matched_direction_harmful": "data/prepared/matched/direction_harmful.jsonl",
    "matched_direction_harmless": "data/prepared/matched/direction_harmless.jsonl",
    "matched_evaluation_harmful": "data/prepared/matched/evaluation_harmful.jsonl",
    "matched_evaluation_harmless": "data/prepared/matched/evaluation_harmless.jsonl",
}


def atomic_save_safetensors(tensors: dict, path: Path) -> None:
    from safetensors.torch import save_file

    temporary = path.with_name(f".{path.name}.tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture base rank-1 search inputs at two token positions."
    )
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument(
        "--candidate-config", default="configs/direction_candidates.toml"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    with Path(args.candidate_config).open("rb") as handle:
        candidate_cfg = tomllib.load(handle)
    layers = [int(value) for value in candidate_cfg["study"]["direction_layers"]]
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    groups_dir = args.output_dir / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    tokenizer = getattr(processor, "tokenizer", processor)
    refusal_token_ids = resolve_single_token_ids(tokenizer, DEFAULT_REFUSAL_OPENERS)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    validate_live_model(model, cfg)

    group_records = {}
    for name, source_path in GROUPS.items():
        activation_path = groups_dir / f"{name}.safetensors"
        score_path = groups_dir / f"{name}.npz"
        record_path = groups_dir / f"{name}.json"
        if args.resume and all(
            path.is_file() for path in (activation_path, score_path, record_path)
        ):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("activation_sha256") == sha256_file(
                activation_path
            ) and record.get("score_sha256") == sha256_file(score_path):
                group_records[name] = record
                continue
        prompts, prompt_digest = read_prompt_jsonl(source_path)
        result = capture_prompt_and_first_output_activations_multi(
            model,
            processor,
            prompts,
            layers,
            args.system_prompt,
            args.batch_size,
        )
        tensors = {}
        for position, by_layer in result["activations"].items():
            for layer, values in by_layer.items():
                tensors[f"{position}_layer_{layer}"] = torch.stack(values).contiguous()
        logits = torch.stack(result["first_step_logits"])
        scores = refusal_scores_from_logits(logits, refusal_token_ids).float().numpy()
        arrays = {
            "refusal_scores": scores,
            "first_token_ids": np.asarray(result["first_token_ids"], dtype=np.int64),
        }
        if name == "standard_evaluation_harmless":
            arrays["first_step_logits"] = logits.numpy()
        atomic_save_safetensors(tensors, activation_path)
        atomic_save_npz(score_path, **arrays)
        record = {
            "source_path": source_path,
            "source_sha256": prompt_digest,
            "count": len(prompts),
            "activation_shapes": {
                key: list(value.shape) for key, value in tensors.items()
            },
            "activation_sha256": sha256_file(activation_path),
            "score_sha256": sha256_file(score_path),
            "refusal_positive_count": int((scores > 0).sum()),
            "refusal_negative_count": int((scores < 0).sum()),
            "raw_text_in_record": False,
        }
        write_json(record_path, record)
        group_records[name] = record

    write_json(
        args.output_dir / "capture_manifest.json",
        {
            "config": args.config,
            "candidate_config": args.candidate_config,
            "direction_layers": layers,
            "positions": ["prompt_end", "first_output"],
            "refusal_openers": list(DEFAULT_REFUSAL_OPENERS),
            "refusal_token_ids": refusal_token_ids,
            "filter_rule": "harmful score > 0; harmless score < 0",
            "groups": group_records,
            "uses_final_test": False,
            "checkpoint_saved": False,
            "openai_models_used": False,
            "remote_api_used": False,
            "raw_text_in_manifest": False,
            "system": system_record(),
        },
    )
    print(json.dumps({"status": "complete", "group_count": len(group_records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
