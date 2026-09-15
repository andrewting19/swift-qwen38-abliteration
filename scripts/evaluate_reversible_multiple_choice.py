#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.live_model import (
    apply_layerwise_runtime_edit,
    render_prompt,
    text_backbone,
    validate_live_model,
)


LABELS = "ABCDEFGHIJ"


def combine_direction_rows(tensors):
    import torch

    rows = []
    for tensor in tensors:
        if tensor.ndim == 1:
            rows.append(tensor.float())
        elif tensor.ndim == 2:
            rows.extend(tensor.float().unbind(0))
        else:
            raise ValueError("Each direction must be a vector or a row basis.")
    if not rows:
        raise ValueError("At least one direction row is required.")
    matrix = torch.stack(rows)
    q, r = torch.linalg.qr(matrix.transpose(0, 1), mode="reduced")
    if int(torch.linalg.matrix_rank(r).item()) != len(rows):
        raise ValueError("Direction rows are linearly dependent.")
    return q.transpose(0, 1).contiguous()


def normalize_item(item: dict) -> tuple[str, list[str], str, int]:
    if "choices" in item or "options" in item:
        choices = list(item.get("choices", item.get("options")))
        answer_index = int(item.get("answer_index", item.get("answer")))
        target = LABELS[answer_index]
        item_id = int(item.get("source_index", 0))
    else:
        answers = item["answers"]
        choices = [answers[label] for label in LABELS if label in answers]
        target = str(item["solution"])
        item_id = int(item.get("id", 0))
    return str(item["question"]), choices, target, item_id


def evaluate(model, tokenizer, items: list[dict], batch_size: int) -> dict:
    import torch

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    results = []
    device = text_backbone(model).embed_tokens.weight.device
    try:
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            normalized = [normalize_item(item) for item in batch_items]
            prompts = []
            for question, choices, _target, _item_id in normalized:
                options = "\n".join(
                    f"{label}. {choice}" for label, choice in zip(LABELS, choices)
                )
                valid_labels = ", ".join(LABELS[: len(choices)])
                user = (
                    f"{question}\n\n{options}\n\n"
                    f"Return only one letter: {valid_labels}."
                )
                prompts.append(
                    render_prompt(tokenizer, user, "You are a helpful assistant.")
                )
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=8,
                    pad_token_id=tokenizer.pad_token_id,
                )
            input_width = encoded["input_ids"].shape[1]
            for row_index, (question, choices, target, item_id) in enumerate(
                normalized
            ):
                del question, choices
                decoded = tokenizer.decode(
                    generated[row_index, input_width:], skip_special_tokens=True
                )
                match = re.search(r"\b([A-J])\b", decoded.upper())
                predicted = match.group(1) if match else None
                results.append(
                    {
                        "id": item_id if item_id else start + row_index,
                        "predicted": predicted,
                        "target": target,
                        "valid": predicted is not None,
                        "correct": predicted == target,
                    }
                )
    finally:
        tokenizer.padding_side = previous_padding_side
    return {
        "count": len(results),
        "accuracy": sum(row["correct"] for row in results) / len(results),
        "valid_rate": sum(row["valid"] for row in results) / len(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare base and reversible edited model on fixed multiple-choice sets."
    )
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--direction-key", action="append", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset in NAME=PATH form.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-embedding", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dataset_specs = {}
    for value in args.dataset:
        if "=" not in value:
            raise ValueError("Each dataset must use NAME=PATH.")
        name, raw_path = value.split("=", maxsplit=1)
        path = Path(raw_path)
        dataset_specs[name] = {
            "path": path,
            "sha256": sha256_file(path),
            "items": json.loads(path.read_text())["questions"],
        }
    directions = load_file(str(args.directions), device="cpu")
    missing = [key for key in args.direction_key if key not in directions]
    if missing:
        raise KeyError(f"Missing direction keys: {missing}")
    basis = combine_direction_rows([directions[key] for key in args.direction_key])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    processor = AutoProcessor.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    validate_live_model(model, cfg)

    reports = {"base": {}, "candidate": {}}
    for name, spec in dataset_specs.items():
        reports["base"][name] = evaluate(
            model, tokenizer, spec["items"], args.batch_size
        )
        write_json(args.output_dir / f"base_{name}.json", reports["base"][name])

    layers = list(range(cfg.edit.first_layer, cfg.edit.last_layer + 1))
    intervention = apply_layerwise_runtime_edit(
        model,
        cfg,
        {index: basis for index in layers},
        args.alpha,
        embedding_direction=basis if args.include_embedding else None,
    )
    for name, spec in dataset_specs.items():
        reports["candidate"][name] = evaluate(
            model, tokenizer, spec["items"], args.batch_size
        )
        write_json(
            args.output_dir / f"candidate_{name}.json", reports["candidate"][name]
        )

    summary = {
        "model": cfg.model.id,
        "revision": cfg.model.revision,
        "direction_file": str(args.directions.resolve()),
        "direction_file_sha256": sha256_file(args.directions),
        "direction_keys": args.direction_key,
        "alpha": args.alpha,
        "datasets": {
            name: {
                "path": str(spec["path"]),
                "sha256": spec["sha256"],
                "count": len(spec["items"]),
                "base_accuracy": reports["base"][name]["accuracy"],
                "candidate_accuracy": reports["candidate"][name]["accuracy"],
                "base_valid_rate": reports["base"][name]["valid_rate"],
                "candidate_valid_rate": reports["candidate"][name]["valid_rate"],
            }
            for name, spec in dataset_specs.items()
        },
        "intervention": intervention,
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_results": False,
        "system": system_record(),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["datasets"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
