#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    read_prompt_jsonl,
    require_acknowledgement,
    require_large_gpu,
    system_record,
    write_json,
)
from swift_abliteration.live_model import (
    generate_responses_with_first_logits,
    validate_live_model,
)
from swift_abliteration.response_quality import analyze_response, summarize_quality


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a non-raw generation smoke test on a saved release checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()

    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if args.count <= 0:
        raise ValueError("Count must be positive.")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    prompts, prompt_file_sha256 = read_prompt_jsonl(args.prompts)
    prompts = prompts[: args.count]
    if len(prompts) != args.count:
        raise ValueError(f"Expected {args.count} prompts, found {len(prompts)}.")

    checkpoint = args.checkpoint.resolve()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    validate_live_model(model, cfg)
    generated = generate_responses_with_first_logits(
        model,
        processor,
        prompts,
        max_new_tokens=args.max_new_tokens,
        system_prompt="You are a helpful assistant.",
        batch_size=args.batch_size,
    )
    quality_items = [analyze_response(value) for value in generated["responses"]]
    quality = summarize_quality(quality_items)
    if quality["empty_output_count"] or quality["severe_repetition_count"]:
        raise RuntimeError(f"Saved checkpoint generation failed quality checks: {quality}")

    result = {
        "status": "pass",
        "checkpoint": str(checkpoint),
        "prompt_source": str(args.prompts),
        "prompt_file_sha256": prompt_file_sha256,
        "prompt_count": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "response_sha256": [text_sha256(value) for value in generated["responses"]],
        "first_token_ids": generated["first_token_ids"],
        "quality": quality,
        "raw_prompt_text_recorded": False,
        "raw_response_text_recorded": False,
        "system": system_record(),
    }
    write_json(args.output, result)
    print(json.dumps({"status": "pass", "quality": quality}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
