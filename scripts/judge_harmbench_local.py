#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from swift_abliteration.gpu_support import require_large_gpu
from swift_abliteration.open_weight_judge import (
    parse_harmbench_output,
    render_harmbench_llama_prompt,
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify harmful behavior success with local HarmBench weights."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="cais/HarmBench-Llama-2-13b-cls")
    parser.add_argument(
        "--revision", default="bda705349d1144fa618770bea64d99ce54e3835b"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    args = parser.parse_args()
    require_large_gpu()
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = next(model.parameters()).device
    judgments: list[dict] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        active_rows = [row for row in batch_rows if row["response"].strip()]
        for row in batch_rows:
            if not row["response"].strip():
                judgments.append(
                    {
                        "id": row["id"],
                        "judgment": {
                            "behavior_success": False,
                            "parse_error": False,
                            "empty_output": True,
                        },
                        "judge": {
                            "backend": "fixed_empty_output_rule",
                            "model": None,
                            "revision": None,
                        },
                    }
                )
        if not active_rows:
            continue
        prompts = [
            render_harmbench_llama_prompt(row["prompt"], row["response"])
            for row in active_rows
        ]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=4,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_width = encoded["input_ids"].shape[1]
        for row, token_ids in zip(active_rows, generated, strict=True):
            decoded = tokenizer.decode(
                token_ids[input_width:], skip_special_tokens=True
            )
            try:
                behavior_success = parse_harmbench_output(decoded)
                parse_error = False
            except ValueError:
                behavior_success = False
                parse_error = True
            judgments.append(
                {
                    "id": row["id"],
                    "judgment": {
                        "behavior_success": behavior_success,
                        "parse_error": parse_error,
                        "empty_output": False,
                        "raw_output_sha256": hashlib.sha256(decoded.encode()).hexdigest(),
                    },
                    "judge": {
                        "backend": "local_transformers",
                        "model": args.model,
                        "revision": args.revision,
                        "dtype": "bfloat16",
                        "do_sample": False,
                    },
                }
            )

    judgments.sort(key=lambda row: row["id"])
    if [row["id"] for row in judgments] != [row["id"] for row in rows]:
        raise ValueError("Judgment IDs or order differ from input.")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in judgments
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    summary = {
        "count": len(judgments),
        "behavior_success_count": sum(
            row["judgment"]["behavior_success"] for row in judgments
        ),
        "parse_error_count": sum(
            row["judgment"]["parse_error"] for row in judgments
        ),
        "empty_output_count": sum(
            row["judgment"]["empty_output"] for row in judgments
        ),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
