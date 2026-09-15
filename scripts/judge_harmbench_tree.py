#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from swift_abliteration.gpu_support import require_large_gpu, write_json
from swift_abliteration.open_weight_judge import (
    parse_harmbench_output,
    render_harmbench_llama_prompt,
)


DEFAULT_GROUPS = ("standard_harmful", "matched_harmful")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_judgments(path: Path, rows: list[dict]) -> dict:
    rows.sort(key=lambda row: str(row["id"]))
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return {
        "count": len(rows),
        "behavior_success_count": sum(
            row["judgment"]["behavior_success"] for row in rows
        ),
        "parse_error_count": sum(row["judgment"]["parse_error"] for row in rows),
        "empty_output_count": sum(row["judgment"]["empty_output"] for row in rows),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def summarize_existing_judgments(path: Path, source_rows: list[dict]) -> dict | None:
    """Return a summary only when an existing file is complete for this source."""
    try:
        rows = read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return None
    source_ids = {str(row["id"]) for row in source_rows}
    result_ids = {str(row["id"]) for row in rows}
    if len(rows) != len(source_rows) or result_ids != source_ids:
        return None
    payload = path.read_bytes()
    return {
        "count": len(rows),
        "behavior_success_count": sum(
            row["judgment"]["behavior_success"] for row in rows
        ),
        "parse_error_count": sum(row["judgment"]["parse_error"] for row in rows),
        "empty_output_count": sum(
            row["judgment"]["empty_output"] for row in rows
        ),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "resumed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify a response tree with one local HarmBench model load."
    )
    parser.add_argument("--responses-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", action="append")
    parser.add_argument("--group", action="append")
    parser.add_argument("--model", default="cais/HarmBench-Llama-2-13b-cls")
    parser.add_argument(
        "--revision", default="bda705349d1144fa618770bea64d99ce54e3835b"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete per-arm judgment files after validating their IDs.",
    )
    args = parser.parse_args()
    # The pinned 13B BF16 classifier needs about 26 GiB. It does not require the
    # 75 GiB guard used for the 55 GB target model.
    require_large_gpu(28 * 1024**3)
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    arms = args.arm or sorted(
        path.name
        for path in args.responses_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    groups = tuple(args.group or DEFAULT_GROUPS)
    unknown = sorted(set(groups).difference(DEFAULT_GROUPS))
    if unknown:
        raise ValueError(f"Unsupported harmful groups: {unknown}")
    files = [
        (arm, group, args.responses_root / arm / f"{group}.jsonl")
        for arm in arms
        for group in groups
    ]
    missing = [str(path) for _, _, path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing response files: {missing}")

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
    summaries = {}

    for arm, group, path in files:
        source_rows = read_jsonl(path)
        output_path = args.output_root / arm / f"{group}.jsonl"
        if args.resume and output_path.is_file():
            existing_summary = summarize_existing_judgments(output_path, source_rows)
            if existing_summary is not None:
                summaries[f"{arm}/{group}"] = existing_summary
                print(
                    json.dumps(
                        {"arm": arm, "group": group, **existing_summary},
                        sort_keys=True,
                    )
                )
                continue
        judgments = []
        active_rows = []
        for row in source_rows:
            if row["response"].strip():
                active_rows.append(row)
            else:
                judgments.append(
                    {
                        "id": row["id"],
                        "judgment": {
                            "behavior_success": False,
                            "parse_error": False,
                            "empty_output": True,
                        },
                        "judge": {"backend": "fixed_empty_output_rule"},
                    }
                )
        for start in range(0, len(active_rows), args.batch_size):
            batch_rows = active_rows[start : start + args.batch_size]
            prompts = [
                render_harmbench_llama_prompt(row["prompt"], row["response"])
                for row in batch_rows
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
            input_width = int(encoded["input_ids"].shape[1])
            for row, token_ids in zip(batch_rows, generated, strict=True):
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
                            "raw_output_sha256": hashlib.sha256(
                                decoded.encode()
                            ).hexdigest(),
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
        if {str(row["id"]) for row in judgments} != {
            str(row["id"]) for row in source_rows
        }:
            raise ValueError(f"Judgment IDs differ for {arm}/{group}.")
        summary = write_judgments(output_path, judgments)
        summaries[f"{arm}/{group}"] = summary
        print(json.dumps({"arm": arm, "group": group, **summary}, sort_keys=True))

    write_json(
        args.output_root / "manifest.json",
        {
            "backend": "local_transformers",
            "model": args.model,
            "revision": args.revision,
            "remote_api_used": False,
            "openai_models_used": False,
            "raw_text_in_manifest": False,
            "summaries": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
