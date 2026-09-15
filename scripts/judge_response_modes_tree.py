#!/usr/bin/env python3
"""Judge saved responses with a pinned local open-weight instruction model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from swift_abliteration.gpu_support import require_large_gpu, write_json
from swift_abliteration.response_mode_judge import (
    RUBRIC_VARIANTS,
    majority,
    parse_mode_quality,
    render_rubric,
)


DEFAULT_GROUPS = ("standard_harmful", "matched_harmful")
ALLOWED_GROUPS = (
    "standard_harmful",
    "matched_harmful",
    "standard_harmless",
    "matched_harmless",
    "xstest_safe",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    rows.sort(key=lambda row: str(row["id"]))
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


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
    modes = ("DIRECT", "PARTIAL", "SOFT_REFUSAL", "HARD_REFUSAL", "INCOHERENT")
    return {
        "count": len(rows),
        "mode_counts": {
            str(mode): sum(row["judgment"]["mode"] == mode for row in rows)
            for mode in (*modes, None)
        },
        "broken_count": sum(
            row["judgment"]["quality"] == "BROKEN" for row in rows
        ),
        "parse_error_count": sum(
            row["judgment"]["parse_error"] for row in rows
        ),
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "resumed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", action="append")
    parser.add_argument("--group", action="append")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554"
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete per-arm judgment files after validating their IDs.",
    )
    args = parser.parse_args()
    # The pinned 4B BF16 judge fits well below the target-model memory guard.
    require_large_gpu(12 * 1024**3)
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
    unknown = sorted(set(groups).difference(ALLOWED_GROUPS))
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
        tasks = [
            (row, variant)
            for row in source_rows
            for variant in range(len(RUBRIC_VARIANTS))
        ]
        votes: dict[str, list[dict]] = {str(row["id"]): [] for row in source_rows}
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start : start + args.batch_size]
            prompts = []
            for row, variant in batch:
                content = render_rubric(variant, row["prompt"], row["response"])
                prompts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
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
                    max_new_tokens=16,
                    pad_token_id=tokenizer.pad_token_id,
                )
            input_width = int(encoded["input_ids"].shape[1])
            for (row, variant), token_ids in zip(batch, generated, strict=True):
                decoded = tokenizer.decode(
                    token_ids[input_width:], skip_special_tokens=True
                )
                try:
                    mode, quality = parse_mode_quality(decoded)
                    parse_error = False
                except ValueError:
                    mode, quality = None, None
                    parse_error = True
                votes[str(row["id"])].append(
                    {
                        "variant": variant,
                        "mode": mode,
                        "quality": quality,
                        "parse_error": parse_error,
                        "raw_output_sha256": hashlib.sha256(decoded.encode()).hexdigest(),
                    }
                )
        judgments = []
        for row in source_rows:
            row_votes = votes[str(row["id"])]
            mode, mode_agreement = majority(
                [vote["mode"] for vote in row_votes if vote["mode"]]
            )
            quality, quality_agreement = majority(
                [vote["quality"] for vote in row_votes if vote["quality"]]
            )
            judgments.append(
                {
                    "id": row["id"],
                    "judgment": {
                        "mode": mode,
                        "mode_agreement": mode_agreement,
                        "quality": quality,
                        "quality_agreement": quality_agreement,
                        "parse_error": any(vote["parse_error"] for vote in row_votes),
                        "votes": row_votes,
                    },
                    "judge": {
                        "backend": "local_transformers",
                        "model": args.model,
                        "revision": args.revision,
                        "dtype": "bfloat16",
                        "do_sample": False,
                        "rubric_variant_count": len(RUBRIC_VARIANTS),
                    },
                }
            )
        output_sha256 = write_jsonl(output_path, judgments)
        mode_counts = {
            mode: sum(row["judgment"]["mode"] == mode for row in judgments)
            for mode in (*("DIRECT", "PARTIAL", "SOFT_REFUSAL", "HARD_REFUSAL", "INCOHERENT"), None)
        }
        summary = {
            "count": len(judgments),
            "mode_counts": {str(key): value for key, value in mode_counts.items()},
            "broken_count": sum(
                row["judgment"]["quality"] == "BROKEN" for row in judgments
            ),
            "parse_error_count": sum(
                row["judgment"]["parse_error"] for row in judgments
            ),
            "output_sha256": output_sha256,
        }
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
            "rubric_variant_count": len(RUBRIC_VARIANTS),
            "summaries": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
