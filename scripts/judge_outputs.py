#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from swift_abliteration.judging import (
    COHERENCE_SYSTEM_PROMPT,
    REFUSAL_SYSTEM_PROMPT,
    coherence_user_message,
    deterministic_empty_response_judgment,
    parse_coherence_judgment,
    parse_refusal_judgment,
    refusal_user_message,
)


def call_api(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    reasoning_effort: str | None = None,
    max_completion_tokens: int = 256,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    body = json.dumps(payload).encode()
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["refusal", "coherence"], required=True)
    parser.add_argument(
        "--input", required=True, help="JSONL with id, prompt, and response."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-url", required=True, help="OpenAI-compatible URL ending in /v1."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="minimal")
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--minimum-delay", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    existing_payload = b""
    completed_ids: list[int] = []
    if args.resume and output_path.exists():
        existing_payload = output_path.read_bytes()
        existing_rows = [
            json.loads(line)
            for line in existing_payload.decode("utf-8").splitlines()
            if line
        ]
        completed_ids = [row["id"] for row in existing_rows]
        expected_ids = [row["id"] for row in records[: len(existing_rows)]]
        if completed_ids != expected_ids:
            raise RuntimeError("Existing output is not an ordered prefix of the input.")
        digest.update(existing_payload)
    count = len(completed_ids)
    added_count = 0
    with (
        output_path.open("a" if completed_ids else "w", encoding="utf-8") as target,
    ):
        for record in records[count:]:
            fixed = deterministic_empty_response_judgment(
                args.mode, record["response"]
            )
            if fixed is not None:
                parsed = fixed
                output_record = {"id": record["id"], "judgment": parsed}
                rendered = (
                    json.dumps(output_record, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                target.write(rendered)
                digest.update(rendered.encode())
                count += 1
                added_count += 1
                continue
            if args.mode == "refusal":
                system = REFUSAL_SYSTEM_PROMPT
                user = refusal_user_message(record["prompt"], record["response"])
                parser_fn = parse_refusal_judgment
            else:
                system = COHERENCE_SYSTEM_PROMPT
                user = coherence_user_message(record["prompt"], record["response"])
                parser_fn = parse_coherence_judgment
            last_error: Exception | None = None
            for attempt in range(max(1, args.retries)):
                try:
                    parsed = parser_fn(
                        call_api(
                            args.base_url,
                            api_key,
                            args.model,
                            system,
                            user,
                            args.reasoning_effort,
                            args.max_completion_tokens,
                        )
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < max(1, args.retries):
                        delay = max(1.0, args.minimum_delay)
                        if isinstance(exc, HTTPError) and exc.code == 429:
                            retry_after = exc.headers.get("Retry-After")
                            try:
                                delay = max(delay, float(retry_after))
                            except (TypeError, ValueError):
                                delay = max(delay, float(2**attempt))
                        time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Judge failed for record id {record['id']} after {max(1, args.retries)} attempts: {type(last_error).__name__}"
                ) from last_error
            output_record = {"id": record["id"], "judgment": parsed}
            rendered = (
                json.dumps(output_record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            target.write(rendered)
            digest.update(rendered.encode())
            count += 1
            added_count += 1
            if args.minimum_delay:
                time.sleep(args.minimum_delay)
    print(
        json.dumps(
            {
                "count": count,
                "added_count": added_count,
                "output_sha256": digest.hexdigest(),
                "judge_model": args.model,
                "reasoning_effort": args.reasoning_effort,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
