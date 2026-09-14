#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

from swift_abliteration.judging import (
    COHERENCE_SYSTEM_PROMPT,
    REFUSAL_SYSTEM_PROMPT,
    coherence_user_message,
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
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 256,
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
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with (
        input_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            record = json.loads(line)
            if args.mode == "refusal":
                system = REFUSAL_SYSTEM_PROMPT
                user = refusal_user_message(record["prompt"], record["response"])
                parsed = parse_refusal_judgment(
                    call_api(
                        args.base_url,
                        api_key,
                        args.model,
                        system,
                        user,
                        args.reasoning_effort,
                    )
                )
            else:
                system = COHERENCE_SYSTEM_PROMPT
                user = coherence_user_message(record["prompt"], record["response"])
                parsed = parse_coherence_judgment(
                    call_api(
                        args.base_url,
                        api_key,
                        args.model,
                        system,
                        user,
                        args.reasoning_effort,
                    )
                )
            output_record = {"id": record["id"], "judgment": parsed}
            rendered = (
                json.dumps(output_record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            target.write(rendered)
            digest.update(rendered.encode())
            count += 1
            if args.minimum_delay:
                time.sleep(args.minimum_delay)
    print(
        json.dumps(
            {
                "count": count,
                "output_sha256": digest.hexdigest(),
                "judge_model": args.model,
                "reasoning_effort": args.reasoning_effort,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
