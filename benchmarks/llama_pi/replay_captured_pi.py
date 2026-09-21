#!/usr/bin/env python3
"""Replay one captured Pi request with fixed sampling and report safe metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:17070")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--model", default="swift-abliterated-q3")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=424242)
    args = parser.parse_args()

    payload = json.loads(args.capture.read_text(encoding="utf-8"))
    payload["model"] = args.model
    payload["stream"] = False
    payload.pop("stream_options", None)
    payload["max_completion_tokens"] = args.max_tokens
    payload["seed"] = args.seed
    request = urllib.request.Request(
        f"{args.base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {args.api_key_file.read_text().strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=7200) as response:
        result = json.load(response)
    elapsed = time.monotonic() - started
    choice = result["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 3),
                "finish_reason": choice.get("finish_reason"),
                "usage": result.get("usage"),
                "timings": result.get("timings"),
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
                "tool_calls": [
                    call.get("function", {}).get("name")
                    for call in message.get("tool_calls") or []
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
