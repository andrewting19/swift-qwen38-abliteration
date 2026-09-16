#!/usr/bin/env python3
"""Generate a fixed response screen through a local OpenAI-compatible API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


GROUP_PATHS = {
    "standard_harmful": "data/prepared/evaluation_harmful.jsonl",
    "matched_harmful": "data/prepared/matched/evaluation_harmful.jsonl",
    "xstest_safe": "data/prepared/xstest-safe/selection.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                handle.write(line)
                digest.update(line.encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return digest.hexdigest()


def read_slice(path: Path, offset: int, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = rows[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(f"Expected {limit} rows at offset {offset} from {path}.")
    for row in selected:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"A row in {path} has no prompt text.")
    return selected


def request_one(
    url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                data=encoded,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            return result["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError):
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("Unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--harmful-limit", type=int, default=16)
    parser.add_argument("--safe-limit", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    if min(args.harmful_limit, args.safe_limit, args.max_tokens, args.workers) <= 0:
        raise ValueError("Limits, token count, and workers must be positive.")

    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    args.output_dir.mkdir(parents=True)
    groups: dict[str, list[dict]] = {}
    sources = {}
    for group, source_name in GROUP_PATHS.items():
        path = Path(source_name)
        limit = args.safe_limit if group == "xstest_safe" else args.harmful_limit
        groups[group] = read_slice(path, args.offset, limit)
        sources[group] = {
            "path": source_name,
            "sha256": sha256_file(path),
            "offset": args.offset,
            "count": limit,
            "ids_sha256": hashlib.sha256(
                "\n".join(str(row.get("id")) for row in groups[group]).encode()
            ).hexdigest(),
        }

    started = time.time()
    output_summary = {}
    for group, rows in groups.items():
        responses: list[str | None] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_indices = {
                pool.submit(
                    request_one,
                    endpoint,
                    args.model,
                    args.system_prompt,
                    row["text"],
                    args.max_tokens,
                    args.timeout,
                    args.retries,
                ): index
                for index, row in enumerate(rows)
            }
            for future in as_completed(future_indices):
                responses[future_indices[future]] = future.result()
        if any(value is None for value in responses):
            raise RuntimeError(f"Missing response in group {group}.")
        records = [
            {"id": row.get("id", index), "prompt": row["text"], "response": response}
            for index, (row, response) in enumerate(zip(rows, responses, strict=True))
        ]
        path = args.output_dir / "raw" / "candidate" / f"{group}.jsonl"
        output_summary[group] = {
            "count": len(records),
            "empty_count": sum(not str(value).strip() for value in responses),
            "sha256": atomic_jsonl(path, records),
        }
        print(
            json.dumps(
                {
                    "group": group,
                    "count": len(records),
                    "empty_count": output_summary[group]["empty_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "experiment": "local_api_behavior_screen",
        "model": {"id": args.model, "revision": args.model_revision},
        "endpoint": "local_openai_compatible",
        "generation": {
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "enable_thinking": False,
            "system_prompt": args.system_prompt,
            "workers": args.workers,
        },
        "sources": sources,
        "outputs": output_summary,
        "elapsed_seconds": time.time() - started,
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_manifest": False,
    }
    temporary = args.output_dir / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
