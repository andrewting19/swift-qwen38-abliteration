#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    questions = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["questions"]
    client = OpenAI(
        base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "local")
    )
    results = []
    for index, item in enumerate(questions):
        choices = "\n".join(
            f"{label}. {value}" for label, value in item["answers"].items()
        )
        user = (
            f"{item['question']}\n\n{choices}\n\nReturn only one letter: A, B, C, or D."
        )
        response = client.chat.completions.create(
            model=args.model,
            temperature=0,
            max_tokens=8,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content or ""
        match = re.search(r"\b([ABCD])\b", text.upper())
        predicted = match.group(1) if match else None
        results.append(
            {
                "id": index,
                "predicted": predicted,
                "correct": predicted == item["solution"],
                "valid": predicted is not None,
            }
        )
    correct = sum(item["correct"] for item in results)
    valid = sum(item["valid"] for item in results)
    report = {
        "model": args.model,
        "count": len(results),
        "accuracy": correct / len(results),
        "valid_rate": valid / len(results),
        "results": results,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: report[key] for key in ("model", "count", "accuracy", "valid_rate")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
