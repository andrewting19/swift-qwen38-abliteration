#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np

from swift_abliteration.metrics import forward_kl_from_logits, summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", required=True, help="NPZ file with array named logits."
    )
    parser.add_argument(
        "--edited", required=True, help="NPZ file with array named logits."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Compare only the first N rows from each file.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = np.load(args.base, allow_pickle=False)["logits"]
    edited = np.load(args.edited, allow_pickle=False)["logits"]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("Limit must be positive.")
        base = base[: args.limit]
        edited = edited[: args.limit]
    if base.shape != edited.shape:
        raise ValueError(
            f"Logit shapes differ after limiting: {base.shape} != {edited.shape}"
        )
    values = forward_kl_from_logits(base, edited)
    report = {
        "metric": "last_prompt_token_forward_kl_nats",
        "summary": summarize(values),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
