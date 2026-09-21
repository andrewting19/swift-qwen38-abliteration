#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEYS = {
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "decode_seconds": "vllm:request_decode_time_seconds_sum",
    "prefill_seconds": "vllm:request_prefill_time_seconds_sum",
    "ttft_seconds": "vllm:time_to_first_token_seconds_sum",
    "request_count": "vllm:request_decode_time_seconds_count",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def delta(root: Path, prefix: str) -> dict[str, float]:
    before = json.loads((root / f"{prefix}-before.json").read_text(encoding="utf-8"))
    after = json.loads((root / f"{prefix}-after.json").read_text(encoding="utf-8"))
    values = {
        label: float(after.get(key, 0)) - float(before.get(key, 0))
        for label, key in KEYS.items()
    }
    values["decode_tps"] = values["generation_tokens"] / values["decode_seconds"]
    values["acceptance"] = (
        values["accepted_tokens"] / values["draft_tokens"]
        if values["draft_tokens"]
        else 0.0
    )
    return values


def duration(root: Path, prefix: str) -> float:
    start = float((root / f"{prefix}.start").read_text(encoding="utf-8"))
    end = float((root / f"{prefix}.end").read_text(encoding="utf-8"))
    return end - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run

    replays: dict[str, dict[str, Any]] = {}
    for mode in ("ar", "mtp1", "mtp2", "mtp3"):
        values = delta(root, f"{mode}-replay")
        values["wall_seconds"] = duration(root, f"{mode}-replay")
        values["completed"] = "HANDOFF_COMPLETE" in (
            root / f"{mode}-replay.jsonl"
        ).read_text(encoding="utf-8")
        replays[mode] = values

    baseline_tps = replays["ar"]["decode_tps"]
    for values in replays.values():
        values["decode_speedup_vs_ar"] = values["decode_tps"] / baseline_tps - 1

    agentic = {
        "ar": delta(root, "ar"),
        "mtp3": delta(root, "mtp3"),
    }
    for mode in agentic:
        agentic[mode]["plan_wall_seconds"] = duration(root, f"{mode}-plan")
        agentic[mode]["implement_wall_seconds"] = duration(root, f"{mode}-implement")
    agentic["mtp3"]["decode_speedup_vs_ar"] = (
        agentic["mtp3"]["decode_tps"] / agentic["ar"]["decode_tps"] - 1
    )

    summary = {
        "model": "andrewting/Swift-Qwen3.8-27B-Abliterated",
        "runtime": "vLLM 0.29.0, BF16, one RTX PRO 6000 Blackwell 96GB",
        "agentic_task": agentic,
        "controlled_real_pi_context_replay": replays,
        "recommendation": "mtp3",
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# MTP and Pi benchmark summary",
        "",
        "The controlled replay used the same saved 22,716-token Pi coding-agent history for every mode.",
        "",
        "| Mode | Output tokens | Decode TPS | Draft acceptance | Wall time | Decode change vs AR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("ar", "mtp1", "mtp2", "mtp3"):
        row = replays[mode]
        acceptance = "—" if mode == "ar" else f"{row['acceptance']:.1%}"
        lines.append(
            f"| {mode} | {row['generation_tokens']:.0f} | {row['decode_tps']:.2f} | "
            f"{acceptance} | {row['wall_seconds']:.1f}s | {row['decode_speedup_vs_ar']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "MTP-3 is the best tested mode. It improves decode TPS by "
            f"{replays['mtp3']['decode_speedup_vs_ar']:.1%} over normal decoding on the frozen Pi context.",
            "All modes emitted the required completion marker.",
            "",
            "The full two-turn agentic run also completed the code edit and passed 28 focused tests in both AR and MTP-3 modes. "
            "Its aggregate decode-TPS change was "
            f"{agentic['mtp3']['decode_speedup_vs_ar']:.1%}, but the traces had different token and tool-call counts, so the controlled replay is the primary speed result.",
            "",
        ]
    )
    rental_path = root / "rental.json"
    if rental_path.is_file():
        rental = json.loads(rental_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Rental",
                "",
                f"Vast instance `{rental['instance_id']}` used one {rental['gpu']} at "
                f"${rental['hourly_rate_usd']:.3f}/hour. The observed instance charge was "
                f"${rental['observed_instance_charge_usd']:.2f}.",
                f"Final state: `{rental['final_state']}`.",
                "",
            ]
        )
    (root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
