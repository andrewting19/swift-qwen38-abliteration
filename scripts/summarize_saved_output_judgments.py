#!/usr/bin/env python3
"""Join deduplicated judge outputs to experiment aliases and rank candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from swift_abliteration.gpu_support import write_json


GROUPS = ("standard_harmful", "matched_harmful")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def harmbench_metrics(path: Path) -> dict:
    rows = read_jsonl(path)
    count = len(rows)
    successes = sum(bool(row["judgment"]["behavior_success"]) for row in rows)
    return {
        "count": count,
        "success_count": successes,
        "success_rate": successes / count,
        "parse_error_count": sum(
            bool(row["judgment"]["parse_error"]) for row in rows
        ),
    }


def response_mode_metrics(path: Path) -> dict:
    rows = read_jsonl(path)
    count = len(rows)
    modes = ("DIRECT", "PARTIAL", "SOFT_REFUSAL", "HARD_REFUSAL", "INCOHERENT")
    counts = {
        mode: sum(row["judgment"]["mode"] == mode for row in rows) for mode in modes
    }
    substantive = counts["DIRECT"] + counts["PARTIAL"]
    return {
        "count": count,
        "mode_counts": counts,
        "substantive_count": substantive,
        "substantive_rate": substantive / count,
        "broken_count": sum(
            row["judgment"]["quality"] == "BROKEN" for row in rows
        ),
        "minor_error_count": sum(
            row["judgment"]["quality"] == "MINOR_ERROR" for row in rows
        ),
        "uncertain_mode_count": sum(
            row["judgment"]["mode"] is None for row in rows
        ),
        "parse_error_count": sum(
            bool(row["judgment"]["parse_error"]) for row in rows
        ),
    }


def screen_and_arm(alias: str) -> tuple[str, str]:
    marker = "/raw/"
    if marker not in alias:
        raise ValueError(f"Alias does not contain {marker}: {alias}")
    return tuple(alias.rsplit(marker, 1))  # type: ignore[return-value]


def summarize(
    input_manifest: Path,
    harmbench_root: Path,
    response_mode_root: Path | None,
) -> dict:
    manifest = json.loads(input_manifest.read_text())
    item_metrics: dict[str, dict] = {}
    alias_to_key: dict[str, str] = {}
    for item in manifest["items"]:
        key = item["key"]
        group_metrics = {}
        for group in GROUPS:
            metrics = {
                "harmbench": harmbench_metrics(harmbench_root / key / f"{group}.jsonl")
            }
            if response_mode_root is not None:
                metrics["response_mode"] = response_mode_metrics(
                    response_mode_root / key / f"{group}.jsonl"
                )
            group_metrics[group] = metrics
        item_metrics[key] = {
            "aliases": item["aliases"],
            "groups": group_metrics,
        }
        for alias in item["aliases"]:
            alias_to_key[alias] = key

    candidates = []
    for alias, key in sorted(alias_to_key.items()):
        screen, arm = screen_and_arm(alias)
        if arm.startswith("base"):
            continue
        base_alias = f"{screen}/raw/base"
        base_key = alias_to_key.get(base_alias)
        if base_key is None:
            raise ValueError(f"Missing sibling base arm for {alias}")
        groups = {}
        for group in GROUPS:
            current = item_metrics[key]["groups"][group]
            base = item_metrics[base_key]["groups"][group]
            joined = {
                "harmbench": current["harmbench"],
                "base_harmbench": base["harmbench"],
                "harmbench_rate_change": current["harmbench"]["success_rate"]
                - base["harmbench"]["success_rate"],
            }
            if response_mode_root is not None:
                joined.update(
                    response_mode=current["response_mode"],
                    base_response_mode=base["response_mode"],
                    substantive_rate_change=current["response_mode"][
                        "substantive_rate"
                    ]
                    - base["response_mode"]["substantive_rate"],
                )
            groups[group] = joined
        candidate = {
            "alias": alias,
            "key": key,
            "base_alias": base_alias,
            "base_key": base_key,
            "groups": groups,
            "minimum_harmbench_success_rate": min(
                groups[group]["harmbench"]["success_rate"] for group in GROUPS
            ),
            "minimum_harmbench_rate_change": min(
                groups[group]["harmbench_rate_change"] for group in GROUPS
            ),
        }
        if response_mode_root is not None:
            candidate.update(
                minimum_substantive_rate=min(
                    groups[group]["response_mode"]["substantive_rate"]
                    for group in GROUPS
                ),
                minimum_substantive_rate_change=min(
                    groups[group]["substantive_rate_change"] for group in GROUPS
                ),
                total_broken_count=sum(
                    groups[group]["response_mode"]["broken_count"] for group in GROUPS
                ),
            )
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -item.get("minimum_substantive_rate_change", -1),
            -item["minimum_harmbench_rate_change"],
            -item.get("minimum_substantive_rate", -1),
            -item["minimum_harmbench_success_rate"],
            item.get("total_broken_count", 0),
            item["alias"],
        )
    )
    source_hashes = {
        "input_manifest_sha256": hashlib.sha256(input_manifest.read_bytes()).hexdigest(),
        "harmbench_manifest_sha256": hashlib.sha256(
            (harmbench_root / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    if response_mode_root is not None:
        source_hashes["response_mode_manifest_sha256"] = hashlib.sha256(
            (response_mode_root / "manifest.json").read_bytes()
        ).hexdigest()
    return {
        "schema_version": 1,
        "raw_text_in_report": False,
        "remote_api_used": False,
        "openai_models_used": False,
        "candidate_alias_count": len(candidates),
        "unique_candidate_count": len(item_metrics),
        "source_hashes": source_hashes,
        "ranked_candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--harmbench-root", type=Path, required=True)
    parser.add_argument("--response-mode-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        args.input_manifest, args.harmbench_root, args.response_mode_root
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_alias_count": report["candidate_alias_count"],
                "unique_candidate_count": report["unique_candidate_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
