from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_open_weight_screen.py"
SPEC = importlib.util.spec_from_file_location("summarize_open_weight_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, judgments: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": f"row-{index}", "judgment": judgment}) + "\n"
            for index, judgment in enumerate(judgments)
        ),
        encoding="utf-8",
    )


def wildguard_row(refusal: bool) -> dict:
    return {
        "response_refusal": refusal,
        "response_harmful": not refusal,
        "parse_error": False,
        "empty_output": False,
    }


def harmbench_row(success: bool) -> dict:
    return {
        "behavior_success": success,
        "parse_error": False,
        "empty_output": False,
    }


def test_build_report_calculates_screen_gates(tmp_path: Path) -> None:
    wildguard_dirs = {}
    harmbench_dirs = {}
    for arm in ("base", "candidate"):
        wildguard_dirs[arm] = tmp_path / "wildguard" / arm
        harmbench_dirs[arm] = tmp_path / "harmbench" / arm
        for group in MODULE.GROUPS:
            if group.endswith("harmful"):
                refusals = [True] * 10 if arm == "base" else [False] * 10
            else:
                refusals = [False] * 10
            write_jsonl(
                wildguard_dirs[arm] / f"{group}.jsonl",
                [wildguard_row(value) for value in refusals],
            )
        for group in MODULE.HARMFUL_GROUPS:
            write_jsonl(
                harmbench_dirs[arm] / f"{group}.jsonl",
                [harmbench_row(arm == "candidate")] * 10,
            )

    kl_path = tmp_path / "candidate-kl.json"
    kl_path.write_text(json.dumps({"summary": {"mean": 0.02}}), encoding="utf-8")
    report = MODULE.build_report(
        wildguard_dirs,
        harmbench_dirs,
        {"candidate": kl_path},
    )

    comparison = report["comparisons_to_base"]["candidate"]
    assert comparison["harmful"]["standard_harmful"][
        "removed_fraction_of_base_refusals"
    ] == 1.0
    assert comparison["gates"]["all_screen_gates"] is True
    assert report["classification_policy"]["openai_models_used"] is False


def test_parse_mapping_rejects_duplicate_names(tmp_path: Path) -> None:
    try:
        MODULE.parse_mapping([f"base={tmp_path}", f"base={tmp_path}"])
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("Expected duplicate-name error.")
