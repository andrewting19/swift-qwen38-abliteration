import hashlib
import json

import numpy as np
import pytest

from scripts.screen_generation_subspaces import (
    comparison_metrics,
    flags_for_response,
    move_incomplete,
    valid_completed_arm,
)


def group(anywhere, opening):
    return {
        "arditi_anywhere_refusal_labels": anywhere,
        "opening_refusal_labels": opening,
        "quality": {"empty_output_count": 0, "severe_repetition_count": 0},
    }


def test_empty_response_is_refusal_and_invalid() -> None:
    flags = flags_for_response("   ")
    assert flags["arditi_anywhere_refusal"] is True
    assert flags["opening_refusal"] is True
    assert flags["empty_output"] is True


def test_comparison_uses_only_base_refusal_removal() -> None:
    base = {
        "groups": {
            "standard_harmful": group([True, True, False], [True, True, False]),
            "matched_harmful": group([True, True, True], [True, True, True]),
            "xstest_safe": group([False, False, False], [False, False, False]),
        }
    }
    candidate = {
        "groups": {
            "standard_harmful": group([False, True, True], [False, True, True]),
            "matched_harmful": group([False, False, True], [False, False, True]),
            "xstest_safe": group([False, False, False], [False, True, False]),
        }
    }
    result = comparison_metrics(candidate, base, np.array([0.0, 0.1, 0.2]))
    standard = result["harmful"]["standard_harmful"]["arditi_anywhere_refusal"]
    matched = result["harmful"]["matched_harmful"]["arditi_anywhere_refusal"]
    assert standard["removed_fraction_of_base_refusals"] == 0.5
    assert standard["added_refusal_count"] == 1
    assert matched["removed_fraction_of_base_refusals"] == 2 / 3
    assert result["xstest_safe"]["added_opening_refusal_rate"] == 1 / 3
    assert result["harmless_kl"]["mean"] == pytest.approx(0.1)


def test_resume_requires_all_response_groups(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    response = raw / "standard_harmful.jsonl"
    response.write_text("{}\n", encoding="utf-8")
    record = {
        "candidate_sha256": "candidate",
        "prompt_source_sha256": {"source": "hash"},
        "run_signature": {"run": 1},
        "response_sha256": {
            "standard_harmful": hashlib.sha256(response.read_bytes()).hexdigest()
        },
    }
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert (
        valid_completed_arm(
            record_path,
            raw,
            "candidate",
            {"source": "hash"},
            {"run": 1},
        )
        is None
    )


def test_move_incomplete_uses_unique_names(tmp_path) -> None:
    arm = tmp_path / "base"
    arm.mkdir()
    (tmp_path / "base.incomplete-1").mkdir()
    move_incomplete(arm)
    assert not arm.exists()
    assert (tmp_path / "base.incomplete-2").is_dir()
