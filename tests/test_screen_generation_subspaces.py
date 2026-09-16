import hashlib
import json

import numpy as np
import pytest

from scripts.screen_generation_subspaces import (
    comparison_metrics,
    copy_reused_base,
    flags_for_response,
    move_incomplete,
    valid_completed_arm,
    validate_basis,
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


def test_validate_basis_accepts_layerwise_rank_two_tensor() -> None:
    import torch

    values = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    assert validate_basis(values) == 2


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
    assert result["harmless_kl"]["values"] == pytest.approx([0.0, 0.1, 0.2])
    assert result["harmless_kl_base_answered"]["mean"] == pytest.approx(0.1)
    assert result["harmless_kl_base_answered"]["excluded_base_refusal_count"] == 0
    assert result["harmless_kl_base_answered"]["included_indices"] == [0, 1, 2]


def test_clean_kl_excludes_safe_prompts_refused_by_base() -> None:
    base = {
        "groups": {
            "standard_harmful": group([True], [True]),
            "matched_harmful": group([True], [True]),
            "xstest_safe": group([True, False, False], [False, False, True]),
        }
    }
    candidate = {
        "groups": {
            "standard_harmful": group([False], [False]),
            "matched_harmful": group([False], [False]),
            "xstest_safe": group([False, False, False], [False, False, False]),
        }
    }
    result = comparison_metrics(candidate, base, np.array([2.0, 0.05, 1.0]))
    clean = result["harmless_kl_base_answered"]
    assert result["harmless_kl"]["mean"] == pytest.approx(1.0166666667)
    assert clean["mean"] == pytest.approx(0.05)
    assert clean["values"] == pytest.approx([0.05])
    assert clean["excluded_base_refusal_count"] == 2
    assert clean["included_indices"] == [1]


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


def test_resume_accepts_required_signature_subset(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    response_hashes = {}
    for group_name in ("standard_harmful", "matched_harmful", "xstest_safe"):
        path = raw / f"{group_name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        response_hashes[group_name] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "candidate_sha256": None,
        "prompt_source_sha256": {"source": "hash"},
        "run_signature": {"generation": 1, "intervention": "old"},
        "response_sha256": response_hashes,
    }
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert valid_completed_arm(
        record_path,
        raw,
        None,
        {"source": "hash"},
        {"generation": 1},
    ) == record


def test_copy_reused_base_checks_and_copies_all_artifacts(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "arms").mkdir(parents=True)
    (source / "raw/base").mkdir(parents=True)
    (destination / "arms").mkdir(parents=True)
    (destination / "raw").mkdir(parents=True)
    response_hashes = {}
    for group_name in ("standard_harmful", "matched_harmful", "xstest_safe"):
        path = source / "raw/base" / f"{group_name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        response_hashes[group_name] = hashlib.sha256(path.read_bytes()).hexdigest()
    logits = source / "arms/base_xstest_first_logits.npz"
    np.savez(logits, logits=np.zeros((1, 2)))
    record = {
        "candidate_sha256": None,
        "prompt_source_sha256": {"source": "hash"},
        "run_signature": {"generation": 1, "intervention": "old"},
        "response_sha256": response_hashes,
        "xstest_logits_sha256": hashlib.sha256(logits.read_bytes()).hexdigest(),
    }
    (source / "arms/base.json").write_text(json.dumps(record), encoding="utf-8")
    copied = copy_reused_base(
        source,
        destination,
        {"source": "hash"},
        {"generation": 1},
    )
    assert copied == record
    assert (destination / "arms/base.json").is_file()
    assert (destination / "arms/base_xstest_first_logits.npz").is_file()
    assert len(list((destination / "raw/base").glob("*.jsonl"))) == 3


def test_move_incomplete_uses_unique_names(tmp_path) -> None:
    arm = tmp_path / "base"
    arm.mkdir()
    (tmp_path / "base.incomplete-1").mkdir()
    move_incomplete(arm)
    assert not arm.exists()
    assert (tmp_path / "base.incomplete-2").is_dir()
