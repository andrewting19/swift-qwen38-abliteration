import pytest

from swift_abliteration.response_mode_judge import majority, parse_mode_quality


def test_parse_mode_quality_accepts_fixed_format() -> None:
    assert parse_mode_quality("MODE=SOFT_REFUSAL; QUALITY=MINOR_ERROR") == (
        "SOFT_REFUSAL",
        "MINOR_ERROR",
    )


def test_parse_mode_quality_rejects_missing_field() -> None:
    with pytest.raises(ValueError):
        parse_mode_quality("MODE=DIRECT")


def test_majority_requires_strict_majority() -> None:
    assert majority(["DIRECT", "DIRECT", "PARTIAL"]) == ("DIRECT", 2)
    assert majority(["DIRECT", "PARTIAL", "SOFT_REFUSAL"]) == (None, 1)
    assert majority([]) == (None, 0)
