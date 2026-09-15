from swift_abliteration.response_quality import (
    analyze_response,
    maximum_nonoverlap_ngram_coverage,
    unique_ngram_ratio,
)


def test_empty_and_short_output_flags() -> None:
    assert analyze_response("   ")["empty_output"] is True
    assert analyze_response("Yes.")["very_short_output"] is True
    assert analyze_response("This is a longer response.")["very_short_output"] is False


def test_repeated_span_is_detected() -> None:
    tokens = ("repeat this exact short phrase ".split()) * 10
    metrics = analyze_response(" ".join(tokens))
    assert maximum_nonoverlap_ngram_coverage(tokens) >= 0.50
    assert metrics["severe_repetition"] is True


def test_normal_sequence_is_not_severe_repetition() -> None:
    text = " ".join(f"word{index}" for index in range(60))
    metrics = analyze_response(text)
    assert unique_ngram_ratio(text.split()) == 1.0
    assert metrics["severe_repetition"] is False
