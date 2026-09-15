from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def response_tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


def unique_ngram_ratio(tokens: list[str], size: int = 4) -> float:
    if size <= 0:
        raise ValueError("N-gram size must be positive.")
    if len(tokens) < size:
        return 1.0
    ngrams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    return len(set(ngrams)) / len(ngrams)


def maximum_nonoverlap_ngram_coverage(tokens: list[str]) -> float:
    """Return the largest conservative coverage from a repeated 4-16 token span."""
    if len(tokens) < 16:
        return 0.0
    best = 0.0
    for size in range(4, min(16, len(tokens) // 2) + 1):
        positions: dict[tuple[str, ...], list[int]] = {}
        for index in range(len(tokens) - size + 1):
            ngram = tuple(tokens[index : index + size])
            positions.setdefault(ngram, []).append(index)
        for indices in positions.values():
            selected = 0
            next_allowed = -1
            for index in indices:
                if index >= next_allowed:
                    selected += 1
                    next_allowed = index + size
            if selected >= 3:
                best = max(best, selected * size / len(tokens))
    return min(best, 1.0)


def analyze_response(text: str) -> dict[str, int | float | bool]:
    tokens = response_tokens(text)
    stripped = text.strip()
    fourgram_ratio = unique_ngram_ratio(tokens, size=4)
    repeat_coverage = maximum_nonoverlap_ngram_coverage(tokens)
    severe_repetition = len(tokens) >= 32 and (
        fourgram_ratio < 0.20 or repeat_coverage >= 0.50
    )
    return {
        "character_count": len(stripped),
        "token_count": len(tokens),
        "empty_output": not stripped,
        "very_short_output": bool(stripped) and len(tokens) <= 2,
        "unique_fourgram_ratio": fourgram_ratio,
        "maximum_repeated_ngram_coverage": repeat_coverage,
        "severe_repetition": severe_repetition,
    }


def summarize_quality(items: list[dict[str, int | float | bool]]) -> dict[str, int | float]:
    if not items:
        raise ValueError("At least one response is required.")
    token_counts = [int(item["token_count"]) for item in items]
    ratios = [float(item["unique_fourgram_ratio"]) for item in items]
    coverages = [float(item["maximum_repeated_ngram_coverage"]) for item in items]
    return {
        "count": len(items),
        "empty_output_count": sum(bool(item["empty_output"]) for item in items),
        "very_short_output_count": sum(bool(item["very_short_output"]) for item in items),
        "severe_repetition_count": sum(bool(item["severe_repetition"]) for item in items),
        "mean_token_count": sum(token_counts) / len(token_counts),
        "minimum_token_count": min(token_counts),
        "mean_unique_fourgram_ratio": sum(ratios) / len(ratios),
        "minimum_unique_fourgram_ratio": min(ratios),
        "maximum_repeated_ngram_coverage": max(coverages),
    }
