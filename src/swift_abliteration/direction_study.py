from __future__ import annotations

import numpy as np

from .math_core import refusal_direction


def winsorized_direction(
    harmful: np.ndarray,
    harmless: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, float, float]:
    """Return direction, shared threshold, and changed-value fraction."""
    harmful = np.asarray(harmful, dtype=np.float32)
    harmless = np.asarray(harmless, dtype=np.float32)
    if harmful.ndim != 2 or harmless.ndim != 2 or harmful.shape[1] != harmless.shape[1]:
        raise ValueError(
            "Activation groups must have matching [prompts, hidden] shapes."
        )
    if not 0.0 < quantile < 1.0:
        raise ValueError("Quantile must be between 0 and 1.")
    pooled = np.concatenate([harmful, harmless], axis=0)
    threshold = float(np.quantile(np.abs(pooled), quantile))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Winsorization threshold is zero or invalid.")
    changed = float(np.count_nonzero(np.abs(pooled) > threshold) / pooled.size)
    harmful_clipped = np.clip(harmful, -threshold, threshold)
    harmless_clipped = np.clip(harmless, -threshold, threshold)
    return refusal_direction(harmful_clipped, harmless_clipped), threshold, changed


def coordinate_masked_direction(
    harmful: np.ndarray,
    harmless: np.ndarray,
    coordinate_mask: np.ndarray,
) -> np.ndarray:
    """Apply a caller-defined mask. True means exclude this coordinate."""
    base = np.asarray(harmful, dtype=np.float32).mean(0) - np.asarray(
        harmless, dtype=np.float32
    ).mean(0)
    mask = np.asarray(coordinate_mask, dtype=bool)
    if base.ndim != 1 or mask.shape != base.shape:
        raise ValueError("Coordinate mask must match the hidden dimension.")
    base[mask] = 0.0
    norm = np.linalg.norm(base)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("The mask removed the complete direction.")
    return base / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 0:
        raise ValueError("Cosine similarity requires nonzero vectors.")
    return float((left @ right) / denominator)


def normalized_average(directions: list[np.ndarray]) -> np.ndarray:
    """Return the unit-normalized arithmetic mean of unit directions."""
    if not directions:
        raise ValueError("At least one direction is required.")
    values = [np.asarray(direction, dtype=np.float32) for direction in directions]
    if any(value.ndim != 1 for value in values):
        raise ValueError("Directions must be vectors.")
    if any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("Directions must have matching shapes.")
    unit_values = []
    for value in values:
        norm = np.linalg.norm(value)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Directions must be finite and nonzero.")
        unit_values.append(value / norm)
    mean = np.mean(unit_values, axis=0, dtype=np.float32)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("The direction average is zero or invalid.")
    return mean / norm


def bootstrap_consensus_stability(
    sources: list[tuple[np.ndarray, np.ndarray]],
    reference: np.ndarray,
    samples: int = 500,
    seed: int = 3819,
    winsor_quantile: float | None = None,
) -> np.ndarray:
    """Bootstrap each source independently, then rebuild the consensus."""
    if not sources:
        raise ValueError("At least one activation source is required.")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    arrays = []
    hidden_size = None
    for harmful, harmless in sources:
        harmful = np.asarray(harmful, dtype=np.float32)
        harmless = np.asarray(harmless, dtype=np.float32)
        if harmful.ndim != 2 or harmless.ndim != 2:
            raise ValueError("Activation groups must be matrices.")
        if harmful.shape[1] != harmless.shape[1]:
            raise ValueError("Harmful and harmless hidden sizes must match.")
        if hidden_size is None:
            hidden_size = harmful.shape[1]
        elif harmful.shape[1] != hidden_size:
            raise ValueError("All source hidden sizes must match.")
        arrays.append((harmful, harmless))
    generator = np.random.default_rng(seed)
    results = np.empty(samples, dtype=np.float32)
    for index in range(samples):
        directions = []
        for harmful, harmless in arrays:
            h_indices = generator.integers(0, len(harmful), size=len(harmful))
            s_indices = generator.integers(0, len(harmless), size=len(harmless))
            if winsor_quantile is None:
                candidate = refusal_direction(
                    harmful[h_indices], harmless[s_indices]
                )
            else:
                candidate, _, _ = winsorized_direction(
                    harmful[h_indices], harmless[s_indices], winsor_quantile
                )
            directions.append(candidate)
        consensus = normalized_average(directions)
        results[index] = cosine_similarity(consensus, reference)
    return results


def bootstrap_cosine_stability(
    harmful: np.ndarray,
    harmless: np.ndarray,
    reference: np.ndarray,
    samples: int = 500,
    seed: int = 3819,
) -> np.ndarray:
    harmful = np.asarray(harmful, dtype=np.float32)
    harmless = np.asarray(harmless, dtype=np.float32)
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    generator = np.random.default_rng(seed)
    results = np.empty(samples, dtype=np.float32)
    for index in range(samples):
        h_indices = generator.integers(0, len(harmful), size=len(harmful))
        s_indices = generator.integers(0, len(harmless), size=len(harmless))
        candidate = refusal_direction(harmful[h_indices], harmless[s_indices])
        results[index] = cosine_similarity(candidate, reference)
    return results


def standardized_separation(
    harmful: np.ndarray,
    harmless: np.ndarray,
    direction: np.ndarray,
) -> float:
    harmful_scores = np.asarray(harmful, dtype=np.float32) @ direction
    harmless_scores = np.asarray(harmless, dtype=np.float32) @ direction
    pooled_variance = (harmful_scores.var(ddof=1) + harmless_scores.var(ddof=1)) / 2
    if pooled_variance <= 0:
        raise ValueError("Projected activation scores have no within-group variance.")
    return float(
        (harmful_scores.mean() - harmless_scores.mean()) / np.sqrt(pooled_variance)
    )
