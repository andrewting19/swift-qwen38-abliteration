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
