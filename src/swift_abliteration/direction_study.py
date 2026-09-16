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


def ridge_fisher_direction(
    harmful: np.ndarray,
    harmless: np.ndarray,
    shrinkage: float,
    regularization_ratio: float = 1e-4,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a low-rank ridge Fisher direction without a dense covariance."""
    harmful = np.asarray(harmful, dtype=np.float64)
    harmless = np.asarray(harmless, dtype=np.float64)
    if harmful.ndim != 2 or harmless.ndim != 2 or harmful.shape[1] != harmless.shape[1]:
        raise ValueError("Activation groups must have matching matrix shapes.")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("Shrinkage must be between 0 and 1.")
    if regularization_ratio <= 0.0:
        raise ValueError("Regularization ratio must be positive.")
    delta = harmful.mean(0) - harmless.mean(0)
    harmful_centered = harmful - harmful.mean(0, keepdims=True)
    harmless_centered = harmless - harmless.mean(0, keepdims=True)
    rows = np.concatenate(
        [
            harmful_centered / np.sqrt(2.0 * max(len(harmful) - 1, 1)),
            harmless_centered / np.sqrt(2.0 * max(len(harmless) - 1, 1)),
        ],
        axis=0,
    )
    variance_scale = float(np.sum(rows * rows) / rows.shape[1])
    if not np.isfinite(variance_scale) or variance_scale <= 0.0:
        raise ValueError("Activation covariance has zero or invalid scale.")
    low_rank = np.sqrt(1.0 - shrinkage) * rows
    ridge = variance_scale * (shrinkage + regularization_ratio)
    gram = low_rank @ low_rank.T
    solved = np.linalg.solve(
        gram + ridge * np.eye(len(low_rank), dtype=np.float64),
        low_rank @ delta,
    )
    value = (delta - low_rank.T @ solved) / ridge
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("The Fisher direction has zero or invalid length.")
    direction = (value / norm).astype(np.float32)
    return direction, {
        "shrinkage": float(shrinkage),
        "regularization_ratio": float(regularization_ratio),
        "variance_scale": variance_scale,
        "mean_difference_norm": float(np.linalg.norm(delta)),
        "projected_within_class_variance": float(
            np.sum((rows @ direction.astype(np.float64)) ** 2)
        ),
    }


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


def massive_activation_coordinate_mask(
    activation_groups: list[np.ndarray],
    log_robust_z_threshold: float = 10.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Detect persistent coordinate outliers without using behavior labels."""
    if not activation_groups:
        raise ValueError("At least one activation group is required.")
    arrays = [np.asarray(group, dtype=np.float32) for group in activation_groups]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("Activation groups must be matrices.")
    hidden_size = arrays[0].shape[1]
    if any(array.shape[1] != hidden_size for array in arrays[1:]):
        raise ValueError("Activation groups must have the same hidden size.")
    if not np.isfinite(log_robust_z_threshold) or log_robust_z_threshold <= 0:
        raise ValueError("The robust-z threshold must be positive and finite.")
    pooled = np.concatenate(arrays, axis=0).astype(np.float64, copy=False)
    rms = np.sqrt(np.mean(np.square(pooled), axis=0))
    if not np.all(np.isfinite(rms)) or np.any(rms <= 0):
        raise ValueError("Coordinate RMS values must be positive and finite.")
    log_rms = np.log(rms)
    median = float(np.median(log_rms))
    mad = float(np.median(np.abs(log_rms - median)))
    if not np.isfinite(mad) or mad <= 0:
        raise ValueError("Log-RMS median absolute deviation must be positive.")
    robust_z = (log_rms - median) / (1.4826 * mad)
    mask = robust_z > log_robust_z_threshold
    selected = np.flatnonzero(mask)
    return mask, {
        "method": "pooled_coordinate_log_rms_robust_z",
        "log_robust_z_threshold": float(log_robust_z_threshold),
        "selected_indices": [int(index) for index in selected],
        "selected_count": int(selected.size),
        "maximum_log_robust_z": float(robust_z.max()),
        "second_largest_log_robust_z": float(np.partition(robust_z, -2)[-2]),
        "maximum_rms_to_median_rms": float(rms.max() / np.median(rms)),
    }


def bootstrap_masked_direction_stability(
    harmful: np.ndarray,
    harmless: np.ndarray,
    coordinate_mask: np.ndarray,
    reference: np.ndarray,
    samples: int = 500,
    seed: int = 3819,
) -> np.ndarray:
    """Bootstrap a fixed coordinate-masked mean-difference estimator."""
    harmful = np.asarray(harmful, dtype=np.float32)
    harmless = np.asarray(harmless, dtype=np.float32)
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    generator = np.random.default_rng(seed)
    results = np.empty(samples, dtype=np.float32)
    for index in range(samples):
        h_indices = generator.integers(0, len(harmful), size=len(harmful))
        s_indices = generator.integers(0, len(harmless), size=len(harmless))
        candidate = coordinate_masked_direction(
            harmful[h_indices], harmless[s_indices], coordinate_mask
        )
        results[index] = cosine_similarity(candidate, reference)
    return results


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


def orthogonalize_direction(
    candidate: np.ndarray, basis: list[np.ndarray]
) -> tuple[np.ndarray, float]:
    """Remove an orthonormal basis from one candidate and normalize it."""
    value = np.asarray(candidate, dtype=np.float32).copy()
    if value.ndim != 1:
        raise ValueError("The candidate direction must be a vector.")
    for direction in basis:
        unit = np.asarray(direction, dtype=np.float32)
        if unit.shape != value.shape:
            raise ValueError("All basis directions must match the candidate shape.")
        norm = float(np.linalg.norm(unit))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("Basis directions must be finite and nonzero.")
        unit = unit / norm
        value -= float(value @ unit) * unit
    residual_norm = float(np.linalg.norm(value))
    if not np.isfinite(residual_norm) or residual_norm <= 1e-6:
        raise ValueError("The candidate is contained in the existing basis.")
    return (value / residual_norm).astype(np.float32), residual_norm


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
