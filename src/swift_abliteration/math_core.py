from __future__ import annotations

import numpy as np


def refusal_direction(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    harmful = np.asarray(harmful, dtype=np.float32)
    harmless = np.asarray(harmless, dtype=np.float32)
    if harmful.ndim != 2 or harmless.ndim != 2:
        raise ValueError("Both activation groups must be two-dimensional.")
    if harmful.shape[1] != harmless.shape[1]:
        raise ValueError("Activation widths do not match.")
    raw = harmful.mean(axis=0) - harmless.mean(axis=0)
    norm = np.linalg.norm(raw)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("The measured direction has zero or invalid length.")
    return raw / norm


def project_output_weight(
    weight: np.ndarray, direction: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Project a hidden-size output axis: W - alpha*r*(r^T W)."""
    w = np.asarray(weight, dtype=np.float32)
    r = _unit_direction(direction)
    if w.ndim != 2 or w.shape[0] != r.shape[0]:
        raise ValueError("Weight output size must match direction size.")
    return w - np.float32(alpha) * np.outer(r, r @ w)


def project_embedding_rows(
    embedding: np.ndarray, direction: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Project a hidden-size embedding axis: E - alpha*(E r)*r^T."""
    e = np.asarray(embedding, dtype=np.float32)
    r = _unit_direction(direction)
    if e.ndim != 2 or e.shape[1] != r.shape[0]:
        raise ValueError("Embedding width must match direction size.")
    return e - np.float32(alpha) * np.outer(e @ r, r)


def _unit_direction(direction: np.ndarray) -> np.ndarray:
    r = np.asarray(direction, dtype=np.float32)
    if r.ndim != 1:
        raise ValueError("Direction must be one-dimensional.")
    norm = np.linalg.norm(r)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Direction has zero or invalid length.")
    return r / norm
