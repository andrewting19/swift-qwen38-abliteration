from __future__ import annotations

import numpy as np

REFUSAL_LABELS = {"hard_refusal", "soft_refusal"}


def log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = values.max(axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def forward_kl_from_logits(
    base_logits: np.ndarray, edited_logits: np.ndarray
) -> np.ndarray:
    base = np.asarray(base_logits)
    edited = np.asarray(edited_logits)
    if base.shape != edited.shape or base.ndim != 2:
        raise ValueError("Logits must have matching [prompts, vocabulary] shapes.")
    log_p = log_softmax(base)
    log_q = log_softmax(edited)
    p = np.exp(log_p)
    return np.sum(p * (log_p - log_q), axis=-1)


def summarize(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("Summary input must be a nonempty finite vector.")
    return {
        "count": len(data),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "maximum": float(data.max()),
    }


def refusal_rate(judgments: list[dict]) -> float:
    if not judgments:
        raise ValueError("At least one refusal judgment is required.")
    labels = [item["judgment"]["label"] for item in judgments]
    return sum(label in REFUSAL_LABELS for label in labels) / len(labels)


def coherence_means(judgments: list[dict], dimensions: list[str]) -> dict[str, float]:
    if not judgments:
        raise ValueError("At least one coherence judgment is required.")
    return {
        dimension: float(np.mean([item["judgment"][dimension] for item in judgments]))
        for dimension in dimensions
    }
