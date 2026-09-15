import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swift_abliteration.direction_study import (
    bootstrap_consensus_stability,
    bootstrap_cosine_stability,
    coordinate_masked_direction,
    cosine_similarity,
    normalized_average,
    orthogonalize_direction,
    ridge_fisher_direction,
    standardized_separation,
    winsorized_direction,
)
from swift_abliteration.math_core import refusal_direction


class DirectionStudyTests(unittest.TestCase):
    def test_orthogonalize_direction_removes_existing_basis(self) -> None:
        candidate = np.array([1.0, 2.0, 2.0], dtype=np.float32)
        basis = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]

        direction, residual_norm = orthogonalize_direction(candidate, basis)

        self.assertAlmostEqual(residual_norm, np.sqrt(8.0))
        self.assertAlmostEqual(float(direction @ basis[0]), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=6)

    def test_ridge_fisher_matches_dense_solve(self) -> None:
        harmful = np.array(
            [[2.0, 1.0, 0.0], [3.0, -1.0, 1.0], [2.5, 0.5, -1.0]],
            dtype=np.float32,
        )
        harmless = np.array(
            [[0.0, 1.0, 0.0], [0.5, -1.0, 1.0], [-0.5, 0.5, -1.0]],
            dtype=np.float32,
        )
        shrinkage = 0.2
        regularization = 1e-4
        observed, details = ridge_fisher_direction(
            harmful, harmless, shrinkage, regularization
        )
        h_centered = harmful - harmful.mean(0)
        b_centered = harmless - harmless.mean(0)
        covariance = (
            h_centered.T @ h_centered / (2 * (len(harmful) - 1))
            + b_centered.T @ b_centered / (2 * (len(harmless) - 1))
        )
        scale = np.trace(covariance) / covariance.shape[0]
        dense = np.linalg.solve(
            (1 - shrinkage) * covariance
            + scale * (shrinkage + regularization) * np.eye(3),
            harmful.mean(0) - harmless.mean(0),
        )
        dense /= np.linalg.norm(dense)
        np.testing.assert_allclose(observed, dense, atol=1e-5)
        self.assertGreater(details["variance_scale"], 0)

    def test_normalized_average_bisects_unit_directions(self) -> None:
        result = normalized_average(
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        )
        np.testing.assert_allclose(result, np.array([2**-0.5, 2**-0.5]), atol=1e-6)

    def test_bootstrap_consensus_is_stable_for_clean_sources(self) -> None:
        rng = np.random.default_rng(8)
        sources = []
        for scale in (1.0, 1.2):
            harmless = rng.normal(0, 0.01, size=(32, 4)).astype(np.float32)
            harmful = harmless + np.array([scale, 0.0, 0.0, 0.0], dtype=np.float32)
            sources.append((harmful, harmless))
        reference = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        values = bootstrap_consensus_stability(sources, reference, samples=20)
        self.assertGreater(float(np.median(values)), 0.999)

    def test_winsorization_uses_one_shared_threshold(self):
        harmful = np.array([[1000, 2], [2, 2]], dtype=np.float32)
        harmless = np.array([[0, 0], [0, 0]], dtype=np.float32)
        direction, threshold, changed = winsorized_direction(harmful, harmless, 0.75)
        self.assertLess(threshold, 1000)
        self.assertGreater(changed, 0)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=6)

    def test_explicit_coordinate_mask(self):
        harmful = np.array([[1000, 2], [1000, 2]], dtype=np.float32)
        harmless = np.zeros((2, 2), dtype=np.float32)
        direction = coordinate_masked_direction(
            harmful, harmless, np.array([True, False])
        )
        np.testing.assert_allclose(direction, [0, 1], atol=1e-6)

    def test_bootstrap_is_stable_for_clean_groups(self):
        harmful = np.array([[2, -0.1], [2, 0.0], [2, 0.1], [2, 0.05]], dtype=np.float32)
        harmless = np.array(
            [[0, -0.1], [0, 0.0], [0, 0.1], [0, 0.05]], dtype=np.float32
        )
        reference = refusal_direction(harmful, harmless)
        scores = bootstrap_cosine_stability(harmful, harmless, reference, samples=30)
        self.assertGreater(float(scores.min()), 0.99)

    def test_separation_and_cosine(self):
        harmful = np.array([[2, -1], [4, 1]], dtype=np.float32)
        harmless = np.array([[0, -1], [0, 1]], dtype=np.float32)
        direction = np.array([1, 0], dtype=np.float32)
        self.assertGreater(standardized_separation(harmful, harmless, direction), 1)
        self.assertAlmostEqual(cosine_similarity(direction, direction), 1.0, places=6)
