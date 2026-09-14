from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swift_abliteration.direction_study import (
    bootstrap_cosine_stability,
    coordinate_masked_direction,
    cosine_similarity,
    standardized_separation,
    winsorized_direction,
)
from swift_abliteration.math_core import refusal_direction


class DirectionStudyTests(unittest.TestCase):
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
        direction = coordinate_masked_direction(harmful, harmless, np.array([True, False]))
        np.testing.assert_allclose(direction, [0, 1], atol=1e-6)

    def test_bootstrap_is_stable_for_clean_groups(self):
        harmful = np.array([[2, -0.1], [2, 0.0], [2, 0.1], [2, 0.05]], dtype=np.float32)
        harmless = np.array([[0, -0.1], [0, 0.0], [0, 0.1], [0, 0.05]], dtype=np.float32)
        reference = refusal_direction(harmful, harmless)
        scores = bootstrap_cosine_stability(harmful, harmless, reference, samples=30)
        self.assertGreater(float(scores.min()), 0.99)

    def test_separation_and_cosine(self):
        harmful = np.array([[2, -1], [4, 1]], dtype=np.float32)
        harmless = np.array([[0, -1], [0, 1]], dtype=np.float32)
        direction = np.array([1, 0], dtype=np.float32)
        self.assertGreater(standardized_separation(harmful, harmless, direction), 1)
        self.assertAlmostEqual(cosine_similarity(direction, direction), 1.0, places=6)
