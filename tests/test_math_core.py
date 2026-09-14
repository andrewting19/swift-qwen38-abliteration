import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swift_abliteration.math_core import (
    project_embedding_rows,
    project_output_weight,
    refusal_direction,
)


class MathTests(unittest.TestCase):
    def test_difference_of_means_is_normalized(self):
        harmful = np.array([[3, 1, 0], [5, 1, 0]], dtype=np.float32)
        harmless = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.float32)
        r = refusal_direction(harmful, harmless)
        np.testing.assert_allclose(r, [1, 0, 0], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(r)), 1.0, places=6)

    def test_full_output_projection_removes_direction(self):
        weight = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        r = np.array([1, 0, 0], dtype=np.float32)
        edited = project_output_weight(weight, r, alpha=1.0)
        np.testing.assert_allclose(r @ edited, [0, 0], atol=1e-6)
        np.testing.assert_allclose(edited[1:], weight[1:])

    def test_full_embedding_projection_removes_direction(self):
        embedding = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        r = np.array([0, 1, 0], dtype=np.float32)
        edited = project_embedding_rows(embedding, r, alpha=1.0)
        np.testing.assert_allclose(edited @ r, [0, 0], atol=1e-6)
        np.testing.assert_allclose(edited[:, [0, 2]], embedding[:, [0, 2]])

    def test_partial_projection_leaves_expected_fraction(self):
        weight = np.eye(3, dtype=np.float32)
        r = np.array([1, 0, 0], dtype=np.float32)
        edited = project_output_weight(weight, r, alpha=0.25)
        np.testing.assert_allclose(r @ edited, [0.75, 0, 0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
