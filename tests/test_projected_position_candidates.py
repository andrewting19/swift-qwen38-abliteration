import numpy as np

from scripts.build_projected_position_candidates import (
    orthonormal_rows,
    project_away_from_mean,
)


def test_projected_direction_is_unit_and_orthogonal_to_mean() -> None:
    direction = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    mean = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    projected = project_away_from_mean(direction, mean)
    np.testing.assert_allclose(np.linalg.norm(projected), 1.0, atol=1e-6)
    np.testing.assert_allclose(projected @ mean, 0.0, atol=1e-6)


def test_orthonormal_rows_returns_rank_two_basis() -> None:
    basis = orthonormal_rows(
        [
            np.array([1.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        ]
    )
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-6)
