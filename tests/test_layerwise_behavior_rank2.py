import numpy as np

from scripts.capture_layerwise_behavior_rank2 import (
    build_layer_basis,
    persistent_coordinate_mask,
    split_half_cosine,
)


def test_build_layer_basis_returns_orthonormal_rank_two_rows() -> None:
    harmful = [
        np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32),
    ]
    harmless = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    ]
    masks = [np.zeros(3, dtype=bool), np.zeros(3, dtype=bool)]
    basis, directions = build_layer_basis(harmful, harmless, masks)
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(directions[0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(directions[1], np.array([0.0, 1.0, 0.0]))


def test_split_half_cosine_is_one_for_stable_direction() -> None:
    harmful = np.array(
        [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        dtype=np.float32,
    )
    harmless = np.zeros((4, 2), dtype=np.float32)
    assert split_half_cosine(harmful, harmless, np.zeros(2, dtype=bool)) == 1.0


def test_persistent_coordinate_mask_requires_requested_frequency() -> None:
    mask, required = persistent_coordinate_mask(
        [
            np.array([True, True, False]),
            np.array([True, False, False]),
            np.array([True, True, True]),
        ],
        2 / 3,
    )
    assert required == 2
    np.testing.assert_array_equal(mask, np.array([True, True, False]))
