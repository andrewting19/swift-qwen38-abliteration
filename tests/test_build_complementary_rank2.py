import numpy as np

from scripts.build_complementary_rank2 import orthonormal_pair, pair_effect


def arm(standard, matched):
    return {
        "groups": {
            "standard_harmful": {
                "arditi_anywhere_refusal_labels": standard,
                "opening_refusal_labels": standard,
            },
            "matched_harmful": {
                "arditi_anywhere_refusal_labels": matched,
                "opening_refusal_labels": matched,
            },
        }
    }


def test_orthonormal_pair_returns_two_perpendicular_rows() -> None:
    basis = orthonormal_pair(np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]))
    assert basis.flags.c_contiguous
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-6)


def test_pair_effect_rewards_complementary_prompt_changes() -> None:
    base = arm([True, True, True, True], [True, True, True, True])
    first = arm([False, True, True, True], [False, True, True, True])
    second = arm([True, False, True, True], [True, False, True, True])
    result = pair_effect(first, second, base)
    assert result["minimum_potential_union_removal"] == 0.5
    assert result["minimum_complementary_gain"] == 0.25
