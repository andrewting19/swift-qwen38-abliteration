from scripts.screen_layerwise_writer_sensitivity import (
    cumulative_counts,
    restoration_priority,
    retained_fraction,
)


def test_retained_fraction_uses_full_effect() -> None:
    assert retained_fraction(4.0, 3.0) == 0.75


def test_restoration_priority_rewards_kl_with_low_effect_loss() -> None:
    assert restoration_priority(0.1, 0.95) > restoration_priority(0.1, 0.5)


def test_restoration_priority_rejects_negative_kl_recovery() -> None:
    assert restoration_priority(-0.1, 1.0) == -0.1


def test_cumulative_counts_stays_within_writer_count() -> None:
    assert cumulative_counts(10) == [1, 2, 4, 8]
