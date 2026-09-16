from scripts.screen_rank2_layer_bands import retained_fraction


def test_retained_fraction_uses_full_effect_as_denominator() -> None:
    assert retained_fraction(4.0, 3.0) == 0.75


def test_retained_fraction_rejects_zero_denominator() -> None:
    assert retained_fraction(0.0, 3.0) is None
