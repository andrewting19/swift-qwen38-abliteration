import pytest

from scripts.screen_alpha_ladder import incremental_alpha


def test_incremental_alpha_composes_projection_strength() -> None:
    current = 0.60
    target = 0.85
    step = incremental_alpha(current, target)
    effective = current + step - current * step
    assert effective == pytest.approx(target)


def test_incremental_alpha_rejects_nonincreasing_target() -> None:
    with pytest.raises(ValueError):
        incremental_alpha(0.6, 0.6)
