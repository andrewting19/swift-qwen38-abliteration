from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "screen_layerwise.py"
SPEC = importlib.util.spec_from_file_location("screen_layerwise", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_nearest_anchor_uses_nearest_and_lower_tie() -> None:
    anchors = [24, 32, 38, 44, 52]
    assert MODULE.nearest_anchor(36, anchors) == 38
    assert MODULE.nearest_anchor(41, anchors) == 38
    assert MODULE.nearest_anchor(55, anchors) == 52


def test_empty_plan_filter_selects_all_plans() -> None:
    plans = [{"name": "one"}, {"name": "two"}]
    assert MODULE.select_plans(plans, set()) == plans


def test_plan_target_range_is_inclusive() -> None:
    assert MODULE.plan_target_layers({"target_first": 3, "target_last": 5}) == [
        3,
        4,
        5,
    ]
