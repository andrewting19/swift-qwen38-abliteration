from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "screen_iterative_directions.py"
SPEC = importlib.util.spec_from_file_location("screen_iterative_directions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_orthogonalize_removes_existing_basis() -> None:
    result, residual_norm = MODULE.orthogonalize(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
    )
    np.testing.assert_allclose(result[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)
    assert residual_norm > 0
