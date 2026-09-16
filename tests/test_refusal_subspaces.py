from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_refusal_subspaces.py"
SPEC = importlib.util.spec_from_file_location("build_refusal_subspaces", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_orthonormal_rows_returns_requested_rank() -> None:
    values = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    basis = MODULE.orthonormal_rows(values, 2)
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-6)
    assert basis.flags.c_contiguous


def test_matched_svd_basis_contains_mean_direction() -> None:
    harmless = np.zeros((8, 6), dtype=np.float32)
    harmful = np.array(
        [[2.0, float(index % 2), 0.0, 0.0, 0.0, 0.0] for index in range(8)],
        dtype=np.float32,
    )
    basis, details = MODULE.matched_svd_basis(harmful, harmless, rank=2)
    mean = (harmful - harmless).mean(axis=0)
    residual = mean - (mean @ basis.T) @ basis
    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-5)
    assert details["rank"] == 2
