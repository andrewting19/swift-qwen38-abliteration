import numpy as np

from scripts.build_writer_aligned_layerwise_candidate import (
    align_resid_pre_to_writers,
)


def test_align_resid_pre_to_writers_shifts_by_one_layer() -> None:
    source = np.arange(4 * 2 * 3).reshape(4, 2, 3)
    observed = align_resid_pre_to_writers(source)
    np.testing.assert_array_equal(observed[0], source[1])
    np.testing.assert_array_equal(observed[1], source[2])
    np.testing.assert_array_equal(observed[2], source[3])
    np.testing.assert_array_equal(observed[3], source[3])
