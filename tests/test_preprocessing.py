import numpy as np
import pytest
from scipy import sparse
from upm.preprocessing import prepare_controls


@pytest.mark.parametrize("as_sparse", [False, True])
def test_counts_normalize_per_cell_including_zero_cells(as_sparse):
    counts = np.array([[1., 3.], [0., 0.]])
    x = sparse.csr_matrix(counts) if as_sparse else counts
    expected = np.log1p([[2500., 7500.], [0., 0.]])
    np.testing.assert_allclose(prepare_controls(x, "counts"), expected, atol=1e-6)


def test_log1p_is_not_normalized_twice(controls):
    np.testing.assert_array_equal(prepare_controls(controls, "log1p-cp10k"), controls)


def test_counts_cannot_overflow_float32():
    with pytest.raises(ValueError):
        prepare_controls(np.array([[1e100, 1.]]), "counts")


@pytest.mark.parametrize("x", [[[1., -1.]], [[np.nan, 1.]], [[np.inf, 1.]]])
def test_invalid_expression_rejected(x):
    with pytest.raises(ValueError):
        prepare_controls(np.asarray(x), "counts")
