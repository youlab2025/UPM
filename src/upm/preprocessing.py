"""Explicit conversion of control expression to log1p(CP10K)."""
import numpy as np
from scipy import sparse


def validate_expression(x, *, log_space=False):
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("Expression must be a nonempty cells × genes matrix")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Expression must be finite and nonnegative")
    if log_space and (x > np.log1p(10000.) + 1e-3).any():
        raise ValueError("Controls must be log1p(CP10K); use input-space counts for raw counts")


def prepare_controls(x, input_space):
    """Normalize across all input gene columns, preserving their order.

    Do not subset genes before count normalization. A zero-count cell remains
    zero. Already normalized expression is validated and never normalized twice.
    """
    if input_space not in ("counts", "log1p-cp10k"):
        raise ValueError("input_space must be counts or log1p-cp10k")
    x = x.toarray() if sparse.issparse(x) else np.asarray(x)
    validate_expression(x, log_space=input_space == "log1p-cp10k")
    if (x > np.finfo(np.float32).max).any():
        raise ValueError("Expression exceeds float32 range")
    if input_space == "counts":
        x = x.astype(np.float64)
        totals = x.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1
        x = np.log1p(x / totals * 10000.)
    return np.asarray(x, dtype=np.float32)
