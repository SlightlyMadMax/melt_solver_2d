import numpy as np
from numba import njit


@njit
def solve_tridiagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    f: np.ndarray,
    result: np.ndarray,
) -> None:
    """
    Thomas solver for a full tridiagonal system A u = f.
    :param a: sub-diagonal of A (length n). Elements a[0] to a[n-2] are the sub-diagonal entries; a[n-1] is unused.
    :param b: diagonal of A (length n), elements b[0..n-1].
    :param c: super-diagonal of A (length n). Elements c[1] to c[n-1] are the super-diagonal entries; c[0] is unused.
    :param f: right-hand side (length n).
    :param result: output array (length n) where the solution u is stored.
    :return: None.
    """
    n = f.shape[0]
    alpha = np.empty(n, dtype=np.float64)
    beta = np.empty(n, dtype=np.float64)

    alpha[0] = -a[0] / b[0]
    beta[0] = f[0] / b[0]

    for j in range(1, n):
        denom = b[j] + c[j] * alpha[j - 1]
        alpha[j] = -a[j] / denom
        beta[j] = (f[j] - c[j] * beta[j - 1]) / denom

    result[n - 1] = beta[n - 1]
    for j in range(n - 2, -1, -1):
        result[j] = alpha[j] * result[j + 1] + beta[j]
