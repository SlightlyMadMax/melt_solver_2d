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
    Solve a tridiagonal system Ax = f using the Thomas algorithm.

    NOTE: `a` is the SUPER-diagonal (coefficient of x[j+1]) and `c` is the
    SUB-diagonal (coefficient of x[j-1]). Each row j satisfies:
        a[j]*x[j+1] + b[j]*x[j] + c[j]*x[j-1] = f[j]
    a[n-1] and c[0] are unused.

    :param a: super-diagonal coefficients (length n); a[n-1] unused.
    :param b: main diagonal coefficients (length n).
    :param c: sub-diagonal coefficients (length n); c[0] unused.
    :param f: right-hand side vector (length n).
    :param result: output array (length n) where the solution x is stored.
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
