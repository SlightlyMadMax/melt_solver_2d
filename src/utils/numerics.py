import numpy as np
import scipy.sparse.linalg as spla
from numba import njit


@njit(inline="always")
def harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b) if (a + b) != 0 else 0.0


def is_positive_definite(A) -> bool:
    """Check if a sparse matrix A is positive definite using eigenvalues."""
    eigvals = spla.eigs(
        A=A,
        k=min(A.shape[0] - 1, 5),
        which="SM",
        return_eigenvectors=False,
    )
    return np.all(eigvals > 0)
