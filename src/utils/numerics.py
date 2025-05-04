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


@njit
def compute_gradient(u, i, j, h_x, h_y):
    """Central difference gradient at (j,i)"""
    du_dx = (u[j, i + 1] - u[j, i - 1]) / (2 * h_x)
    du_dy = (u[j + 1, i] - u[j - 1, i]) / (2 * h_y)
    return (du_dx**2 + du_dy**2) ** 0.5
