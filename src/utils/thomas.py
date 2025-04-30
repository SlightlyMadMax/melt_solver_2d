import numpy as np
from numba import njit


@njit
def solve_tridiagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    f: np.ndarray,
    result: np.ndarray,
    left_type: int,
    right_type: int,
    left_value: float = 0.0,
    right_value: float = 0.0,
    left_flux: float = 0.0,
    left_psi: float = 0.0,
    left_phi: float = 0.0,
    right_flux: float = 0.0,
    right_psi: float = 0.0,
    right_phi: float = 0.0,
    h: float = 0.0,
):
    """
    Solve a tridiagonal system using the Thomas algorithm with support for Dirichlet, Neumann
    and Robin boundary conditions.

    :param a: Sub-diagonal elements of the tridiagonal matrix.
    :param b: Diagonal elements of the tridiagonal matrix.
    :param c: Super-diagonal elements of the tridiagonal matrix.
    :param f: Right-hand side vector.
    :param result: Link to the array where the result will be stored.
    :param left_type: Type of left boundary condition (1: Dirichlet, 2: Neumann, 3: Robin).
    :param left_value: Value for the left boundary condition (optional).
    :param left_flux: Flux value for Neumann condition on the left (optional).
    :param left_psi: Psi parameter for Robin condition on the left (optional).
    :param left_phi: Phi parameter for Robin condition on the left (optional).
    :param right_type: Type of right boundary condition (1: Dirichlet, 2: Neumann, 3: Robin).
    :param right_value: Value for the right boundary condition (optional).
    :param right_flux: Flux value for Neumann condition on the right (optional).
    :param right_psi: Psi parameter for Robin condition on the right (optional).
    :param right_phi: Phi parameter for Robin condition on the right (optional).
    :param h: The grid step size (dx for x-direction, dy for y-direction) (optional).
    :return: None.
    """
    n = len(f)
    alpha = np.zeros(n - 1)
    beta = np.zeros(n - 1)

    if left_type != 1 or right_type != 1:
        assert h != 0.0, "Please, set the grid step size"

    # Boundary condition handling for the left side
    if left_type == 1:  # Dirichlet
        alpha[0] = 0.0
        beta[0] = left_value
    elif left_type == 2:  # Neumann
        alpha[0] = 1.0
        beta[0] = -h * left_flux
    else:  # Robin
        alpha[0] = 1.0 / (1.0 + h * left_phi)
        beta[0] = -h * left_psi / (1.0 + h * left_phi)

    # Forward sweep
    for j in range(1, n - 1):
        alpha[j] = -a[j] / (b[j] + c[j] * alpha[j - 1])
        beta[j] = (f[j] - c[j] * beta[j - 1]) / (b[j] + c[j] * alpha[j - 1])

    # Boundary condition handling for the right side
    if right_type == 1:  # Dirichlet
        result[n - 1] = right_value
    elif right_type == 2:  # Neumann
        result[n - 1] = (h * right_flux + beta[n - 2]) / (1 - alpha[n - 2])
    else:  # Robin
        result[n - 1] = (h * right_psi + beta[n - 2]) / (
            1 - alpha[n - 2] - h * right_phi
        )

    # Backward substitution to find the solution
    for j in range(n - 2, -1, -1):
        result[j] = alpha[j] * result[j + 1] + beta[j]
