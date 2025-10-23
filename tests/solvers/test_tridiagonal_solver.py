import numpy as np
from numpy.testing import assert_almost_equal

from src.core.solvers.tridiagonal_solver import solve_tridiagonal


def test_solve_tridiagonal_bc_1():
    n = 10
    a = np.ones(n, dtype=np.float64)
    b = -2 * np.ones(n, dtype=np.float64)
    c = np.ones(n, dtype=np.float64)
    f = np.ones(n)

    left_value = 2.5
    right_value = 2.5

    # Dirichlet boundary conditions
    a[0] = 0.0
    b[0] = 1.0
    f[0] = left_value

    b[n - 1] = 1.0
    c[n - 1] = 0.0
    f[n - 1] = right_value

    result = np.zeros(n)

    solve_tridiagonal(a, b, c, f, result)

    a = np.ones(n - 3, dtype=np.float64)
    b = -2 * np.ones(n - 2, dtype=np.float64)
    c = np.ones(n - 3, dtype=np.float64)
    f = np.ones(n - 2)

    A = np.zeros((n - 2, n - 2), dtype=np.float64)
    np.fill_diagonal(A, b)
    np.fill_diagonal(A[1:], a)
    np.fill_diagonal(A[:, 1:], c)

    f[0] -= left_value
    f[-1] -= right_value

    expected_result = np.zeros(n)
    expected_result[1:-1] = np.linalg.solve(A, f)
    expected_result[0] = left_value
    expected_result[-1] = right_value

    assert_almost_equal(result, expected_result, decimal=6)
    print("Test passed (Dirichlet bc).")


def test_solve_tridiagonal_bc_2():
    n = 10
    a = np.ones(n, dtype=np.float64)
    b = -2 * np.ones(n, dtype=np.float64)
    c = np.ones(n, dtype=np.float64)
    f = np.ones(n)

    left_flux = 0.0
    right_value = 2.5

    # Neumann bc on the left
    a[0] = 1.0
    b[0] = -1.0
    f[0] = left_flux

    # Dirichlet bc on the right
    b[n - 1] = 1.0
    c[n - 1] = 0.0
    f[n - 1] = right_value

    result = np.zeros(n)

    solve_tridiagonal(a, b, c, f, result)

    a = np.ones(n - 1, dtype=np.float64)
    b = -2 * np.ones(n, dtype=np.float64)
    c = np.ones(n - 1, dtype=np.float64)
    f = np.ones(n)

    A = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(A, b)
    np.fill_diagonal(A[1:], a)
    np.fill_diagonal(A[:, 1:], c)

    A[0, 0] = -1
    A[-1, -1] = -1
    A[-1, -2] = 0

    f[0] = left_flux
    f[-1] = -right_value

    expected_result = np.linalg.solve(A, f)

    assert_almost_equal(result, expected_result, decimal=6)
    print("Test passed (Neumann bc).")


test_solve_tridiagonal_bc_1()
test_solve_tridiagonal_bc_2()
