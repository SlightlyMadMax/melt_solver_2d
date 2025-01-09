import numpy as np
from numpy.testing import assert_almost_equal

from src.utils import solve_tridiagonal


def test_solve_tridiagonal():
    n = 5
    a = -1 * np.ones(n - 1, dtype=np.float64)
    b = 2 * np.ones(n, dtype=np.float64)
    c = -1 * np.ones(n - 1, dtype=np.float64)
    f = np.array([1, 1, 1, 1, 1], dtype=np.float64)

    # Dirichlet boundary conditions
    left_type = 1
    right_type = 1
    left_value = 2.5
    right_value = 2.5

    result = np.zeros_like(f)

    solve_tridiagonal(a, b, c, f, result, left_type, right_type, left_value, right_value)

    A = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(A, b)
    np.fill_diagonal(A[1:], a)
    np.fill_diagonal(A[:, 1:], c)

    expected_result = np.linalg.solve(A, f)

    assert_almost_equal(result, expected_result, decimal=6)
    print("Test passed!")


test_solve_tridiagonal()

