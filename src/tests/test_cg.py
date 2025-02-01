import time

import numpy as np
from matplotlib import pyplot as plt

from src.boundary_conditions import BoundaryCondition, BoundaryConditionType
from src.fluid_dynamics.solvers.stream_function_solvers.cg import (
    ConjugateGradientSolver,
)
from src.geometry import DomainGeometry


def test_cg_elliptic_solver():
    geometry = DomainGeometry(
        width=1.0,
        height=1.0,
        end_time=100,
        n_x=300,
        n_y=300,
        n_t=100,
    )

    x = np.linspace(0, 1, geometry.n_x, dtype=np.float64)
    y = np.linspace(0, 1, geometry.n_y, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    # Exact solution
    analytical_solution = -0.5 * (X**2 + Y**2)

    # Right-hand side (-2 everywhere in the domain)
    rhs = -2 * np.ones((geometry.n_y, geometry.n_x), dtype=np.float64)

    initial_guess = np.zeros((geometry.n_y, geometry.n_x), dtype=np.float64)

    c = np.zeros((geometry.n_y, geometry.n_x), dtype=np.float64)

    solver = ConjugateGradientSolver(
        geometry=geometry,
        top_bc=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_x,
            value_func=lambda t, n: -0.5 * x**2,
        ),
        right_bc=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_y,
            value_func=lambda t, n: -0.5 * (1 + y**2),
        ),
        bottom_bc=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_x,
            value_func=lambda t, n: -0.5 * (x**2 + 1),
        ),
        left_bc=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_y,
            value_func=lambda t, n: -0.5 * y**2,
        ),
        max_iters=10000,
        stopping_criteria=1e-8,
    )

    start_time = time.perf_counter()
    result = solver.solve(initial_guess=initial_guess, c=c, f=-rhs, time=0.0)
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s.")

    error = np.linalg.norm(result - analytical_solution, ord=2)
    print(f"L-2 error: {error}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(
        analytical_solution, extent=[0, 1, 0, 1], origin="lower", cmap="viridis"
    )
    axes[0].set_title("Analytical Solution")
    axes[1].imshow(result, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
    axes[1].set_title(f"Numerical Solution")
    axes[2].imshow(
        result - analytical_solution,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="coolwarm",
    )
    axes[2].set_title("Error (Numerical - Analytical)")

    plt.show()


def analytical_solution_sinusoidal(f0, Lx, Ly, c, m, n, n_x, n_y):
    r"""
    Compute the analytical solution for \Delta u - c u = f with sinusoidal forcing.

    Parameters:
        f0 (float): Amplitude of the forcing term.
        Lx (float): Length of the domain in the x-direction.
        Ly (float): Length of the domain in the y-direction.
        c (float): Constant coefficient in the equation.
        m (int): Mode number in the x-direction.
        n (int): Mode number in the y-direction.
        n_x (int): Number of grid points in the x-direction.
        n_y (int): Number of grid points in the y-direction.

    Returns:
        u (2D array): Analytical solution on the grid.
    """
    # Define the grid
    x = np.linspace(0, Lx, n_x)
    y = np.linspace(0, Ly, n_y)
    X, Y = np.meshgrid(x, y)

    # Compute wavenumbers
    kx = m * np.pi / Lx
    ky = n * np.pi / Ly

    # Compute the solution
    denominator = kx**2 + ky**2 + c
    u = (f0 / denominator) * np.sin(kx * X) * np.sin(ky * Y)

    return u, X, Y


def test_cg_elliptic_solver_2():
    geometry = DomainGeometry(
        width=1.0,
        height=1.0,
        end_time=100,
        n_x=100,
        n_y=100,
        n_t=100,
    )

    # Parameters
    f0 = 1.0  # Amplitude of forcing
    c = 10.0  # Constant coefficient
    m, n = 1, 1  # Mode numbers

    # Compute the solution
    analytical_solution, X, Y = analytical_solution_sinusoidal(
        f0=f0,
        Lx=geometry.width,
        Ly=geometry.height,
        c=c,
        m=m,
        n=n,
        n_x=geometry.n_x,
        n_y=geometry.n_y,
    )

    f = np.zeros((geometry.n_y, geometry.n_x), dtype=np.float64)

    for j in range(1, geometry.n_y - 1):
        for i in range(1, geometry.n_x - 1):
            f[j, i] = -(
                f0
                * np.sin(m * np.pi * i * geometry.dx / geometry.width)
                * np.sin(n * np.pi * j * geometry.dy / geometry.height)
            )

    initial_guess = np.zeros((geometry.n_y, geometry.n_x), dtype=np.float64)

    c = 10.0 * np.ones((geometry.n_y, geometry.n_x), dtype=np.float64)

    solver = ConjugateGradientSolver(
        geometry=geometry,
        top_bc=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_x,
            value_func=lambda t, n: np.zeros(n),
        ),
        right_bc=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_y,
            value_func=lambda t, n: np.zeros(n),
        ),
        bottom_bc=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_x,
            value_func=lambda t, n: np.zeros(n),
        ),
        left_bc=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_y,
            value_func=lambda t, n: np.zeros(n),
        ),
        max_iters=10000,
        stopping_criteria=1e-8,
    )

    start_time = time.perf_counter()
    result = solver.solve(initial_guess=initial_guess, c=c, f=-f, time=0.0)
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s.")

    error = np.linalg.norm(result - analytical_solution, ord=2)
    print(f"L-2 error: {error}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(
        analytical_solution, extent=[0, 1, 0, 1], origin="lower", cmap="viridis"
    )
    axes[0].set_title("Analytical Solution")
    axes[1].imshow(result, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
    axes[1].set_title(f"Numerical Solution")
    axes[2].imshow(
        result - analytical_solution,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="coolwarm",
    )
    axes[2].set_title("Error (Numerical - Analytical)")

    plt.show()


print("TEST 1")
test_cg_elliptic_solver()
print("TEST 2")
test_cg_elliptic_solver_2()
