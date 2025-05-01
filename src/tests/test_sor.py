import time

import numpy as np
from matplotlib import pyplot as plt

from src.core.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditionType,
    BoundaryConditions,
)
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.solvers.stream_function_solvers.sor import SORPoissonSolver


def test_sor_poisson_solver():
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

    solver = SORPoissonSolver(
        geometry=geometry,
        bcs=BoundaryConditions(
            top=BoundaryCondition(
                boundary_type=BoundaryConditionType.DIRICHLET,
                n=geometry.n_x,
                value_func=lambda t, n: -0.5 * x**2,
            ),
            right=BoundaryCondition(
                boundary_type=BoundaryConditionType.DIRICHLET,
                n=geometry.n_y,
                value_func=lambda t, n: -0.5 * (1 + y**2),
            ),
            bottom=BoundaryCondition(
                boundary_type=BoundaryConditionType.DIRICHLET,
                n=geometry.n_x,
                value_func=lambda t, n: -0.5 * (x**2 + 1),
            ),
            left=BoundaryCondition(
                boundary_type=BoundaryConditionType.DIRICHLET,
                n=geometry.n_y,
                value_func=lambda t, n: -0.5 * y**2,
            ),
        ),
        max_iters=10000,
    )

    initial_guess = np.zeros((geometry.n_y, geometry.n_x), dtype=np.float64)

    start_time = time.perf_counter()
    result = solver.solve(initial_guess=initial_guess, rhs=rhs, time=0.0)
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


test_sor_poisson_solver()
