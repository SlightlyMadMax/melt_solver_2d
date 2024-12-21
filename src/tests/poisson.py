from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from src.utils import solve_poisson_sor


def test_solve_poisson(
    solve_poisson: Callable,
):
    nx, ny = 300, 300
    dx = dy = 1.0 / (nx - 1)

    x = np.linspace(0, 1, nx, dtype=np.float64)
    y = np.linspace(0, 1, ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    # Exact solution
    analytical_solution = -0.5 * (X**2 + Y**2)

    # Right-hand side (-2 everywhere in the domain)
    rhs = -2 * np.ones((ny, nx), dtype=np.float64)

    # Boundary conditions
    top_value = -0.5 * x**2
    bottom_value = -0.5 * (x**2 + 1)
    left_value = -0.5 * y**2
    right_value = -0.5 * (1 + y**2)

    initial_guess = np.zeros((ny, nx), dtype=np.float64)

    result = solve_poisson(
        initial_guess=initial_guess,
        rhs=-rhs,
        dx=dx,
        dy=dy,
        max_iters=10000,
        stopping_criteria=1e-6,
        right_value=right_value,
        left_value=left_value,
        top_value=top_value,
        bottom_value=bottom_value,
    )

    error = np.linalg.norm(result - analytical_solution, ord=2)
    print(f"L-2 error: {error}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(
        analytical_solution, extent=[0, 1, 0, 1], origin="lower", cmap="viridis"
    )
    axes[0].set_title("Analytical Solution")
    axes[1].imshow(result, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
    axes[1].set_title(f"Numerical Solution (using {solve_poisson.__name__})")
    axes[2].imshow(
        result - analytical_solution,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="coolwarm",
    )
    axes[2].set_title("Error (Numerical - Analytical)")

    plt.show()


test_solve_poisson(solve_poisson_sor)
