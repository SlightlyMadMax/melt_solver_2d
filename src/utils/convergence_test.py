import math
import numpy as np
import matplotlib.pyplot as plt


def solve_advection_diffusion(nx, scheme="upwind"):
    """Solve 1D advection-diffusion equation with given grid resolution"""

    # Physical parameters
    u = 0.05  # advection velocity
    alpha = 1e-3  # diffusion coefficient
    L = 3.0  # domain length
    t_end = 5.0  # final time

    # Grid setup
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    # Time step - ensure stability and reasonable temporal accuracy
    # For implicit schemes, we can use larger time steps, but for convergence
    # testing we should scale dt with dx to maintain temporal accuracy
    dt = dx**2  # Scale with dx for better convergence
    nt = int(t_end / dt)
    actual_t_end = nt * dt

    # Dimensionless numbers
    C = u * dt / dx  # Courant number
    Fo = alpha * dt / dx**2  # Fourier number

    print(f"nx={nx}, dx={dx:.4f}, dt={dt:.6f}, C={C:.3f}, Fo={Fo:.6f}")

    # Initial condition: step function
    phi_left = 1.0
    phi_right = 0.0
    x0 = L / 3
    phi = phi_left * 0.5 * (1 + np.tanh((x0 - x) / 0.1))

    # Build coefficient matrix based on scheme
    A = np.zeros((nx, nx))

    # Boundary conditions
    A[0, 0] = 1.0
    A[nx - 1, nx - 1] = 1.0

    if scheme == "upwind":
        for i in range(1, nx - 1):
            A[i, i - 1] = -C - Fo
            A[i, i] = 1 + C + 2 * Fo
            A[i, i + 1] = -Fo

    elif scheme == "central":
        for i in range(1, nx - 1):
            A[i, i - 1] = -0.5 * C - Fo
            A[i, i] = 1 + 2 * Fo
            A[i, i + 1] = 0.5 * C - Fo

    elif scheme == "samarski":
        eps = alpha / (1 + dx * u * 0.5 / alpha)
        for i in range(1, nx - 1):
            A[i, i - 1] = -C - eps * dt / dx**2
            A[i, i] = 1 + C + 2 * eps * dt / dx**2
            A[i, i + 1] = -eps * dt / dx**2

    # Time stepping
    for n in range(nt):
        # Apply boundary conditions
        rhs = phi.copy()
        rhs[0] = phi_left
        rhs[nx - 1] = phi_right

        phi = np.linalg.solve(A, rhs)

    return x, phi, actual_t_end


def analytical_solution(x, t, u, alpha, phi_left, phi_right, x0) -> np.ndarray:
    """Analytical solution for advection-diffusion with step initial condition in the infinite domain"""
    phi_analytical = np.zeros_like(x)

    for i, xi in enumerate(x):
        y = xi - x0
        w = (y - u * t) / t**0.5

        if t > 0:
            phi_analytical[i] = phi_left * 0.5 * (1 - math.erf(w * 0.5 / alpha**0.5))
        else:
            phi_analytical[i] = phi_left if xi <= x0 else phi_right

    return phi_analytical


def convergence_test():
    """Perform convergence test for all three schemes"""

    # Grid resolutions to test
    nx_values = [51, 101, 151, 201, 301, 401, 501]

    schemes = ["upwind", "central", "samarski"]
    errors = {scheme: [] for scheme in schemes}
    dx_values = []

    # Physical parameters
    u = 0.05
    alpha = 1e-3
    L = 3.0
    phi_left = 1.0
    phi_right = 0.0
    x0 = L / 3

    for nx in nx_values:
        dx = L / nx
        dx_values.append(dx)

        for scheme in schemes:
            # Solve numerically
            x, phi_num, t_end = solve_advection_diffusion(nx, scheme)

            # Compute analytical solution
            phi_analytical = analytical_solution(x, t_end, u, alpha, phi_left, phi_right, x0)

            # Compute L2 error (exclude boundary points)
            error = np.linalg.norm(phi_num[1:-1] - phi_analytical[1:-1]) / np.sqrt(
                nx - 2
            )
            errors[scheme].append(float(error))

            print(f"Scheme: {scheme:8s}, nx: {nx:3d}, Error: {error:.6e}")
            print()

    for scheme in errors.keys():
        for error in errors[scheme]:
            print(error)
        print()

    # Plot convergence
    # plt.figure(figsize=(10, 6))
    #
    # for scheme in schemes:
    #     plt.loglog(
    #         dx_values, errors[scheme], "o-", label=f"{scheme.capitalize()} scheme"
    #     )
    #
    # # Add reference lines for convergence rates
    # dx_array = np.array(dx_values)
    # plt.loglog(dx_array, 0.1 * dx_array, "k--", alpha=0.5, label="1st order")
    # plt.loglog(dx_array, 0.01 * dx_array**2, "k:", alpha=0.5, label="2nd order")
    #
    # plt.xlabel("Grid spacing (dx)")
    # plt.ylabel("L2 Error")
    # plt.title("Convergence Test: 1D Advection-Diffusion Equation")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # Print convergence rates
    # print("\nConvergence rates:")
    # for scheme in schemes:
    #     rates = []
    #     for i in range(1, len(errors[scheme])):
    #         rate = np.log(errors[scheme][i] / errors[scheme][i - 1]) / np.log(
    #             dx_values[i] / dx_values[i - 1]
    #         )
    #         rates.append(rate)
    #     avg_rate = np.mean(rates[-3:])  # Average of last 3 rates
    #     print(f"{scheme.capitalize()} scheme: {avg_rate:.2f}")


if __name__ == "__main__":
    convergence_test()
