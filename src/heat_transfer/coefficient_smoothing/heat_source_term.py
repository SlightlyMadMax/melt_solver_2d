from typing import Tuple

import numpy as np


def find_interface_points(
    u: np.ndarray, u_pt: float, dx: float, dy: float
) -> np.ndarray:
    """Find interface points where T = T_fusion using interpolation"""
    n_y, n_x = u.shape
    interface_points = []

    # Check horizontal edges
    for i in range(n_x - 1):
        for j in range(n_y):
            u1, u2 = u[j, i], u[j, i + 1]
            if (u1 - u_pt) * (u2 - u_pt) <= 0 and u1 != u2:
                # Linear interpolation to find exact interface location
                alpha = (u_pt - u1) / (u2 - u1)
                x_int = dx * i + alpha * dx
                y_int = dy * j
                interface_points.append([x_int, y_int])

    # Check vertical edges
    for i in range(n_x):
        for j in range(n_y - 1):
            u1, u2 = u[j, i], u[j + 1, i]
            if (u1 - u_pt) * (u2 - u_pt) <= 0 and u1 != u2:
                alpha = (u_pt - u1) / (u2 - u1)
                x_int = dx * i
                y_int = dy * j + alpha * dy
                interface_points.append([x_int, y_int])

    return (
        np.array(interface_points) if interface_points else np.array([]).reshape(0, 2)
    )


def compute_temperature_gradients(
    u: np.ndarray, dx: float, dy: float
) -> Tuple[np.ndarray, np.ndarray]:
    grad_x = np.zeros_like(u)
    grad_y = np.zeros_like(u)

    # Central differences in interior
    grad_y[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
    grad_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)

    # Forward/backward differences at boundaries
    grad_y[0, :] = (u[1, :] - u[0, :]) / dy
    grad_y[-1, :] = (u[-1, :] - u[-2, :]) / dy
    grad_x[:, 0] = (u[:, 1] - u[:, 0]) / dx
    grad_x[:, -1] = (u[:, -1] - u[:, -2]) / dx

    return grad_x, grad_y


def interpolate_to_point(n_x, n_y, field, x, y, dx, dy):
    """Bilinear interpolation of field value at point (x, y)"""
    # Find surrounding grid points
    i = np.clip(int(x / dx), 0, n_x - 2)
    j = np.clip(int(y / dy), 0, n_y - 2)

    # Local coordinates
    xi = (x - i * dx) / dx
    eta = (y - j * dy) / dy

    # Bilinear interpolation
    f00 = field[j, i]
    f10 = field[j, i + 1]
    f01 = field[j + 1, i]
    f11 = field[j + 1, i + 1]

    return (
        f00 * (1 - xi) * (1 - eta)
        + f10 * xi * (1 - eta)
        + f01 * (1 - xi) * eta
        + f11 * xi * eta
    )


def compute_normal_derivatives_at_interface(
    u: np.ndarray,
    u_pt: float,
    interface_points: np.ndarray,
    dx: float,
    dy: float,
    k_l: float,
    k_s: float,
    k_0: float,
    peclet_number: float,
) -> np.ndarray:
    """Compute normal temperature derivatives at interface points"""
    if len(interface_points) == 0:
        return np.array([])

    n_y, n_x = u.shape

    grad_x, grad_y = compute_temperature_gradients(u, dx, dy)
    LV_n_values = []

    for point in interface_points:
        x_int, y_int = point

        # Interpolate gradient at interface point
        grad_x_int = interpolate_to_point(n_x, n_y, grad_x, x_int, y_int, dx, dy)
        grad_y_int = interpolate_to_point(n_x, n_y, grad_y, x_int, y_int, dx, dy)

        # Normal vector (pointing from solid to liquid)
        grad_mag = np.sqrt(grad_x_int**2 + grad_y_int**2)
        if grad_mag > 1e-12:
            nx = grad_x_int / grad_mag  # Changed variable name to avoid conflict
            ny = grad_y_int / grad_mag  # Changed variable name to avoid conflict
        else:
            nx, ny = 1.0, 0.0  # Default normal if gradient is zero

        # Compute one-sided derivatives along normal
        h = min(dx, dy) * 0.5  # Step size for derivative approximation

        # Liquid side (positive normal direction)
        x_l, y_l = x_int + h * nx, y_int + h * ny
        u_l = interpolate_to_point(n_x, n_y, u, x_l, y_l, dx, dy)
        du_dn_l = (u_l - u_pt) / h

        # Solid side (negative normal direction)
        x_s, y_s = x_int - h * nx, y_int - h * ny
        u_s = interpolate_to_point(n_x, n_y, u, x_s, y_s, dx, dy)
        du_dn_s = (u_pt - u_s) / h

        # Stefan condition: k_l * dT_l/dn - k_s * dT_s/dn = L * V_n
        LV_n = (k_l * du_dn_l - k_s * du_dn_s) / (k_0 * peclet_number)
        LV_n_values.append(LV_n)

    return np.array(LV_n_values)


def compute_stefan_source_term(
    u: np.ndarray,
    u_pt: float,
    dx: float,
    dy: float,
    k_l: float,
    k_s: float,
    k_0: float,
    peclet_number: float,
) -> np.ndarray:
    """Compute the Stefan source term -δ_s * L * V_n"""
    n_y, n_x = u.shape

    # Find interface points
    interface_points = find_interface_points(u, u_pt, dx, dy)

    if len(interface_points) == 0:
        return np.zeros_like(u)

    # Compute L*V_n at interface points
    LV_n_values = compute_normal_derivatives_at_interface(
        u, u_pt, interface_points, dx, dy, k_l, k_s, k_0, peclet_number
    )

    # Distribute each L*V_n value to nearby grid points
    source_term = np.zeros_like(u)
    sigma = min(dx, dy)  # Gaussian width

    x = np.linspace(0, (n_x - 1) * dx, n_x)
    y = np.linspace(0, (n_y - 1) * dy, n_y)
    X, Y = np.meshgrid(x, y, indexing="xy")

    for k, (interface_point, LV_n) in enumerate(zip(interface_points, LV_n_values)):
        x_int, y_int = interface_point

        # Create local Gaussian centered at this interface point
        distances_sq = (X - x_int) ** 2 + (Y - y_int) ** 2
        local_delta = np.exp(-0.5 * distances_sq / sigma**2) / (
            sigma * np.sqrt(2 * np.pi)
        )

        # Add contribution from this interface point
        source_term -= local_delta * LV_n

    return source_term
