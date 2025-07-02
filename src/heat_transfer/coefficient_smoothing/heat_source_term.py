from typing import Tuple
import numpy as np
from numba import njit
from numba.typed.typedlist import List


@njit
def find_interface_points(u, u_pt, dx, dy) -> List:
    n_y, n_x = u.shape
    interface_points = List()

    # Check horizontal edges (between vertically adjacent cells)
    for j in range(n_y - 1):
        for i in range(n_x):
            u1 = u[j, i]
            u2 = u[j + 1, i]

            # Check if interface crosses this edge
            if (u1 <= u_pt <= u2) or (u2 <= u_pt <= u1):
                # Handle case where interface is exactly at a node
                if u1 == u_pt:
                    point = (i * dx, j * dy)
                elif u2 == u_pt:
                    point = (i * dx, (j + 1) * dy)
                else:
                    # Linear interpolation to find exact crossing point
                    if abs(u2 - u1) > 1e-12:  # Avoid division by very small numbers
                        alpha = (u_pt - u1) / (u2 - u1)
                        point = (i * dx, (j + alpha) * dy)
                    else:
                        continue  # Skip if temperature difference is too small

                # Check for exact duplicates before adding
                is_duplicate = False
                for k in range(len(interface_points)):
                    if (
                        interface_points[k][0] == point[0]
                        and interface_points[k][1] == point[1]
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    interface_points.append(point)

    # Check vertical edges (between horizontally adjacent cells)
    for j in range(n_y):
        for i in range(n_x - 1):
            u1 = u[j, i]
            u2 = u[j, i + 1]

            # Check if interface crosses this edge
            if (u1 <= u_pt <= u2) or (u2 <= u_pt <= u1):
                # Handle case where interface is exactly at a node
                if u1 == u_pt:
                    point = (i * dx, j * dy)
                elif u2 == u_pt:
                    point = ((i + 1) * dx, j * dy)
                else:
                    # Linear interpolation to find exact crossing point
                    if abs(u2 - u1) > 1e-12:  # Avoid division by very small numbers
                        alpha = (u_pt - u1) / (u2 - u1)
                        point = ((i + alpha) * dx, j * dy)
                    else:
                        continue  # Skip if temperature difference is too small

                # Check for exact duplicates before adding
                is_duplicate = False
                for k in range(len(interface_points)):
                    if (
                        interface_points[k][0] == point[0]
                        and interface_points[k][1] == point[1]
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    interface_points.append(point)

    return interface_points


@njit
def compute_segment_lengths(
    interface_pts: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    dx: float,
    dy: float,
    search_radius: float,
) -> np.ndarray:
    N = interface_pts.shape[0]
    seglen = np.empty(N, dtype=np.float64)
    r2max = search_radius * search_radius

    for i in range(N):
        xi, yi = interface_pts[i, 0], interface_pts[i, 1]
        # compute local normal & tangent
        Tx = interpolate_to_point(grad_x, xi, yi, dx, dy)
        Ty = interpolate_to_point(grad_y, xi, yi, dx, dy)
        mag = np.hypot(Tx, Ty) + 1e-16
        tx, ty = -Ty / mag, Tx / mag  # unit tangent

        best_left = np.inf
        best_right = np.inf

        # search neighbors
        for j in range(N):
            if j == i:
                continue
            xj, yj = interface_pts[j, 0], interface_pts[j, 1]
            dxij = xj - xi
            dyij = yj - yi
            if dxij * dxij + dyij * dyij > r2max:
                continue
            proj = dxij * tx + dyij * ty

            if 0.0 < proj < best_right:
                best_right = proj
            elif proj < 0.0 and -proj < best_left:
                best_left = -proj

        # form length
        if best_left < np.inf and best_right < np.inf:
            ds_i = 0.5 * (best_left + best_right)
        elif best_left < np.inf:
            ds_i = 0.5 * best_left
        elif best_right < np.inf:
            ds_i = 0.5 * best_right
        else:
            ds_i = max(dx, dy)

        # enforce minimum
        seglen[i] = max(ds_i, 0.1 * max(dx, dy))

    return seglen


@njit
def compute_temperature_gradients(
    u: np.ndarray, dx: float, dy: float
) -> Tuple[np.ndarray, np.ndarray]:
    grad_x = np.zeros_like(u)
    grad_y = np.zeros_like(u)

    # Central differences in interior
    grad_y[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
    grad_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)

    # Forward/backward differences at boundaries
    grad_y[0, :] = (-3 * u[0, :] + 4 * u[1, :] - u[2, :]) / (2 * dy)
    grad_y[-1, :] = (3 * u[-1, :] - 4 * u[-2, :] + u[-3, :]) / (2 * dy)
    grad_x[:, 0] = (-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2]) / (2 * dx)
    grad_x[:, -1] = (3 * u[:, -1] - 4 * u[:, -2] + u[:, -3]) / (2 * dx)

    return grad_x, grad_y


@njit
def interpolate_to_point(
    field: np.ndarray, x: float, y: float, dx: float, dy: float
) -> float:
    """Bilinear interpolation of field value at point (x, y)"""
    n_y, n_x = field.shape

    # Find surrounding grid points
    i = max(0, min(int(x / dx + 1e-16), n_x - 2))
    j = max(0, min(int(y / dy + 1e-16), n_y - 2))

    # Local coordinates
    xi = (x - i * dx) / dx
    eta = (y - j * dy) / dy

    # Bilinear interpolation
    f00 = field[j, i]
    f10 = field[j, i + 1]
    f01 = field[j + 1, i]
    f11 = field[j + 1, i + 1]

    return float(
        f00 * (1 - xi) * (1 - eta)
        + f10 * xi * (1 - eta)
        + f01 * (1 - xi) * eta
        + f11 * xi * eta
    )


@njit
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
    if interface_points.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    n_y, n_x = u.shape
    grad_x, grad_y = compute_temperature_gradients(u, dx, dy)
    LV_n_values = np.empty(interface_points.shape[0], dtype=np.float64)

    for idx in range(interface_points.shape[0]):
        x_int = float(interface_points[idx, 0])
        y_int = float(interface_points[idx, 1])

        # Interpolate gradient at interface point
        grad_x_int = interpolate_to_point(grad_x, x_int, y_int, dx, dy)
        grad_y_int = interpolate_to_point(grad_y, x_int, y_int, dx, dy)

        # Normal vector (pointing from solid to liquid)
        grad_mag = np.sqrt(grad_x_int**2 + grad_y_int**2)
        if grad_mag > 1e-12:
            nx = grad_x_int / grad_mag
            ny = grad_y_int / grad_mag
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
        LV_n_values[idx] = LV_n

    return LV_n_values


@njit
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
    interface_points = np.asarray(find_interface_points(u, u_pt, dx, dy))

    print(interface_points.shape)

    if interface_points.shape[0] == 0:
        return np.zeros_like(u)

    # Compute L*V_n at interface points
    LV_n_values = compute_normal_derivatives_at_interface(
        u, u_pt, interface_points, dx, dy, k_l, k_s, k_0, peclet_number
    )

    # Distribute each L*V_n value to nearby grid points
    source_term = np.zeros_like(u)
    sigma = 2 * min(dx, dy)  # Gaussian width

    # Create coordinate arrays
    for i in range(n_x):
        for j in range(n_y):
            x_grid = dx * i
            y_grid = dy * j

            for k in range(interface_points.shape[0]):
                x_int = interface_points[k, 0]
                y_int = interface_points[k, 1]
                LV_n = LV_n_values[k]

                # Create local Gaussian centered at this interface point
                distance_sq = (x_grid - x_int) ** 2 + (y_grid - y_int) ** 2
                local_delta = np.exp(-0.5 * distance_sq / sigma**2) / (
                    sigma * np.sqrt(2 * np.pi)
                )

                # Add contribution from this interface point
                source_term[j, i] -= local_delta * LV_n

    return source_term
