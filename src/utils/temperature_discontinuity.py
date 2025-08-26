import numpy as np
from matplotlib import pyplot as plt

from src.core.geometry import DomainGeometry
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from tests.numerical_experiments.one_dim.analytic_solution_1d_2ph import (
    get_ice_temp,
    get_water_temp,
    calculate_gamma,
)
from src.core.constants import ABS_ZERO


geometry = DomainGeometry(
    width=1.0,
    height=1.0,
    end_time=60.0 * 60.0 * 24.0,
    n_x=5,
    n_y=301,
    n_t=60 * 24,
)

material_props = MaterialProperties(
    u_pt=273.15,
    specific_heat_liquid=4120.7,
    specific_heat_solid=2056.8,
    specific_latent_heat=3.33e5,
    density_liquid=999.84,
    density_solid=918.9,
    thermal_conductivity_liquid=0.59,
    thermal_conductivity_solid=2.21,
    kinematic_viscosity_coeffs=[
        0.000108963453,
        -9.28722151e-07,
        2.65889022e-09,
        -2.54761652e-12,
    ],
    volumetric_thermal_exp_coeffs=[7.68e-6],
)

max_temp = 278.15
min_temp = 268.15

cfg = ExperimentConfig(
    geometry=geometry,
    material_props=material_props,
    u_ref=0.5 * (min_temp + max_temp),
    delta_u=0.5 * (max_temp - min_temp),
    v=0.01,
    l=geometry.max_dimension,
    u_solid=273.0,
    u_liquid=273.3,
    epsilon=1e-6,
    save_interval=1,
)

gamma = calculate_gamma(cfg=cfg, min_temp=min_temp, max_temp=max_temp)
t = geometry.end_time
s = t**0.5 * gamma  # interface position

print(f"True interface position: s = {s:.6f}")
u_star = material_props.u_pt + ABS_ZERO


def lin_interp_interface(xs: np.ndarray, u: np.ndarray, u_pt: float, h: float) -> float:
    thetas = u - u_pt
    i = np.where(thetas[:-1] * thetas[1:] < 0)[0][0]

    s_lin = xs[i] + h * (u_pt - u[i]) / (u[i + 1] - u[i])

    return s_lin


def piecewise_interface(xs: np.ndarray, u: np.ndarray, u_pt: float, h: float) -> float:
    thetas = u - u_pt
    i = np.where(thetas[:-1] * thetas[1:] < 0)[0][0]

    m_left = (u[i] - u[i - 1]) / h
    s_left_piece = xs[i] + (u_pt - u[i]) / m_left

    m_right = (u[i + 2] - u[i + 1]) / h
    s_right_piece = xs[i + 1] + (u_pt - u[i + 1]) / m_right

    return 0.5 * (s_left_piece + s_right_piece)


def quadratic_interface(xs: np.ndarray, u: np.ndarray, u_pt: float, h: float) -> float:
    """
    Fit separate quadratics on each side of the interface and find intersection.
    Uses 3 points on each side to fit quadratic polynomials.

    Parameters:
    xs: spatial grid points
    u: temperature values at grid points
    u_pt: phase transition temperature
    h: grid spacing

    Returns:
    s_quad: interface position from quadratic reconstruction
    """
    thetas = u - u_pt
    i = np.where(thetas[:-1] * thetas[1:] < 0)[0][0]

    # Fit quadratic on the left side (solid phase)
    # Use points i-2, i-1, i (ensuring we have enough points)
    if i >= 2:
        x_left = xs[i - 2 : i + 1]
        u_left = u[i - 2 : i + 1]
    else:
        # If not enough points on the left, use i, i+1, i+2 but note this is less accurate
        x_left = xs[i : i + 3]
        u_left = u[i : i + 3]

    # Fit quadratic on the right side (liquid phase)
    # Use points i+1, i+2, i+3 (ensuring we have enough points)
    if i + 3 < len(xs):
        x_right = xs[i + 1 : i + 4]
        u_right = u[i + 1 : i + 4]
    else:
        # If not enough points on the right, use the last 3 points
        x_right = xs[-3:]
        u_right = u[-3:]

    # Fit quadratic polynomials: u = ax² + bx + c
    # Using numpy's polyfit with degree 2
    poly_left = np.polyfit(x_left, u_left, 2)
    poly_right = np.polyfit(x_right, u_right, 2)

    # Find roots of each quadratic at u_pt
    # ax² + bx + c - u_pt = 0
    # Coefficients for left quadratic
    a_left, b_left, c_left = poly_left
    coeffs_left = [a_left, b_left, c_left - u_pt]

    # Coefficients for right quadratic
    a_right, b_right, c_right = poly_right
    coeffs_right = [a_right, b_right, c_right - u_pt]

    # Find roots
    roots_left = np.roots(coeffs_left)
    roots_right = np.roots(coeffs_right)

    # Select the most reasonable root from each side
    # For left side, choose root closest to the interface region
    real_roots_left = roots_left[np.isreal(roots_left)].real
    if len(real_roots_left) > 0:
        # Choose root closest to xs[i]
        s_left = real_roots_left[np.argmin(np.abs(real_roots_left - xs[i]))]
    else:
        # Fallback to linear interpolation on left
        s_left = xs[i - 1] + (u_pt - u[i - 1]) * h / (u[i] - u[i - 1])

    # For right side, choose root closest to the interface region
    real_roots_right = roots_right[np.isreal(roots_right)].real
    if len(real_roots_right) > 0:
        # Choose root closest to xs[i+1]
        s_right = real_roots_right[np.argmin(np.abs(real_roots_right - xs[i + 1]))]
    else:
        # Fallback to linear interpolation on right
        s_right = xs[i + 1] + (u_pt - u[i + 1]) * h / (u[i + 2] - u[i + 1])

    # Average the two estimates
    s_quad = 0.5 * (s_left + s_right)

    return s_quad


def robust_quadratic_interface(
    xs: np.ndarray, u: np.ndarray, u_pt: float, h: float
) -> float:
    """
    Robust quadratic fitting using more points and weighted least squares.
    Maintains high accuracy while reducing noise sensitivity.
    """
    thetas = u - u_pt
    i = np.where(thetas[:-1] * thetas[1:] < 0)[0][0]

    # Use 4-5 points on each side for robust fitting
    n_points = min(5, i + 1, len(xs) - i - 1)

    # Left side points - use more points for overdetermined system
    left_indices = range(max(0, i - n_points + 1), i + 1)
    x_left = xs[left_indices]
    u_left = u[left_indices]

    # Right side points
    right_indices = range(i + 1, min(len(xs), i + 2 + n_points))
    x_right = xs[right_indices]
    u_right = u[right_indices]

    # Weighted least squares - weight points closer to interface more heavily
    def fit_weighted_quadratic(x_data, u_data, ref_point):
        if len(x_data) < 3:
            # Fallback to linear if not enough points
            return np.polyfit(x_data, u_data, min(1, len(x_data) - 1))

        # Weights decrease with distance from interface
        weights = 1.0 / (1.0 + (x_data - ref_point) ** 2 / h**2)

        try:
            return np.polyfit(x_data, u_data, 2, w=weights)
        except np.linalg.LinAlgError:
            # Fallback to unweighted if weighted fails
            return np.polyfit(x_data, u_data, 2)

    # Fit quadratics
    try:
        poly_left = fit_weighted_quadratic(x_left, u_left, xs[i])
        poly_right = fit_weighted_quadratic(x_right, u_right, xs[i + 1])

        # Find interface positions
        def find_interface_position(poly, x_ref):
            if len(poly) == 3:  # Quadratic
                a, b, c = poly
                discriminant = b**2 - 4 * a * (c - u_pt)
                if discriminant >= 0:
                    roots = [
                        (-b + np.sqrt(discriminant)) / (2 * a),
                        (-b - np.sqrt(discriminant)) / (2 * a),
                    ]
                    # Choose root closest to the interface region
                    real_roots = [r for r in roots if np.isreal(r)]
                    if real_roots:
                        return min(real_roots, key=lambda r: abs(r - x_ref))
            else:  # Linear fallback
                b, c = poly
                if abs(b) > 1e-12:
                    return (u_pt - c) / b
            return x_ref

        s_left = find_interface_position(poly_left, xs[i])
        s_right = find_interface_position(poly_right, xs[i + 1])

        # Sanity check: make sure positions are reasonable
        if xs[i - 1] <= s_left <= xs[i + 1] and xs[i] <= s_right <= xs[i + 2]:
            return 0.5 * (s_left + s_right)

    except:
        pass

    # Fallback to your original quadratic method
    return quadratic_interface(xs, u, u_pt, h)


def filtered_quadratic_interface(
    xs: np.ndarray, u: np.ndarray, u_pt: float, h: float, filter_strength: float = 0.1
) -> float:
    """
    Apply minimal filtering to temperature data before quadratic fitting.
    Uses a very mild filter that preserves discontinuities while reducing high-frequency noise.
    """
    thetas = u - u_pt
    i = np.where(thetas[:-1] * thetas[1:] < 0)[0][0]

    # Create filtered version using a very conservative approach
    u_filtered = u.copy()

    # Only filter points that are clearly outliers (more than 2-3 standard deviations)
    # Left side filtering
    if i >= 3:
        left_data = u[max(0, i - 4) : i + 1]
        left_mean = np.mean(left_data)
        left_std = np.std(left_data)
        for j in range(max(0, i - 2), i + 1):
            if abs(u[j] - left_mean) > 2.5 * left_std:
                # Replace outlier with local median
                local_window = u[max(0, j - 1) : min(len(u), j + 2)]
                u_filtered[j] = (1 - filter_strength) * u[
                    j
                ] + filter_strength * np.median(local_window)

    # Right side filtering
    if i < len(u) - 4:
        right_data = u[i + 1 : min(len(u), i + 5)]
        right_mean = np.mean(right_data)
        right_std = np.std(right_data)
        for j in range(i + 1, min(len(u), i + 3)):
            if abs(u[j] - right_mean) > 2.5 * right_std:
                local_window = u[max(0, j - 1) : min(len(u), j + 2)]
                u_filtered[j] = (1 - filter_strength) * u[
                    j
                ] + filter_strength * np.median(local_window)

    # Apply quadratic method to filtered data
    return quadratic_interface(xs, u_filtered, u_pt, h)


def ensemble_quadratic_interface(
    xs: np.ndarray, u: np.ndarray, u_pt: float, h: float
) -> float:
    """
    Use ensemble of slightly different quadratic fits and average results.
    Reduces sensitivity to individual point choices.
    """
    thetas = u - u_pt
    i = np.where(thetas[:-1] * thetas[1:] < 0)[0][0]

    estimates = []

    # Try different point selections for fitting
    max_points = min(6, i + 1, len(xs) - i - 1)

    for n_pts in range(3, max_points + 1):
        for offset in range(max(0, min(2, n_pts - 3))):  # Small offset variations
            try:
                # Left side
                left_start = max(0, i - n_pts + 1 + offset)
                left_end = min(i + 1, left_start + n_pts)
                x_left = xs[left_start:left_end]
                u_left = u[left_start:left_end]

                # Right side
                right_start = max(i + 1, i + 1 - offset)
                right_end = min(len(xs), right_start + n_pts)
                x_right = xs[right_start:right_end]
                u_right = u[right_start:right_end]

                if len(x_left) >= 3 and len(x_right) >= 3:
                    # Fit quadratics
                    poly_left = np.polyfit(x_left, u_left, 2)
                    poly_right = np.polyfit(x_right, u_right, 2)

                    # Find roots
                    a_left, b_left, c_left = poly_left
                    a_right, b_right, c_right = poly_right

                    roots_left = np.roots([a_left, b_left, c_left - u_pt])
                    roots_right = np.roots([a_right, b_right, c_right - u_pt])

                    # Select best roots
                    real_roots_left = roots_left[np.isreal(roots_left)].real
                    real_roots_right = roots_right[np.isreal(roots_right)].real

                    if len(real_roots_left) > 0 and len(real_roots_right) > 0:
                        s_left = real_roots_left[
                            np.argmin(np.abs(real_roots_left - xs[i]))
                        ]
                        s_right = real_roots_right[
                            np.argmin(np.abs(real_roots_right - xs[i + 1]))
                        ]

                        # Sanity check
                        if (
                            xs[max(0, i - 1)] <= s_left <= xs[min(len(xs) - 1, i + 1)]
                            and xs[max(0, i)] <= s_right <= xs[min(len(xs) - 1, i + 2)]
                        ):
                            estimates.append(0.5 * (s_left + s_right))
            except:
                continue

    if estimates:
        # Return median of all estimates (robust to outliers)
        return np.median(estimates)
    else:
        # Fallback to original method
        return quadratic_interface(xs, u, u_pt, h)


n_values = list(range(50, 800, 5))
h_values = []
errors_lin = []
errors_pw = []
errors_quad = []
errors_robust = []
errors_filtered = []
errors_ensemble = []

print("Running grid refinement test...")
for n_y in n_values:
    xs = np.linspace(0, geometry.height, n_y)
    h = xs[1] - xs[0]
    h_values.append(h)

    # Analytical temperatures
    Ts = []
    for y in xs:
        if y <= s:
            Ts.append(
                get_ice_temp(
                    y=y,
                    gamma=gamma,
                    t=t,
                    min_temp=min_temp + ABS_ZERO,
                    material_props=material_props,
                )
            )
        else:
            Ts.append(
                get_water_temp(
                    y=y,
                    gamma=gamma,
                    t=t,
                    max_temp=max_temp + ABS_ZERO,
                    material_props=material_props,
                )
            )
    Ts = np.array(Ts)

    # Reconstruct interface
    s_lin = lin_interp_interface(xs, Ts, u_star, h)
    s_pw = piecewise_interface(xs, Ts, u_star, h)
    s_quad = quadratic_interface(xs, Ts, u_star, h)
    s_robust = robust_quadratic_interface(xs, Ts, u_star, h)
    s_filtered = filtered_quadratic_interface(xs, Ts, u_star, h)
    s_ensemble = ensemble_quadratic_interface(xs, Ts, u_star, h)

    errors_lin.append(abs(s_lin - s))
    errors_pw.append(abs(s_pw - s))
    errors_quad.append(abs(s_quad - s))
    errors_robust.append(abs(s_robust - s))
    errors_filtered.append(abs(s_filtered - s))
    errors_ensemble.append(abs(s_ensemble - s))

print(f"Generated {len(h_values)} data points")

plt.figure(figsize=(6, 5))
plt.plot(h_values, errors_lin, "o-", label="Linear interpolation")
plt.plot(h_values, errors_pw, "s-", label="Piecewise one-sided")
plt.plot(h_values, errors_quad, "^-", label="Quadratic reconstruction", markersize=4)
plt.plot(h_values, errors_robust, "^-", label="Robust reconstruction", markersize=4)
plt.plot(h_values, errors_ensemble, "^-", label="Ensemble reconstruction", markersize=4)

plt.xlabel("Grid spacing h")
plt.ylabel("Error in interface position")
plt.title("Grid refinement test")
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which="both", alpha=0.3)
plt.show()
