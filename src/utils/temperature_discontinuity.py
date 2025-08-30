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


# geometry = DomainGeometry(
#     width=1.0,
#     height=1.0,
#     end_time=60.0 * 60.0 * 24.0,
#     n_x=5,
#     n_y=301,
#     n_t=60 * 24,
# )
#
# material_props = MaterialProperties(
#     u_pt=273.15,
#     specific_heat_liquid=4120.7,
#     specific_heat_solid=2056.8,
#     specific_latent_heat=3.33e5,
#     density_liquid=999.84,
#     density_solid=918.9,
#     thermal_conductivity_liquid=0.59,
#     thermal_conductivity_solid=2.21,
#     kinematic_viscosity_coeffs=[
#         0.000108963453,
#         -9.28722151e-07,
#         2.65889022e-09,
#         -2.54761652e-12,
#     ],
#     volumetric_thermal_exp_coeffs=[7.68e-6],
# )
#
# max_temp = 278.15
# min_temp = 268.15
#
# cfg = ExperimentConfig(
#     geometry=geometry,
#     material_props=material_props,
#     u_ref=0.5 * (min_temp + max_temp),
#     delta_u=0.5 * (max_temp - min_temp),
#     v=0.01,
#     l=geometry.max_dimension,
#     u_solid=273.0,
#     u_liquid=273.3,
#     epsilon=1e-6,
#     save_interval=1,
# )
#
# gamma = calculate_gamma(cfg=cfg, min_temp=min_temp, max_temp=max_temp)
# t = geometry.end_time
# s = t**0.5 * gamma  # interface position
#
# print(f"True interface position: s = {s:.6f}")
# u_star = material_props.u_pt + ABS_ZERO


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


def piecewise_interface_v2(
    xs: np.ndarray, u: np.ndarray, u_pt: float, h: float
) -> float:
    thetas = u - u_pt
    i = np.where(thetas[:-1] * thetas[1:] < 0)[0][0]

    m_left = (u[i] - u[i - 1]) / h
    s_left_piece = xs[i] + (u_pt - u[i]) / m_left

    m_right = (u[i + 2] - u[i + 1]) / h
    s_right_piece = xs[i + 1] + (u_pt - u[i + 1]) / m_right

    if abs(m_left) > abs(m_right):
        return s_left_piece

    return s_right_piece


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

    der_left = 0.5 * (3 * u[i] - 4 * u[i - 1] + u[i - 2]) / h
    der_right = 0.5 * (-3 * u[i + 1] + 4 * u[i + 2] - u[i + 3]) / h

    real_roots_left = roots_left[np.isreal(roots_left)].real
    real_roots_right = roots_right[np.isreal(roots_right)].real

    if abs(der_left) > abs(der_right):
        if len(real_roots_left) > 0:
            return real_roots_left[np.argmin(np.abs(real_roots_left - xs[i]))]
        else:
            return xs[i - 1] + (u_pt - u[i - 1]) * h / (u[i] - u[i - 1])

    if len(real_roots_right) > 0:
        return real_roots_right[np.argmin(np.abs(real_roots_right - xs[i + 1]))]

    return xs[i + 1] + (u_pt - u[i + 1]) * h / (u[i + 2] - u[i + 1])


# n_values = list(range(50, 800, 5))
# h_values = []
# errors_lin = []
# errors_pw = []
# errors_quad = []
# errors_pw_v2 = []
#
# print("Running grid refinement test...")
# for n_y in n_values:
#     xs = np.linspace(0, geometry.height, n_y)
#     h = xs[1] - xs[0]
#     h_values.append(h)
#
#     # Analytical temperatures
#     Ts = []
#     for y in xs:
#         if y <= s:
#             Ts.append(
#                 get_ice_temp(
#                     y=y,
#                     gamma=gamma,
#                     t=t,
#                     min_temp=min_temp + ABS_ZERO,
#                     material_props=material_props,
#                 )
#             )
#         else:
#             Ts.append(
#                 get_water_temp(
#                     y=y,
#                     gamma=gamma,
#                     t=t,
#                     max_temp=max_temp + ABS_ZERO,
#                     material_props=material_props,
#                 )
#             )
#     Ts = np.array(Ts)
#
#     # Reconstruct interface
#     s_lin = lin_interp_interface(xs, Ts, u_star, h)
#     s_pw = piecewise_interface(xs, Ts, u_star, h)
#     s_quad = quadratic_interface(xs, Ts, u_star, h)
#     s_pw_v2 = piecewise_interface_v2(xs, Ts, u_star, h)
#
#     errors_lin.append(abs(s_lin - s))
#     errors_pw.append(abs(s_pw - s))
#     errors_quad.append(abs(s_quad - s))
#     errors_pw_v2.append(abs(s_pw_v2 - s))
#
# print(f"Generated {len(h_values)} data points")
#
# plt.figure(figsize=(6, 5))
# # plt.plot(h_values, errors_lin, "o-", label="Linear interpolation")
# plt.plot(h_values, errors_pw, "s-", label="Piecewise one-sided")
# plt.plot(h_values, errors_quad, "^-", label="Quadratic reconstruction", markersize=4)
# plt.plot(h_values, errors_pw_v2, "^-", label="Piecewise one-sided v2", markersize=4)
#
# plt.xlabel("Grid spacing h")
# plt.ylabel("Error in interface position")
# plt.title("Grid refinement test")
# plt.legend()
# plt.yscale("log")
# plt.xscale("log")
# plt.grid(True, which="both", alpha=0.3)
# plt.show()
