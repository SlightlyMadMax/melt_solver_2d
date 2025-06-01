import numpy as np
from numba import njit

from scipy.special import erf
from numpy.typing import NDArray


def calculate_indicator_function(
    u: NDArray[np.float64],
    u_pt: float,
    eps: float,
    result: NDArray[np.float64],
    # delta: NDArray[np.float64],
    delta: float,
) -> None:
    """
    Indicator function for the fictitious domain method.
    Is equal to 0 for liquid phase and 1 / eps^2 for solid phase.

    :param u: The dimensional temperature value.
    :param u_pt: The phase transition temperature.
    :param eps: A small parameter.
    :param result: A ndarray for storing the calculated indicator function values.
    :return: None.
    """
    result[:, :] = 0.0

    inv_eps2 = 1.0 / (eps * eps)
    interior = (slice(1, -1), slice(1, -1))

    diff_u = u[interior] - u_pt
    # delta_inner = delta[interior]

    # --- Variant 1: sharp step ----------------------
    # mask = (diff_u < 0.0)
    # result_interior = np.zeros_like(diff_u)
    # result_interior[mask] = inv_eps2
    # result[interior] = result_interior

    # --- Variant 2: error‐function form -------------------
    result[interior] = 0.5 * inv_eps2 * (1.0 - erf(diff_u / (np.sqrt(2.0) * delta)))

    # --- Variant 3: hyperbolic‐tangent form ---------------
    # result[interior] = (
    #     0.5
    #     * inv_eps2
    #     * (
    #         1.0
    #         - np.tanh(
    #             3.0 * diff_u / np.sqrt(delta_inner * delta_inner - diff_u * diff_u)
    #         )
    #     )
    # )

    # --- Variant 4: exponential form (one-sided smoothing) ----------------------
    # exp_term = np.exp(-diff_u / delta_inner)
    # result[interior] = inv_eps2 * 0.5 * (2.0 + exp_term / (0.5 - exp_term))


@njit
def calculate_vorticity_from_sf(
    sf: NDArray[np.float64],
    result: NDArray[np.float64],
    dx: float,
    dy: float,
):
    n_y, n_x = sf.shape
    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2

    result[-1, :] = -2.0 * inv_dy2 * sf[-2, :]
    result[:, -1] = -2.0 * inv_dx2 * sf[:, -2]
    result[0, :] = -2.0 * inv_dy2 * sf[1, :]
    result[:, 0] = -2.0 * inv_dx2 * sf[:, 1]

    for j in range(1, n_y - 1):
        for i in range(1, n_x - 1):
            result[j, i] = (
                -(sf[j + 1, i] - 2 * sf[j, i] + sf[j - 1, i]) * inv_dy2
                - (sf[j, i + 1] - 2 * sf[j, i] + sf[j, i - 1]) * inv_dx2
            )


@njit
def calculate_velocity_from_sf(
    sf: NDArray[np.float64],
    v_x: NDArray[np.float64],
    v_y: NDArray[np.float64],
    dx: float,
    dy: float,
):
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)

    v_x[:, :] = 0.0
    v_y[:, :] = 0.0

    v_x[1:-1, 1:-1] = inv_2dy * (sf[2:, 1:-1] - sf[:-2, 1:-1])
    v_y[1:-1, 1:-1] = -inv_2dx * (sf[1:-1, 2:] - sf[1:-1, :-2])


class VorticityBCMixin:
    @staticmethod
    def calculate_boundary_conditions(
        sf: NDArray[np.float64],
        top_bc: NDArray[np.float64],
        right_bc: NDArray[np.float64],
        bottom_bc: NDArray[np.float64],
        left_bc: NDArray[np.float64],
        order: int,
        dx: float,
        dy: float,
    ) -> None:
        dx2_2_inv = 2.0 / dx**2
        dy2_2_inv = 2.0 / dy**2
        inv_2dx2 = 1.0 / (2.0 * dx**2)
        inv_2dy2 = 1.0 / (2.0 * dy**2)

        if order == 1:
            top_bc[:] = -dy2_2_inv * sf[-2, :]
            right_bc[:] = -dx2_2_inv * sf[:, -2]
            bottom_bc[:] = -dy2_2_inv * sf[1, :]
            left_bc[:] = -dx2_2_inv * sf[:, 1]
        elif order == 2:
            top_bc[:] = inv_2dy2 * (sf[-3, :] - 8.0 * sf[-2, :])
            right_bc[:] = inv_2dx2 * (sf[:, -3] - 8.0 * sf[:, -2])
            bottom_bc[:] = inv_2dy2 * (sf[2, :] - 8.0 * sf[1, :])
            left_bc[:] = inv_2dx2 * (sf[:, 2] - 8.0 * sf[:, 1])
        else:
            raise NotImplementedError
