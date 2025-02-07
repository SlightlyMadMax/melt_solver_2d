import numpy as np
from numba import njit
from numpy._typing import NDArray
from numpy.typing import NDArray


@njit
def calculate_indicator_function(
    u: NDArray[np.float64],
    u_pt: float,
    eps: float,
    result: NDArray[np.float64],
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
    n_y, n_x = u.shape

    result[:, :] = 0.0

    for j in range(1, n_y - 1):
        for i in range(1, n_x - 1):
            if u[j, i] - u_pt < 0.0:
                result[j, i] = 1.0 / (eps * eps)


@njit
def calculate_vorticity_from_sf(
    sf: NDArray[np.float64], result: NDArray[np.float64], dx: float, dy: float
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
            if j == 1 or j == n_y - 2:
                d2_psi_dy2 = (sf[j + 1, i] - 2 * sf[j, i]) * inv_dy2
            else:
                d2_psi_dy2 = (sf[j + 1, i] - 2 * sf[j, i] + sf[j - 1, i]) * inv_dy2

            if i == 1 or i == n_x - 2:
                d2_psi_dx2 = (sf[j, i + 1] - 2 * sf[j, i]) * inv_dx2
            else:
                d2_psi_dx2 = (sf[j, i + 1] - 2 * sf[j, i] + sf[j, i - 1]) * inv_dx2

            result[j, i] = -(d2_psi_dy2 + d2_psi_dx2)


class VorticityBCMixin:
    @staticmethod
    @njit
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
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy
        inv_dx2 = inv_dx * inv_dx
        inv_dy2 = inv_dy * inv_dy

        if order == 1:
            top_bc[:] = -2.0 * inv_dy2 * sf[-2, :]
            right_bc[:] = -2.0 * inv_dx2 * sf[:, -2]
            bottom_bc[:] = -2.0 * inv_dy2 * sf[1, :]
            left_bc[:] = -2.0 * inv_dx2 * sf[:, 1]
        elif order == 2:
            top_bc[:] = 0.5 * inv_dy2 * (sf[-3, :] - 8.0 * sf[-2, :])
            right_bc[:] = 0.5 * inv_dx2 * (sf[:, -3] - 8.0 * sf[:, -2])
            bottom_bc[:] = 0.5 * inv_dy2 * (sf[2, :] - 8.0 * sf[1, :])
            left_bc[:] = 0.5 * inv_dx2 * (sf[:, 2] - 8.0 * sf[:, 1])
        else:
            raise NotImplementedError
