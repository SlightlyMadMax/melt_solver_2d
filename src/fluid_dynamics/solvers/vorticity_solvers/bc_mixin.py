import numpy as np
from numba import njit
from numpy.typing import NDArray


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
        homogeneous: bool = False,
    ) -> None:
        n_y, n_x = sf.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy
        inv_dx2 = inv_dx * inv_dx
        inv_dy2 = inv_dy * inv_dy

        if homogeneous:
            top_bc[:] = 0.0
            right_bc[:] = 0.0
            bottom_bc[:] = 0.0
            left_bc[:] = 0.0
            return

        if order == 1:
            top_bc[:] = -2.0 * inv_dy2 * sf[n_y - 2, :]
            right_bc[:] = -2.0 * inv_dx2 * sf[:, n_x - 2]
            bottom_bc[:] = -2.0 * inv_dy2 * sf[1, :]
            left_bc[:] = -2.0 * inv_dx2 * sf[:, 1]
        elif order == 2:
            top_bc[:] = 0.5 * inv_dy2 * (sf[n_y - 3, :] - 8.0 * sf[n_y - 2, :])
            right_bc[:] = 0.5 * inv_dx2 * (sf[:, n_x - 3] - 8.0 * sf[:, n_x - 2])
            bottom_bc[:] = 0.5 * inv_dy2 * (sf[2, :] - 8.0 * sf[1, :])
            left_bc[:] = 0.5 * inv_dx2 * (sf[:, 2] - 8.0 * sf[:, 1])
        else:
            raise NotImplementedError
