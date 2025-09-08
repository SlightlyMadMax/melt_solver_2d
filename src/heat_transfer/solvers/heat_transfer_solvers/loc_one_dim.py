import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.heat_transfer.solvers.heat_transfer_solvers.base import (
    ADIHeatSolver,
)
from src.heat_transfer.solvers.heat_transfer_solvers.registry import (
    HeatTransferSolverName,
    register_solver,
)


@register_solver(HeatTransferSolverName.LOC_ONE_DIM)
class LocOneDimSolver(ADIHeatSolver):
    def _compute_sweep_x_coeffs(
        self, state: np.ndarray, dt: float, dx: float, dy: float
    ) -> None:
        self._compute_sweep_x_coeffs_jit(
            u=state,
            conv_x=self._conv_x,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dx=dx,
            dt=dt,
            peclet_number=self.cfg.peclet_number,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
        )

    def _compute_sweep_y_coeffs(
        self, state: np.ndarray, dt: float, dx: float, dy: float
    ) -> None:
        self._compute_sweep_y_coeffs_jit(
            u=self._new_u,
            conv_y=self._conv_y,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dy=dy,
            dt=dt,
            peclet_number=self.cfg.peclet_number,
            a_y=self._a_y,
            b_y=self._b_y,
            c_y=self._c_y,
            rhs=self._rhs_y,
        )

    @staticmethod
    @njit
    def _compute_sweep_x_coeffs_jit(
        u: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        dx: float,
        dt: float,
        peclet_number: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u.shape
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        inv_peclet_number = 1.0 / peclet_number

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_ip1j = 0.5 * (k_eff[j, i] + k_eff[j, i + 1])
                k_im1j = 0.5 * (k_eff[j, i] + k_eff[j, i - 1])

                # Coefficient at T_{i + 1, j}^{n + 1/2}
                a[j, i] = dt * (
                    conv_x[j, i, 0] - k_ip1j * inv_peclet_number * inv_c_eff * inv_dx2
                )

                # Coefficient at T_{i, j}^{n + 1/2}
                b[j, i] = 1.0 + dt * (
                    conv_x[j, i, 1]
                    + (k_ip1j + k_im1j) * inv_peclet_number * inv_c_eff * inv_dx2
                )

                # Coefficient at T_{i - 1, j}^{n + 1/2}
                c[j, i] = dt * (
                    conv_x[j, i, 2] - k_im1j * inv_peclet_number * inv_c_eff * inv_dx2
                )

                rhs[j, i] = u[j, i]

    @staticmethod
    @njit
    def _compute_sweep_y_coeffs_jit(
        u: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        dy: float,
        dt: float,
        peclet_number: float,
        a_y: NDArray[np.float64],
        b_y: NDArray[np.float64],
        c_y: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u.shape
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy
        inv_peclet_number = 1.0 / peclet_number

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_ijp1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                # Coefficient at T_{i, j + 1}^{n + 1}
                a_y[i, j] = dt * (
                    conv_y[j, i, 0] - k_ijp1 * inv_peclet_number * inv_c_eff * inv_dy2
                )

                # Coefficient at T_{i, j}^{n + 1}
                b_y[i, j] = 1.0 + dt * (
                    conv_y[j, i, 1]
                    + (k_ijp1 + k_ijm1) * inv_peclet_number * inv_c_eff * inv_dy2
                )

                # Coefficient at T_{i, j - 1}^{n + 1}
                c_y[i, j] = dt * (
                    conv_y[j, i, 2] - k_ijm1 * inv_peclet_number * inv_c_eff * inv_dy2
                )

                rhs[i, j] = u[j, i]
