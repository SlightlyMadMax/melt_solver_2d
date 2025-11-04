import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import (
    ADIHeatSolver,
)
from src.heat_transfer.solvers.heat_transfer_solvers.registry import (
    HeatTransferSolverName,
    register_solver,
)


@register_solver(HeatTransferSolverName.LOC_ONE_DIM)
class LocOneDimSolver(ADIHeatSolver):
    def _compute_sweep_x_coeffs(self, u: np.ndarray) -> None:
        dx, dy, dt = self.cfg.scaled_grid_steps
        self._compute_sweep_x_coeffs_jit(
            u=u,
            conv_x=self._conv_x,
            corr_x=self._correction_x,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dx=dx,
            dt=dt,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
        )

    def _compute_sweep_y_coeffs(self, u: np.ndarray) -> None:
        dx, dy, dt = self.cfg.scaled_grid_steps
        self._compute_sweep_y_coeffs_jit(
            u=self._new_u,
            conv_y=self._conv_y,
            corr_y=self._correction_y,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dy=dy,
            dt=dt,
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
        )

    @staticmethod
    @njit
    def _compute_sweep_x_coeffs_jit(
        u: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        corr_x: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        dx: float,
        dt: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u.shape
        inv_dx2 = 1.0 / (dx * dx)

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_ip1j = 0.5 * (k_eff[j, i] + k_eff[j, i + 1])
                k_im1j = 0.5 * (k_eff[j, i] + k_eff[j, i - 1])

                # Coefficient at T_{i + 1, j}^{n + 1/2}
                a[j, i] = dt * (conv_x[j, i, 0] - k_ip1j * inv_c_eff * inv_dx2)

                # Coefficient at T_{i, j}^{n + 1/2}
                b[j, i] = 1.0 + dt * (
                    conv_x[j, i, 1] + (k_ip1j + k_im1j) * inv_c_eff * inv_dx2
                )

                # Coefficient at T_{i - 1, j}^{n + 1/2}
                c[j, i] = dt * (conv_x[j, i, 2] - k_im1j * inv_c_eff * inv_dx2)

                rhs[j, i] = u[j, i] - dt * corr_x[j, i]

    @staticmethod
    @njit
    def _compute_sweep_y_coeffs_jit(
        u: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        corr_y: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        dy: float,
        dt: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u.shape
        inv_dy2 = 1.0 / (dy * dy)

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_ijp1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                # Coefficient at T_{i, j + 1}^{n + 1}
                a[i, j] = dt * (conv_y[j, i, 0] - k_ijp1 * inv_c_eff * inv_dy2)

                # Coefficient at T_{i, j}^{n + 1}
                b[i, j] = 1.0 + dt * (
                    conv_y[j, i, 1] + (k_ijp1 + k_ijm1) * inv_c_eff * inv_dy2
                )

                # Coefficient at T_{i, j - 1}^{n + 1}
                c[i, j] = dt * (conv_y[j, i, 2] - k_ijm1 * inv_c_eff * inv_dy2)

                rhs[i, j] = u[j, i] - dt * corr_y[j, i]
