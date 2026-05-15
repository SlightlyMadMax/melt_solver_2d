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


@register_solver(HeatTransferSolverName.DOUGLAS_RACHFORD)
class DouglasRachfordSolver(ADIHeatSolver):
    def _compute_sweep_x_coeffs(self, u: np.ndarray) -> None:
        dx, dy, dt = self.cfg.scaled_grid_steps
        self._compute_sweep_x_coeffs_jit(
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            corr_x=self._correction_x,
            corr_y=self._correction_y,
            c_eff=self._c_eff,
            k_x=self._k_x,
            k_y=self._k_y,
            dx=dx,
            dy=dy,
            dt=dt,
            pe=self.cfg.peclet_number,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
        )

    def _compute_sweep_y_coeffs(self, u: np.ndarray) -> None:
        dx, dy, dt = self.cfg.scaled_grid_steps
        self._compute_sweep_y_coeffs_jit(
            u_old=u,
            u_prev=self._u_new,
            conv_y=self._conv_y,
            c_eff=self._c_eff,
            k_y=self._k_y,
            dy=dy,
            dt=dt,
            pe=self.cfg.peclet_number,
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
        conv_y: NDArray[np.float64],
        corr_x: NDArray[np.float64],
        corr_y: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        k_x: NDArray[np.float64],
        k_y: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        pe: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u.shape
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        inv_pe = 1.0 / pe

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]

                # Coefficient at T_{i + 1, j}^{n + 1/2}
                a[j, i] = dt * (
                    conv_x[j, i, 0] - k_x[j, i + 1] * inv_pe * inv_c_eff * inv_dx2
                )

                # Coefficient at T_{i, j}^{n + 1/2}
                b[j, i] = 1.0 + dt * (
                    conv_x[j, i, 1]
                    + (k_x[j, i + 1] + k_x[j, i]) * inv_pe * inv_c_eff * inv_dx2
                )

                # Coefficient at T_{i - 1, j}^{n + 1/2}
                c[j, i] = dt * (
                    conv_x[j, i, 2] - k_x[j, i] * inv_pe * inv_c_eff * inv_dx2
                )

                rhs[j, i] = u[j, i] + dt * (
                    inv_dy2
                    * inv_pe
                    * inv_c_eff
                    * (
                        k_y[j + 1, i] * (u[j + 1, i] - u[j, i])
                        - k_y[j, i] * (u[j, i] - u[j - 1, i])
                    )
                    - (
                        conv_y[j, i, 0] * u[j + 1, i]
                        + conv_y[j, i, 1] * u[j, i]
                        + conv_y[j, i, 2] * u[j - 1, i]
                    )
                    - corr_x[j, i]
                    - corr_y[j, i]
                )

    @staticmethod
    @njit
    def _compute_sweep_y_coeffs_jit(
        u_old: NDArray[np.float64],
        u_prev: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        k_y: NDArray[np.float64],
        dy: float,
        dt: float,
        pe: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u_old.shape
        inv_dy2 = 1.0 / (dy * dy)
        inv_pe = 1.0 / pe

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]

                # Coefficient at T_{i, j + 1}^{n + 1}
                a[i, j] = dt * (
                    conv_y[j, i, 0] - k_y[j + 1, i] * inv_pe * inv_c_eff * inv_dy2
                )

                # Coefficient at T_{i, j}^{n + 1}
                b[i, j] = 1.0 + dt * (
                    conv_y[j, i, 1]
                    + (k_y[j + 1, i] + k_y[j, i]) * inv_pe * inv_c_eff * inv_dy2
                )

                # Coefficient at T_{i, j - 1}^{n + 1}
                c[i, j] = dt * (
                    conv_y[j, i, 2] - k_y[j, i] * inv_pe * inv_c_eff * inv_dy2
                )

                # Right-hand side of the equation
                rhs[i, j] = u_prev[j, i] - dt * (
                    inv_dy2
                    * inv_c_eff
                    * inv_pe
                    * (
                        k_y[j + 1, i] * (u_old[j + 1, i] - u_old[j, i])
                        - k_y[j, i] * (u_old[j, i] - u_old[j - 1, i])
                    )
                    - (
                        conv_y[j, i, 0] * u_old[j + 1, i]
                        + conv_y[j, i, 1] * u_old[j, i]
                        + conv_y[j, i, 2] * u_old[j - 1, i]
                    )
                )
