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


@register_solver(HeatTransferSolverName.PEACEMAN_RACHFORD)
class PeacemanRachfordSolver(ADIHeatSolver):
    def _compute_sweep_x_coeffs(
        self, u: np.ndarray, dt: float, dx: float, dy: float
    ) -> None:
        self._compute_sweep_x_coeffs_jit(
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dx=dx,
            dy=dy,
            dt=dt,
            peclet_number=self.cfg.peclet_number,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
        )

    def _compute_sweep_y_coeffs(
        self, u: np.ndarray, dt: float, dx: float, dy: float
    ) -> None:
        self._compute_sweep_y_coeffs_jit(
            u=self._new_u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dx=dx,
            dy=dy,
            dt=dt,
            peclet_number=self.cfg.peclet_number,
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
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        dx: float,
        dy: float,
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
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy
        inv_peclet_number = 1.0 / peclet_number
        dt_half = 0.5 * dt

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_ip1j = 0.5 * (k_eff[j, i] + k_eff[j, i + 1])
                k_im1j = 0.5 * (k_eff[j, i] + k_eff[j, i - 1])
                k_ijp1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                # Coefficient at T_{i + 1, j}^{n + 1/2}
                a[j, i] = dt_half * (
                    conv_x[j, i, 0] - k_ip1j * inv_peclet_number * inv_c_eff * inv_dx2
                )

                # Coefficient at T_{i, j}^{n + 1/2}
                b[j, i] = 1.0 + dt_half * (
                    conv_x[j, i, 1]
                    + (k_ip1j + k_im1j) * inv_peclet_number * inv_c_eff * inv_dx2
                )

                # Coefficient at T_{i - 1, j}^{n + 1/2}
                c[j, i] = dt_half * (
                    conv_x[j, i, 2] - k_im1j * inv_peclet_number * inv_c_eff * inv_dx2
                )

                # Right-hand side of the equation
                rhs[j, i] = u[j, i] + dt_half * (
                    inv_dy2
                    * inv_c_eff
                    * inv_peclet_number
                    * (
                        k_ijp1 * (u[j + 1, i] - u[j, i])
                        - k_ijm1 * (u[j, i] - u[j - 1, i])
                    )
                    - (
                        conv_y[j, i, 0] * u[j + 1, i]
                        + conv_y[j, i, 1] * u[j, i]
                        + conv_y[j, i, 2] * u[j - 1, i]
                    )
                )

    @staticmethod
    @njit
    def _compute_sweep_y_coeffs_jit(
        u: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        dx: float,
        dy: float,
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
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy
        inv_peclet_number = 1.0 / peclet_number
        dt_half = 0.5 * dt

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_i1pj = 0.5 * (k_eff[j, i] + k_eff[j, i + 1])
                k_im1j = 0.5 * (k_eff[j, i] + k_eff[j, i - 1])
                k_ijp1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                # Coefficient at T_{i, j + 1}^{n + 1}
                a[i, j] = dt_half * (
                    conv_y[j, i, 0] - k_ijp1 * inv_peclet_number * inv_c_eff * inv_dy2
                )

                # Coefficient at T_{i, j}^{n + 1}
                b[i, j] = 1.0 + dt_half * (
                    conv_y[j, i, 1]
                    + (k_ijp1 + k_ijm1) * inv_peclet_number * inv_c_eff * inv_dy2
                )

                # Coefficient at T_{i, j - 1}^{n + 1}
                c[i, j] = dt_half * (
                    conv_y[j, i, 2] - k_ijm1 * inv_peclet_number * inv_c_eff * inv_dy2
                )

                # Right-hand side of the equation
                rhs[i, j] = u[j, i] + dt_half * (
                    inv_dx2
                    * inv_c_eff
                    * inv_peclet_number
                    * (
                        k_i1pj * (u[j, i + 1] - u[j, i])
                        - k_im1j * (u[j, i] - u[j, i - 1])
                    )
                    - (
                        conv_x[j, i, 0] * u[j, i + 1]
                        + conv_x[j, i, 1] * u[j, i]
                        + conv_x[j, i, 2] * u[j, i - 1]
                    )
                )
