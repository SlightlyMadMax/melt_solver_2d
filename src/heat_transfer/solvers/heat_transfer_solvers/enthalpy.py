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


@register_solver(HeatTransferSolverName.ENTHALPY)
class EnthalpySolver(ADIHeatSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        self._liquid_frac_old: NDArray[np.float64] = -np.ones((n_y, n_x))
        self._liquid_frac_new: NDArray[np.float64] = np.empty((n_y, n_x))

    def _init_liquid_fraction(self):
        u_pt = self.cfg.u_pt_nd
        delta = self.cfg.delta_nd
        diff_u = self._iter_u - u_pt
        self._liquid_frac_old[:, :] = 0.5 * (1.0 + np.tanh(diff_u / delta))

    def _recalculate_liquid_fraction(self):
        ste = self.cfg.stefan_number
        delta = self.cfg.delta_nd
        u_pt = self.cfg.u_pt_nd
        diff_u = self._iter_u - u_pt
        # self._liquid_frac_new[:, :] = 0.5 * (1.0 + np.tanh(diff_u / delta))
        self._liquid_frac_new[:, :] = (
            ste * (self._iter_u - delta) + self._liquid_frac_old
        ) / (1.0 + 2.0 * delta * ste)

    def _compute_sweep_x_coeffs(self, u: np.ndarray) -> None:
        dx, dy, dt = self.cfg.scaled_grid_steps
        if np.max(self._liquid_frac_old) <= 0.0:
            print("bruh")
            self._init_liquid_fraction()

        self._recalculate_liquid_fraction()
        self._compute_sweep_x_coeffs_jit(
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dx=dx,
            dy=dy,
            dt=dt,
            stefan_number=self.cfg.stefan_number,
            l_f_n=self._liquid_frac_old,
            l_f_np1=self._liquid_frac_new,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
        )

    def _compute_sweep_y_coeffs(self, u: np.ndarray) -> None:
        dx, dy, dt = self.cfg.scaled_grid_steps
        self._compute_sweep_y_coeffs_jit(
            u=self._new_u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dx=dx,
            dy=dy,
            dt=dt,
            stefan_number=self.cfg.stefan_number,
            l_f_n=self._liquid_frac_old,
            l_f_np1=self._liquid_frac_new,
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
        stefan_number: float,
        l_f_n: NDArray[np.float64],
        l_f_np1: NDArray[np.float64],
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u.shape
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        inv_ste = 1.0 / stefan_number
        dt_half = 0.5 * dt

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_ip1j = 0.5 * (k_eff[j, i] + k_eff[j, i + 1])
                k_im1j = 0.5 * (k_eff[j, i] + k_eff[j, i - 1])
                k_ijp1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                # Coefficient at T_{i + 1, j}^{n + 1/2}
                a[j, i] = dt_half * (conv_x[j, i, 0] - k_ip1j * inv_c_eff * inv_dx2)

                # Coefficient at T_{i, j}^{n + 1/2}
                b[j, i] = 1.0 + dt_half * (
                    conv_x[j, i, 1] + (k_ip1j + k_im1j) * inv_c_eff * inv_dx2
                )

                # Coefficient at T_{i - 1, j}^{n + 1/2}
                c[j, i] = dt_half * (conv_x[j, i, 2] - k_im1j * inv_c_eff * inv_dx2)

                # Right-hand side of the equation
                rhs[j, i] = u[j, i] + dt_half * (
                    inv_dy2
                    * inv_c_eff
                    * (
                        k_ijp1 * (u[j + 1, i] - u[j, i])
                        - k_ijm1 * (u[j, i] - u[j - 1, i])
                    )
                    - (
                        conv_y[j, i, 0] * u[j + 1, i]
                        + conv_y[j, i, 1] * u[j, i]
                        + conv_y[j, i, 2] * u[j - 1, i]
                    )
                    - inv_ste * inv_c_eff * (l_f_np1[j, i] - l_f_n[j, i]) / dt
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
        stefan_number: float,
        l_f_n: NDArray[np.float64],
        l_f_np1: NDArray[np.float64],
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u.shape
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        inv_ste = 1.0 / stefan_number
        dt_half = 0.5 * dt

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_ip1j = 0.5 * (k_eff[j, i] + k_eff[j, i + 1])
                k_im1j = 0.5 * (k_eff[j, i] + k_eff[j, i - 1])
                k_ijp1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                # Coefficient at T_{i, j + 1}^{n + 1}
                a[i, j] = dt_half * (conv_y[j, i, 0] - k_ijp1 * inv_c_eff * inv_dy2)

                # Coefficient at T_{i, j}^{n + 1}
                b[i, j] = 1.0 + dt_half * (
                    conv_y[j, i, 1] + (k_ijp1 + k_ijm1) * inv_c_eff * inv_dy2
                )

                # Coefficient at T_{i, j - 1}^{n + 1}
                c[i, j] = dt_half * (conv_y[j, i, 2] - k_ijm1 * inv_c_eff * inv_dy2)

                # Right-hand side of the equation
                rhs[i, j] = u[j, i] + dt_half * (
                    inv_dx2
                    * inv_c_eff
                    * (
                        k_ip1j * (u[j, i + 1] - u[j, i])
                        - k_im1j * (u[j, i] - u[j, i - 1])
                    )
                    - (
                        conv_x[j, i, 0] * u[j, i + 1]
                        + conv_x[j, i, 1] * u[j, i]
                        + conv_x[j, i, 2] * u[j, i - 1]
                    )
                    - inv_ste * inv_c_eff * (l_f_np1[j, i] - l_f_n[j, i]) / dt
                )
