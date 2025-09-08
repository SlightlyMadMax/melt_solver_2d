import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import (
    ADIVorticitySolver,
)
from src.fluid_dynamics.solvers.vorticity_solvers.registry import (
    VorticitySolverName,
    register_solver,
)


@register_solver(VorticitySolverName.PEACEMAN_RACHFORD)
class PRNavierStokesScheme(ADIVorticitySolver):
    def _compute_sweep_x_coeffs(
        self,
        w: np.ndarray,
        sf: np.ndarray,
        u: np.ndarray,
        dt: float,
        dx: float,
        dy: float,
    ) -> None:
        self._compute_sweep_x_coeffs_jit(
            w=w,
            sf=sf,
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_ind=self.c_ind,
            dx=dx,
            dy=dy,
            dt=dt,
            u_pt_ref=self.cfg.u_pt_ref,
            delta_u=self.cfg.delta_u,
            reynolds_number=self.cfg.reynolds_number,
            grashof_number=self.cfg.grashof_number,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
        )

    def _compute_sweep_y_coeffs(
        self,
        w: np.ndarray,
        sf: np.ndarray,
        u: np.ndarray,
        dt: float,
        dx: float,
        dy: float,
    ) -> None:
        self._compute_sweep_y_coeffs_jit(
            w=self._new_w,
            sf=sf,
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_ind=self.c_ind,
            dx=dx,
            dy=dy,
            dt=dt,
            u_pt_ref=self.cfg.u_pt_ref,
            delta_u=self.cfg.delta_u,
            reynolds_number=self.cfg.reynolds_number,
            grashof_number=self.cfg.grashof_number,
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
        )

    @staticmethod
    @njit
    def _compute_sweep_x_coeffs_jit(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_ind: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        reynolds_number: float,
        grashof_number: float,
        u_pt_ref: float,
        delta_u: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = w.shape
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy
        inv_re = 1.0 / reynolds_number
        inv_re2 = inv_re * inv_re
        dt_half = 0.5 * dt

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                gr = 0.0 if u[j, i] * delta_u - u_pt_ref < 0.0 else grashof_number

                a[j, i] = dt_half * (conv_x[j, i, 0] - inv_re * inv_dx2)

                b[j, i] = 1.0 + dt_half * (conv_x[j, i, 1] + 2.0 * inv_re * inv_dx2)

                c[j, i] = dt_half * (conv_x[j, i, 2] - inv_re * inv_dx2)

                rhs[j, i] = w[j, i] + dt_half * (
                    gr * inv_re2 * 0.5 * inv_dx * (u[j, i + 1] - u[j, i - 1])
                    + inv_re * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    - (
                        conv_y[j, i, 0] * w[j + 1, i]
                        + conv_y[j, i, 1] * w[j, i]
                        + conv_y[j, i, 2] * w[j - 1, i]
                    )
                    - c_ind[j, i] * sf[j, i]
                )

    @staticmethod
    @njit
    def _compute_sweep_y_coeffs_jit(
        w: NDArray[np.float64],
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_ind: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        reynolds_number: float,
        grashof_number: float,
        u_pt_ref: float,
        delta_u: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = w.shape
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy
        inv_re = 1.0 / reynolds_number
        inv_re2 = inv_re * inv_re
        dt_half = 0.5 * dt

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                gr = 0.0 if u[j, i] * delta_u - u_pt_ref < 0.0 else grashof_number

                a[i, j] = dt_half * (conv_y[j, i, 0] - inv_re * inv_dy2)

                b[i, j] = 1.0 + dt_half * (conv_y[j, i, 1] + 2.0 * inv_re * inv_dy2)

                c[i, j] = dt_half * (conv_y[j, i, 2] - inv_re * inv_dy2)

                rhs[i, j] = w[j, i] + dt_half * (
                    gr * inv_re2 * 0.5 * inv_dx * (u[j, i + 1] - u[j, i - 1])
                    + inv_re * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    - (
                        conv_x[j, i, 0] * w[j, i + 1]
                        + conv_x[j, i, 1] * w[j, i]
                        + conv_x[j, i, 2] * w[j, i - 1]
                    )
                    - c_ind[j, i] * sf[j, i]
                )
