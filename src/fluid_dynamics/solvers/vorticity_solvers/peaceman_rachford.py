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
        dt: float,
        dx: float,
        dy: float,
    ) -> None:
        self._compute_sweep_x_coeffs_jit(
            w=w,
            sf=sf,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            p=self.penalty_term,
            px_half=self.px_half,
            py_half=self.py_half,
            buoy=self.buoyancy_term,
            dx=dx,
            dy=dy,
            dt=dt,
            re=self.cfg.reynolds_number,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
        )

    def _compute_sweep_y_coeffs(
        self,
        w: np.ndarray,
        sf: np.ndarray,
        dt: float,
        dx: float,
        dy: float,
    ) -> None:
        self._compute_sweep_y_coeffs_jit(
            w=self._new_w,
            sf=sf,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            p=self.penalty_term,
            px_half=self.px_half,
            py_half=self.py_half,
            buoy=self.buoyancy_term,
            dx=dx,
            dy=dy,
            dt=dt,
            re=self.cfg.reynolds_number,
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
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        p: NDArray[np.float64],
        px_half: NDArray[np.float64],
        py_half: NDArray[np.float64],
        buoy: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        re: float,
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
        inv_re = 1.0 / re
        dt_half = 0.5 * dt

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                p_ip1j = px_half[j, i]
                p_im1j = px_half[j, i - 1]
                p_ijp1 = py_half[j, i]
                p_ijm1 = py_half[j - 1, i]

                a[j, i] = dt_half * (conv_x[j, i, 0] - inv_re * inv_dx2)

                b[j, i] = 1.0 + dt_half * (conv_x[j, i, 1] + 2.0 * inv_re * inv_dx2)

                c[j, i] = dt_half * (conv_x[j, i, 2] - inv_re * inv_dx2)

                rhs[j, i] = w[j, i] + dt_half * (
                    buoy[j, i]
                    + inv_re * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    - (
                        conv_y[j, i, 0] * w[j + 1, i]
                        + conv_y[j, i, 1] * w[j, i]
                        + conv_y[j, i, 2] * w[j - 1, i]
                    )
                    # - p[j, i] * sf[j, i]
                    + inv_dx2
                    * (
                        p_ip1j * (sf[j, i + 1] - sf[j, i])
                        - p_im1j * (sf[j, i] - sf[j, i - 1])
                    )
                    + inv_dy2
                    * (
                        p_ijp1 * (sf[j + 1, i] - sf[j, i])
                        - p_ijm1 * (sf[j, i] - sf[j - 1, i])
                    )
                )

    @staticmethod
    @njit
    def _compute_sweep_y_coeffs_jit(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        p: NDArray[np.float64],
        px_half: NDArray[np.float64],
        py_half: NDArray[np.float64],
        buoy: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        re: float,
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
        inv_re = 1.0 / re
        dt_half = 0.5 * dt

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                p_ip1j = px_half[j, i]
                p_im1j = px_half[j, i - 1]
                p_ijp1 = py_half[j, i]
                p_ijm1 = py_half[j - 1, i]

                a[i, j] = dt_half * (conv_y[j, i, 0] - inv_re * inv_dy2)

                b[i, j] = 1.0 + dt_half * (conv_y[j, i, 1] + 2.0 * inv_re * inv_dy2)

                c[i, j] = dt_half * (conv_y[j, i, 2] - inv_re * inv_dy2)

                rhs[i, j] = w[j, i] + dt_half * (
                    buoy[j, i]
                    + inv_re * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    - (
                        conv_x[j, i, 0] * w[j, i + 1]
                        + conv_x[j, i, 1] * w[j, i]
                        + conv_x[j, i, 2] * w[j, i - 1]
                    )
                    # - p[j, i] * sf[j, i]
                    + inv_dx2
                    * (
                        p_ip1j * (sf[j, i + 1] - sf[j, i])
                        - p_im1j * (sf[j, i] - sf[j, i - 1])
                    )
                    + inv_dy2
                    * (
                        p_ijp1 * (sf[j + 1, i] - sf[j, i])
                        - p_ijm1 * (sf[j, i] - sf[j - 1, i])
                    )
                )
