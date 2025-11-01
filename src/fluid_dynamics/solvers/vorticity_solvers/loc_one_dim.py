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


@register_solver(VorticitySolverName.LOC_ONE_DIM)
class LODNavierStokesScheme(ADIVorticitySolver):
    def _compute_sweep_x_coeffs(self, w: np.ndarray, sf: np.ndarray) -> None:
        dx, _, dt = self.cfg.scaled_grid_steps
        self._compute_sweep_x_coeffs_jit(
            w=w,
            sf=sf,
            conv_x=self._conv_x,
            penalty_term=self.penalty_term,
            buoyancy_term=self.buoyancy_term,
            dx=dx,
            dt=dt,
            reynolds_number=self.cfg.reynolds_number,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
        )

    def _compute_sweep_y_coeffs(self, w: np.ndarray, sf: np.ndarray) -> None:
        _, dy, dt = self.cfg.scaled_grid_steps
        self._compute_sweep_y_coeffs_jit(
            w=self._new_w,
            sf=sf,
            conv_y=self._conv_y,
            penalty_term=self.penalty_term,
            buoyancy_term=self.buoyancy_term,
            dy=dy,
            dt=dt,
            reynolds_number=self.cfg.reynolds_number,
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
        penalty_term: NDArray[np.float64],
        buoyancy_term: NDArray[np.float64],
        dx: float,
        dt: float,
        reynolds_number: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = w.shape
        inv_dx2 = 1.0 / (dx * dx)
        inv_re = 1.0 / reynolds_number

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                a[j, i] = dt * (conv_x[j, i, 0] - inv_re * inv_dx2)

                b[j, i] = 1.0 + dt * (conv_x[j, i, 1] + 2.0 * inv_re * inv_dx2)

                c[j, i] = dt * (conv_x[j, i, 2] - inv_re * inv_dx2)

                rhs[j, i] = w[j, i] + 0.5 * dt * (
                    buoyancy_term[j, i] - penalty_term[j, i] * sf[j, i]
                )

    @staticmethod
    @njit
    def _compute_sweep_y_coeffs_jit(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        penalty_term: NDArray[np.float64],
        buoyancy_term: NDArray[np.float64],
        dy: float,
        dt: float,
        reynolds_number: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = w.shape
        inv_dy2 = 1.0 / (dy * dy)
        inv_re = 1.0 / reynolds_number

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                a[i, j] = dt * (conv_y[j, i, 0] - inv_re * inv_dy2)

                b[i, j] = 1.0 + 2.0 * dt * (conv_y[j, i, 1] + inv_re * inv_dy2)

                c[i, j] = dt * (conv_y[j, i, 2] - inv_re * inv_dy2)

                rhs[i, j] = w[j, i] + 0.5 * dt * (
                    buoyancy_term[j, i] - penalty_term[j, i] * sf[j, i]
                )
