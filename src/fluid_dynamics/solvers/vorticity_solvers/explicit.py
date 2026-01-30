from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import (
    ExplicitVorticitySolver,
)
from src.fluid_dynamics.solvers.vorticity_solvers.registry import (
    VorticitySolverName,
    register_solver,
)


@register_solver(VorticitySolverName.EXPLICIT)
class ExplicitNavierStokesSolver(ExplicitVorticitySolver):
    @staticmethod
    @njit
    def _compute_vorticity(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        left_bc: NDArray[np.float64],
        right_bc: NDArray[np.float64],
        top_bc: NDArray[np.float64],
        bottom_bc: NDArray[np.float64],
        result: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        re: float,
        px_half: NDArray[np.float64],
        py_half: NDArray[np.float64],
        buoy: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n_y, n_x = w.shape
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        inv_re = 1.0 / re

        result[0, :] = bottom_bc[:]
        result[n_y - 1, :] = top_bc[:]
        result[:, 0] = left_bc[:]
        result[:, n_x - 1] = right_bc[:]

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                convection_x = (
                    conv_x[j, i, 0] * sf[j, i + 1]
                    + conv_x[j, i, 1] * sf[j, i]
                    + conv_x[j, i, 2] * sf[j, i - 1]
                )
                convection_y = (
                    conv_y[j, i, 0] * sf[j + 1, i]
                    + conv_y[j, i, 1] * sf[j, i]
                    + conv_y[j, i, 2] * sf[j - 1, i]
                )

                convection = convection_x + convection_y

                result[j, i] = w[j, i] + dt * (
                    buoy[j, i]
                    + inv_re * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    + inv_re * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    - convection
                    + inv_dx2
                    * (
                        px_half[j, i] * (sf[j, i + 1] - sf[j, i])
                        - px_half[j, i - 1] * (sf[j, i] - sf[j, i - 1])
                    )
                    + inv_dy2
                    * (
                        py_half[j, i] * (sf[j + 1, i] - sf[j, i])
                        - py_half[j - 1, i] * (sf[j, i] - sf[j - 1, i])
                    )
                )

        return result

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        delta: Optional[float] = None,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        dx_scaled, dy_scaled, dt_scaled = self.cfg.scaled_grid_steps

        self._prepare(sf=sf, u=u, conv_w=w, delta=delta)

        self._w_new[:, :] = w

        self._compute_vorticity(
            w=w,
            sf=sf,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            px_half=self.px_half,
            py_half=self.py_half,
            left_bc=self.left_bc,
            right_bc=self.right_bc,
            top_bc=self.top_bc,
            bottom_bc=self.bottom_bc,
            result=self._w_new,
            dx=dx_scaled,
            dy=dy_scaled,
            dt=dt_scaled,
            re=self.cfg.reynolds_number,
            buoy=self.buoyancy_term,
        )

        return self._w_new
