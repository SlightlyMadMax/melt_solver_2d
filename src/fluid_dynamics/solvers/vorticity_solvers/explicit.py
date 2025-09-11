from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.core.geometry import DomainGeometry
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
        u: NDArray[np.float64],
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
        reynolds_number: float,
        grashof_number: float,
        u_pt_ref: float,
        delta_u: float,
        penalty_term: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n_y, n_x = w.shape
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy

        inv_re = 1.0 / reynolds_number
        inv_re2 = inv_re * inv_re

        result[0, :] = bottom_bc[:]
        result[n_y - 1, :] = top_bc[:]
        result[:, 0] = left_bc[:]
        result[:, n_x - 1] = right_bc[:]

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                gr = 0.0 if u[j, i] * delta_u - u_pt_ref < 0.0 else grashof_number

                convection_x = (
                    conv_x[j][i][0] * sf[j, i + 1]
                    + conv_x[j][i][1] * sf[j, i]
                    + conv_x[j][i][2] * sf[j, i - 1]
                )
                convection_y = (
                    conv_y[j][i][0] * sf[j + 1, i]
                    + conv_y[j][i][1] * sf[j, i]
                    + conv_y[j][i][2] * sf[j - 1, i]
                )

                convection = convection_x + convection_y

                result[j, i] = w[j, i] + dt * (
                    gr * inv_re2 * 0.5 * inv_dx * (u[j, i + 1] - u[j, i - 1])
                    + inv_re * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    + inv_re * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    - convection
                    - penalty_term[j, i] * sf[j, i]
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
        geometry: DomainGeometry = self.cfg.geometry
        dx, dy, dt = geometry.dx, geometry.dy, geometry.dt
        dx_scaled = dx / self.cfg.l
        dy_scaled = dy / self.cfg.l
        dt_scaled = dt * self.cfg.v / self.cfg.l

        self._prepare(sf=sf, u=u, conv_w=w, delta=delta)

        self._new_w = np.copy(w)

        self._compute_vorticity(
            w=w,
            sf=sf,
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            left_bc=self.left_bc,
            right_bc=self.right_bc,
            top_bc=self.top_bc,
            bottom_bc=self.bottom_bc,
            result=self._new_w,
            dx=dx_scaled,
            dy=dy_scaled,
            dt=dt_scaled,
            u_pt_ref=self.cfg.u_pt_ref,
            delta_u=self.cfg.delta_u,
            reynolds_number=self.cfg.reynolds_number,
            grashof_number=self.cfg.grashof_number,
            penalty_term=self.penalty_term,
        )

        return self._new_w
