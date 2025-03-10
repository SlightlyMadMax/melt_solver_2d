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
from src.fluid_dynamics.utils import calculate_indicator_function


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
        c_ind: NDArray[np.float64],
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
                    conv_x[j][i][0] * w[j, i + 1]
                    + conv_x[j][i][1] * w[j, i]
                    + conv_x[j][i][2] * w[j, i - 1]
                )
                convection_y = (
                    conv_y[j][i][0] * w[j + 1, i]
                    + conv_y[j][i][1] * w[j, i]
                    + conv_y[j][i][2] * w[j - 1, i]
                )

                convection = convection_x + convection_y

                result[j, i] = w[j, i] + dt * (
                    gr * inv_re2 * 0.5 * inv_dx * (u[j, i + 1] - u[j, i - 1])
                    + inv_re * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    + inv_re * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    - convection
                    - c_ind[j, i] * sf[j, i]
                )

        return result

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        conv_x, conv_y = self.convective_operator(sf=sf)
        calculate_indicator_function(
            u=u * self.parameters.delta_u + self.parameters.u_ref,
            u_pt=self.parameters.u_pt,
            eps=self.parameters.epsilon,
            result=self.c_ind,
        )
        self.c_ind *= self.geometry.length_scale**3 / self.parameters.v
        self._new_w = np.copy(w)
        self.calculate_boundary_conditions(
            sf=sf,
            top_bc=self.top_bc,
            right_bc=self.right_bc,
            bottom_bc=self.bottom_bc,
            left_bc=self.left_bc,
            order=self.bc_order,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
        )
        self._compute_vorticity(
            w=w,
            sf=sf,
            u=u,
            conv_x=conv_x,
            conv_y=conv_y,
            left_bc=self.left_bc,
            right_bc=self.right_bc,
            top_bc=self.top_bc,
            bottom_bc=self.bottom_bc,
            result=self._new_w,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
            dt=self.geometry.dt * self.parameters.v / self.geometry.length_scale,
            u_pt_ref=self.parameters.u_pt_ref,
            delta_u=self.parameters.delta_u,
            reynolds_number=self.parameters.reynolds_number,
            grashof_number=self.parameters.grashof_number,
            c_ind=self.c_ind,
        )

        return self._new_w
