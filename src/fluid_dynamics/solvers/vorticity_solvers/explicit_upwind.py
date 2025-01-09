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
from src.fluid_dynamics.utils import get_indicator_function as c_ind


@register_solver(VorticitySolverName.EXPLICIT_UPWIND)
class ExpUpwindNavierStokesSolver(ExplicitVorticitySolver):
    @staticmethod
    @njit
    def _compute_vorticity(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        v_x: NDArray[np.float64],
        v_y: NDArray[np.float64],
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
        u_ref: float,
        u_pt_ref: float,
        epsilon: float,
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
                if v_x[j, i] > 0:
                    advection_x = (w[j, i] * v_x[j, i] - w[j, i - 1] * v_x[j, i]) * inv_dx
                else:
                    advection_x = (w[j, i + 1] * v_x[j, i] - w[j, i] * v_x[j, i]) * inv_dx

                if v_y[j, i] > 0:
                    advection_y = (w[j, i] * v_y[j, i] - w[j - 1, i] * v_y[j, i]) * inv_dy
                else:
                    advection_y = (w[j + 1, i] * v_y[j, i] - w[j, i] * v_y[j, i]) * inv_dy

                advection = advection_y + advection_x

                result[j, i] = w[j, i] + dt * (
                    grashof_number
                    * inv_re2
                    * 0.5
                    * inv_dx
                    * (u[j, i + 1] - u[j, i - 1])
                    + inv_re * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    + inv_re * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    - advection
                    # + inv_re * c_ind(u=u[j, i], u_pt_ref=u_pt_ref, eps=epsilon) * sf[j, i]
                )

        return result

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> (NDArray[np.float64], NDArray[np.float64]):
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
        self._v_x, self._v_y = compute_velocity_from_sf(
            sf=sf,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
        )
        self._compute_vorticity(
            w=w,
            sf=sf,
            u=u,
            v_x=self._v_x,
            v_y=self._v_y,
            left_bc=self.left_bc,
            right_bc=self.right_bc,
            top_bc=self.top_bc,
            bottom_bc=self.bottom_bc,
            result=self._new_w,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
            dt=self.geometry.dt * self.parameters.v / self.geometry.length_scale,
            u_ref=self.parameters.u_ref,
            u_pt_ref=self.parameters.u_pt_ref,
            reynolds_number=self.parameters.reynolds_number,
            grashof_number=self.parameters.grashof_number,
            epsilon=self.parameters.epsilon,
        )

        return self._new_w
