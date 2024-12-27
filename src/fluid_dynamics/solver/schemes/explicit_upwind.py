import numba
import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryConditionType, BoundaryCondition
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.solver.registry import NavierStokesSchemeName, register_scheme
from src.geometry import DomainGeometry
from src.base_scheme import BaseScheme
from src.fluid_dynamics.utils import get_indicator_function as c_ind
from src.utils import solve_poisson_sor


@register_scheme(NavierStokesSchemeName.EXPLICIT_UPWIND)
class ExpUpwindNavierStokesScheme(BaseScheme):
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
        sf_max_iters: int = 50,
        sf_stopping_criteria: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__(
            geometry=geometry,
            top_bc=top_bc,
            right_bc=right_bc,
            bottom_bc=bottom_bc,
            left_bc=left_bc,
        )

        self.parameters = parameters
        self.sf_max_iters = sf_max_iters
        self.sf_stopping_criteria = sf_stopping_criteria

        # Pre-allocate some arrays that will be used in the calculations
        self._new_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._sf: NDArray[np.float64] = np.empty((self.geometry.n_y, self.geometry.n_x))

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_vorticity(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
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

        result[0, :] = -2.0 * inv_dy2 * sf[1, :]
        result[n_y - 1, :] = -2.0 * inv_dy2 * sf[n_y - 2, :]
        result[:, 0] = -2.0 * inv_dx2 * sf[:, 1]
        result[:, n_x - 1] = -2.0 * inv_dx2 * sf[:, n_x - 2]

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                v_x = (sf[i, j + 1] - sf[i, j - 1]) * 0.5 * inv_dy
                v_y = -(sf[i + 1, j] - sf[i - 1, j]) * 0.5 * inv_dx

                if v_x > 0:
                    advection_x = (w[i, j] * v_x - w[i - 1, j] * v_x) * inv_dx
                else:
                    advection_x = (w[i + 1, j] * v_x - w[i, j] * v_x) * inv_dx

                if v_y > 0:
                    advection_y = (w[i, j] * v_y - w[i, j - 1] * v_y) * inv_dy
                else:
                    advection_y = (w[i, j + 1] * v_y - w[i, j] * v_y) * inv_dy

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
        self._compute_vorticity(
            w=w,
            sf=sf,
            u=u,
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
        self._sf = solve_poisson_sor(
            initial_guess=sf,
            rhs=self._new_w,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
            max_iters=self.sf_max_iters,
            stopping_criteria=self.sf_stopping_criteria,
            right_value=(
                self.right_bc.get_value(t=time)
                if self.right_bc.boundary_type == BoundaryConditionType.DIRICHLET
                else None
            ),
            left_value=(
                self.left_bc.get_value(t=time)
                if self.left_bc.boundary_type == BoundaryConditionType.DIRICHLET
                else None
            ),
            top_value=(
                self.top_bc.get_value(t=time)
                if self.top_bc.boundary_type == BoundaryConditionType.DIRICHLET
                else None
            ),
            bottom_value=(
                self.bottom_bc.get_value(t=time)
                if self.bottom_bc.boundary_type == BoundaryConditionType.DIRICHLET
                else None
            ),
        )

        return self._sf, self._new_w
