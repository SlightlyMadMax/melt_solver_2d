import numba
import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryConditionType, BoundaryCondition
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.schemes.registry import NavierStokesSchemeName
from src.fluid_dynamics.schemes.utils import register_scheme
from src.geometry import DomainGeometry
from src.base_solver import BaseScheme
from src.fluid_dynamics.utils import (
    get_indicator_function as c_ind,
    thermal_expansion_coefficient as thermal_exp,
)
from src import constants as cfg


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
        u_ref: float,
        u_pt_ref: float,
        visc: float,
        epsilon: float,
    ) -> NDArray[np.float64]:
        n_y, n_x = w.shape
        inv_dx = 1.0 / dx
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy = 1.0 / dy
        inv_dy2 = 1.0 / (dy * dy)

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
                    -cfg.G
                    * thermal_exp(u=u[j, i], u_ref=u_ref, u_pt_ref=u_pt_ref)
                    * 0.5
                    * inv_dx
                    * (u[j, i + 1] - u[j, i - 1])
                    + visc * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    + visc * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    - advection
                    # + visc * c_ind(u=u[j, i], u_pt_ref=u_pt_ref, eps=epsilon) * sf[j, i]
                )

        return result

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_stream_function(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        dx: float,
        dy: float,
        max_iters: int,
        stopping_criteria: float,
        right_value: NDArray[np.float64] = None,
        left_value: NDArray[np.float64] = None,
        top_value: NDArray[np.float64] = None,
        bottom_value: NDArray[np.float64] = None,
    ) -> NDArray[np.float64]:
        n_y, n_x = w.shape
        beta = dx / dy
        factor = 0.5 / (1.0 + beta * beta)

        result = np.copy(sf)

        result[0, :] = top_value
        result[n_y - 1, :] = bottom_value
        result[:, 0] = left_value
        result[:, n_x - 1] = right_value

        temp = np.copy(result)

        for iteration in range(max_iters):
            for i in range(1, n_x - 1):
                for j in range(1, n_y - 1):
                    result[j, i] = factor * (
                        temp[j, i + 1]
                        + result[j, i - 1]
                        + beta * beta * temp[j + 1, i]
                        + beta * beta * result[j - 1, i]
                        + dx * dx * w[j, i]
                    )
            diff = np.linalg.norm(temp - result)
            if diff < stopping_criteria:
                break
            temp = np.copy(result)
        return result

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> (NDArray[np.float64], NDArray[np.float64]):
        self._compute_vorticity(
            w=w,
            sf=sf,
            u=u,
            result=self._new_w,
            dx=self.geometry.dx,
            dy=self.geometry.dy,
            dt=self.geometry.dt,
            u_ref=self.parameters.u_ref,
            u_pt_ref=self.parameters.u_pt_ref,
            visc=self.parameters.kinematic_viscosity_at_u_ref,
            epsilon=self.parameters.epsilon,
        )
        self._sf = self._compute_stream_function(
            w=self._new_w,
            sf=sf,
            dx=self.geometry.dx,
            dy=self.geometry.dy,
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
