import numba
import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition, BoundaryConditionType
from src.constants import G
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.solver.registry import NavierStokesSchemeName
from src.fluid_dynamics.solver.utils import register_scheme
from src.geometry import DomainGeometry
from src.base_solver import Sweep2DScheme
from src.fluid_dynamics.utils import (
    get_indicator_function as c_ind,
    thermal_expansion_coefficient as thermal_exp,
)
from src.utils import solve_tridiagonal


@register_scheme(NavierStokesSchemeName.PEACEMAN_RACHFORD)
class PRNavierStokesScheme(Sweep2DScheme):
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
        implicit_sf_max_iters: int = 5,
        implicit_sf_stopping_criteria: float = 1e-6,
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
        self.implicit_sf_max_iters = implicit_sf_max_iters
        self.implicit_sf_stopping_criteria = implicit_sf_stopping_criteria

        # Pre-allocate some arrays that will be used in the calculations
        self._temp_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._new_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._sf: NDArray[np.float64] = np.empty((self.geometry.n_y, self.geometry.n_x))

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_x(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        result: NDArray[np.float64],
        a_x: NDArray[np.float64],
        b_x: NDArray[np.float64],
        c_x: NDArray[np.float64],
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

        f = np.empty(n_x)

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                a_x[i] = (
                    0.5
                    * dt
                    * inv_dx
                    * (
                        (sf[j + 1, i + 1] - sf[j - 1, i + 1]) * 0.25 * inv_dy
                        - visc * inv_dx
                    )
                )

                b_x[i] = 1.0 + visc * dt * inv_dx2

                c_x[i] = (
                    0.5
                    * dt
                    * inv_dx
                    * (
                        (sf[j - 1, i - 1] - sf[j + 1, i - 1]) * 0.25 * inv_dy
                        - visc * inv_dx
                    )
                )

                f[i] = w[j, i] + 0.5 * dt * (
                    -G
                    * thermal_exp(u=u[j, i], u_ref=u_ref, u_pt_ref=u_pt_ref)
                    * 0.5
                    * inv_dx
                    * (u[j, i + 1] - u[j, i - 1])
                    + visc * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j - 1, i - 1] - sf[j - 1, i + 1])
                    * w[j - 1, i]
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j + 1, i + 1] - sf[j + 1, i - 1])
                    * w[j + 1, i]
                    # + visc * c_ind(u=u[j, i], u_pt_ref=u_pt_ref, eps=epsilon) * sf[j, i]
                )

            result[j, :] = solve_tridiagonal(
                a=a_x,
                b=b_x,
                c=c_x,
                f=f,
                left_type=1,  # Dirichlet
                left_value=-0.5 * inv_dx2 * (8.0 * sf[j, 1] - sf[j, 2]),
                right_type=1,  # Dirichlet
                right_value=-0.5 * inv_dx2 * (8.0 * sf[j, n_x - 2] - sf[j, n_x - 3]),
                h=dx,
            )

        return result

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_y(
        w: NDArray[np.float64],
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        result: NDArray[np.float64],
        a_y: NDArray[np.float64],
        b_y: NDArray[np.float64],
        c_y: NDArray[np.float64],
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

        f = np.empty(n_y)

        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                a_y[j] = (
                    0.5
                    * dt
                    * inv_dy
                    * (
                        (sf[j + 1, i - 1] - sf[j + 1, i + 1]) * 0.25 * inv_dx
                        - visc * inv_dy
                    )
                )

                b_y[j] = 1.0 + visc * dt * inv_dy2

                c_y[j] = (
                    0.5
                    * dt
                    * inv_dy
                    * (
                        (sf[j - 1, i + 1] - sf[j - 1, i - 1]) * 0.25 * inv_dx
                        - visc * inv_dy
                    )
                )

                f[j] = w[j, i] + 0.5 * dt * (
                    -G
                    * thermal_exp(u=u[j, i], u_ref=u_ref, u_pt_ref=u_pt_ref)
                    * 0.5
                    * inv_dx
                    * (u[j, i + 1] - u[j, i - 1])
                    + visc * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j + 1, i - 1] - sf[j - 1, i - 1])
                    * w[j, i - 1]
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j - 1, i + 1] - sf[j + 1, i + 1])
                    * w[j, i + 1]
                    # + visc * c_ind(u=u[j, i], u_pt_ref=u_pt_ref, eps=epsilon) * sf[j, i]
                )

            result[:, i] = solve_tridiagonal(
                a=a_y,
                b=b_y,
                c=c_y,
                f=f,
                left_type=1,  # Dirichlet
                left_value=-0.5 * inv_dy2 * (8.0 * sf[1, i] - sf[2, i]),
                right_type=1,  # Dirichlet
                right_value=-0.5 * inv_dy2 * (8.0 * sf[n_y - 2, i] - sf[n_y - 3, i]),
                h=dy,
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
        temp_sf = np.copy(sf)
        for iteration in range(self.implicit_sf_max_iters):
            self._compute_sweep_x(
                w=w,
                sf=temp_sf,
                u=u,
                result=self._temp_w,
                a_x=self._a_x,
                b_x=self._b_x,
                c_x=self._c_x,
                dx=self.geometry.dx,
                dy=self.geometry.dy,
                dt=self.geometry.dt,
                u_ref=self.parameters.u_ref,
                u_pt_ref=self.parameters.u_pt_ref,
                visc=self.parameters.kinematic_viscosity_at_u_ref,
                epsilon=self.parameters.epsilon,
            )
            self._new_w = np.copy(self._temp_w)
            self._compute_sweep_y(
                w=self._temp_w,
                sf=temp_sf,
                u=u,
                result=self._new_w,
                a_y=self._a_y,
                b_y=self._b_y,
                c_y=self._c_y,
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
                sf=temp_sf,
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
            diff = np.linalg.norm(temp_sf - self._sf)
            if diff < self.implicit_sf_stopping_criteria:
                break
            # temp_sf = 0.5 * (temp_sf + self._sf)
            temp_sf = np.copy(self._sf)

        return self._sf, self._new_w
