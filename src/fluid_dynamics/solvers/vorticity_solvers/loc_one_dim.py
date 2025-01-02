import numba
import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition, BoundaryConditionType
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.solvers.vorticity_solvers.registry import (
    VorticitySolverName,
    register_solver,
)
from src.geometry import DomainGeometry
from src.base_solver import Sweep2DSolver
from src.utils import solve_tridiagonal


@register_solver(VorticitySolverName.LOC_ONE_DIM)
class LODNavierStokesScheme(Sweep2DSolver):
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
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

        # Pre-allocate some arrays that will be used in the calculations
        self._temp_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._new_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_x(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        result: NDArray[np.float64],
        rhs: NDArray[np.float64],
        a_x: NDArray[np.float64],
        b_x: NDArray[np.float64],
        c_x: NDArray[np.float64],
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

        inv_re = 1.0 / reynolds_number
        inv_re2 = inv_re * inv_re

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                a_x[i] = (
                    dt
                    * inv_dx
                    * (
                        (sf[j + 1, i + 1] - sf[j - 1, i + 1]) * 0.25 * inv_dy
                        - inv_re * inv_dx
                    )
                )

                b_x[i] = 1.0 + 2.0 * inv_re * dt * inv_dx2

                c_x[i] = (
                    dt
                    * inv_dx
                    * (
                        (sf[j - 1, i - 1] - sf[j + 1, i - 1]) * 0.25 * inv_dy
                        - inv_re * inv_dx
                    )
                )

                rhs[i] = w[j, i] + dt * (
                    grashof_number
                    * inv_re2
                    * 0.5
                    * inv_dx
                    * (u[j, i + 1] - u[j, i - 1])
                    # + inv_re * c_ind(u=u[j, i], u_pt_ref=u_pt_ref, eps=epsilon) * sf[j, i]
                )

            result[j, :] = solve_tridiagonal(
                a=a_x,
                b=b_x,
                c=c_x,
                f=rhs,
                left_type=1,  # Dirichlet
                left_value=0.5 * inv_dx2 * (sf[j, 2] - 8.0 * sf[j, 1]),
                right_type=1,  # Dirichlet
                right_value=0.5 * inv_dx2 * (sf[j, n_x - 3] - 8.0 * sf[j, n_x - 2]),
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
        rhs: NDArray[np.float64],
        a_y: NDArray[np.float64],
        b_y: NDArray[np.float64],
        c_y: NDArray[np.float64],
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
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy

        inv_re = 1.0 / reynolds_number

        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                a_y[j] = (
                    dt
                    * inv_dy
                    * (
                        (sf[j + 1, i - 1] - sf[j + 1, i + 1]) * 0.25 * inv_dx
                        - inv_re * inv_dy
                    )
                )

                b_y[j] = 1.0 + 2.0 * inv_re * dt * inv_dy2

                c_y[j] = (
                    dt
                    * inv_dy
                    * (
                        (sf[j - 1, i + 1] - sf[j - 1, i - 1]) * 0.25 * inv_dx
                        - inv_re * inv_dy
                    )
                )

                rhs[j] = w[j, i]

            result[:, i] = solve_tridiagonal(
                a=a_y,
                b=b_y,
                c=c_y,
                f=rhs,
                left_type=1,  # Dirichlet
                left_value=0.5 * inv_dy2 * (sf[2, i] - 8.0 * sf[1, i]),
                right_type=1,  # Dirichlet
                right_value=0.5 * inv_dy2 * (sf[n_y - 3, i] - 8.0 * sf[n_y - 2, i]),
                h=dy,
            )

        return result

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> (NDArray[np.float64], NDArray[np.float64]):
        self._temp_w = np.copy(w)

        self._compute_sweep_x(
            w=w,
            sf=sf,
            u=u,
            result=self._temp_w,
            rhs=self._rhs_x,
            a_x=self._a_x,
            b_x=self._b_x,
            c_x=self._c_x,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
            dt=self.geometry.dt * self.parameters.v / self.geometry.length_scale,
            u_ref=self.parameters.u_ref,
            u_pt_ref=self.parameters.u_pt_ref,
            reynolds_number=self.parameters.reynolds_number,
            grashof_number=self.parameters.grashof_number,
            epsilon=self.parameters.epsilon,
        )
        self._new_w = np.copy(self._temp_w)
        self._compute_sweep_y(
            w=self._temp_w,
            sf=sf,
            u=u,
            result=self._new_w,
            rhs=self._rhs_y,
            a_y=self._a_y,
            b_y=self._b_y,
            c_y=self._c_y,
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
