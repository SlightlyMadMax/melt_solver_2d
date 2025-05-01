import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import (
    ImplicitVorticitySolver,
)
from src.fluid_dynamics.solvers.vorticity_solvers.registry import (
    VorticitySolverName,
    register_solver,
)
from src.fluid_dynamics.utils import calculate_indicator_function
from src.heat_transfer.coefficient_smoothing.delta import get_max_delta
from src.utils.thomas import solve_tridiagonal


@register_solver(VorticitySolverName.PEACEMAN_RACHFORD)
class PRNavierStokesScheme(ImplicitVorticitySolver):
    @staticmethod
    @njit
    def _compute_sweep_x(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        left_bc: NDArray[np.float64],
        right_bc: NDArray[np.float64],
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

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                gr = 0.0 if u[j, i] * delta_u - u_pt_ref < 0.0 else grashof_number

                a_x[i] = 0.5 * dt * (conv_x[j, i, 0] - inv_re * inv_dx2)

                b_x[i] = 1.0 + 0.5 * dt * (conv_x[j, i, 1] + 2.0 * inv_re * inv_dx2)

                c_x[i] = 0.5 * dt * (conv_x[j, i, 2] - inv_re * inv_dx2)

                rhs[i] = w[j, i] + 0.5 * dt * (
                    gr * inv_re2 * 0.5 * inv_dx * (u[j, i + 1] - u[j, i - 1])
                    + inv_re * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    - (
                        conv_y[j, i, 0] * w[j + 1, i]
                        + conv_y[j, i, 1] * w[j, i]
                        + conv_y[j, i, 2] * w[j - 1, i]
                    )
                    - c_ind[j, i] * sf[j, i]
                )

            solve_tridiagonal(
                a=a_x,
                b=b_x,
                c=c_x,
                f=rhs,
                result=result[j, :],
                left_type=1,  # Dirichlet
                left_value=left_bc[j],
                right_type=1,  # Dirichlet
                right_value=right_bc[j],
                h=dx,
            )

        return result

    @staticmethod
    @njit
    def _compute_sweep_y(
        w: NDArray[np.float64],
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        top_bc: NDArray[np.float64],
        bottom_bc: NDArray[np.float64],
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

        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                gr = 0.0 if u[j, i] * delta_u - u_pt_ref < 0.0 else grashof_number

                a_y[j] = 0.5 * dt * (conv_y[j, i, 0] - inv_re * inv_dy2)

                b_y[j] = 1.0 + 0.5 * dt * (conv_y[j, i, 1] + 2.0 * inv_re * inv_dy2)

                c_y[j] = 0.5 * dt * (conv_y[j, i, 2] - inv_re * inv_dy2)

                rhs[j] = w[j, i] + 0.5 * dt * (
                    gr * inv_re2 * 0.5 * inv_dx * (u[j, i + 1] - u[j, i - 1])
                    + inv_re * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    - (
                        conv_x[j, i, 0] * w[j, i + 1]
                        + conv_x[j, i, 1] * w[j, i]
                        + conv_x[j, i, 2] * w[j, i - 1]
                    )
                    - c_ind[j, i] * sf[j, i]
                )

            solve_tridiagonal(
                a=a_y,
                b=b_y,
                c=c_y,
                f=rhs,
                result=result[:, i],
                left_type=1,  # Dirichlet
                left_value=bottom_bc[i],
                right_type=1,  # Dirichlet
                right_value=top_bc[i],
                h=dy,
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
        # delta = get_max_delta(
        #     u=u * self.parameters.delta_u + self.parameters.u_ref,
        #     u_pt=self.parameters.u_pt,
        # )
        calculate_indicator_function(
            u=u * self.parameters.delta_u + self.parameters.u_ref,
            u_pt=self.parameters.u_pt,
            eps=self.parameters.epsilon,
            # delta=delta,
            result=self.c_ind,
        )
        self.c_ind *= self.geometry.length_scale**3 / self.parameters.v

        self._temp_w = np.copy(w)

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
        self._compute_sweep_x(
            w=w,
            sf=sf,
            u=u,
            conv_x=conv_x,
            conv_y=conv_y,
            left_bc=self.left_bc,
            right_bc=self.right_bc,
            result=self._temp_w,
            rhs=self._rhs_x,
            a_x=self._a_x,
            b_x=self._b_x,
            c_x=self._c_x,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
            dt=self.geometry.dt * self.parameters.v / self.geometry.length_scale,
            u_pt_ref=self.parameters.u_pt_ref,
            delta_u=self.parameters.delta_u,
            reynolds_number=self.parameters.reynolds_number,
            grashof_number=self.parameters.grashof_number,
            c_ind=self.c_ind,
        )
        self._new_w = np.copy(self._temp_w)
        self._compute_sweep_y(
            w=self._temp_w,
            sf=sf,
            u=u,
            conv_x=conv_x,
            conv_y=conv_y,
            top_bc=self.top_bc,
            bottom_bc=self.bottom_bc,
            result=self._new_w,
            rhs=self._rhs_y,
            a_y=self._a_y,
            b_y=self._b_y,
            c_y=self._c_y,
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
