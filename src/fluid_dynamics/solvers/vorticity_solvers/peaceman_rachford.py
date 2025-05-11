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
from src.heat_transfer.coefficient_smoothing.mushy_zone import get_mushy_zone_width


@register_solver(VorticitySolverName.PEACEMAN_RACHFORD)
class PRNavierStokesScheme(ImplicitVorticitySolver):
    @staticmethod
    @njit
    def _compute_sweep_x_coeff(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_ind: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        reynolds_number: float,
        grashof_number: float,
        u_pt_ref: float,
        delta_u: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
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

                a[j, i] = 0.5 * dt * (conv_x[j, i, 0] - inv_re * inv_dx2)

                b[j, i] = 1.0 + 0.5 * dt * (conv_x[j, i, 1] + 2.0 * inv_re * inv_dx2)

                c[j, i] = 0.5 * dt * (conv_x[j, i, 2] - inv_re * inv_dx2)

                rhs[j, i] = w[j, i] + 0.5 * dt * (
                    gr * inv_re2 * 0.5 * inv_dx * (u[j, i + 1] - u[j, i - 1])
                    + inv_re * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    - (
                        conv_y[j, i, 0] * w[j + 1, i]
                        + conv_y[j, i, 1] * w[j, i]
                        + conv_y[j, i, 2] * w[j - 1, i]
                    )
                    - c_ind[j, i] * sf[j, i]
                )

    @staticmethod
    @njit
    def _compute_sweep_y_coeff(
        w: NDArray[np.float64],
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_ind: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        reynolds_number: float,
        grashof_number: float,
        u_pt_ref: float,
        delta_u: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
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

                a[i, j] = 0.5 * dt * (conv_y[j, i, 0] - inv_re * inv_dy2)

                b[i, j] = 1.0 + 0.5 * dt * (conv_y[j, i, 1] + 2.0 * inv_re * inv_dy2)

                c[i, j] = 0.5 * dt * (conv_y[j, i, 2] - inv_re * inv_dy2)

                rhs[i, j] = w[j, i] + 0.5 * dt * (
                    gr * inv_re2 * 0.5 * inv_dx * (u[j, i + 1] - u[j, i - 1])
                    + inv_re * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    - (
                        conv_x[j, i, 0] * w[j, i + 1]
                        + conv_x[j, i, 1] * w[j, i]
                        + conv_x[j, i, 2] * w[j, i - 1]
                    )
                    - c_ind[j, i] * sf[j, i]
                )

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        dx, dy, dt = self.geometry.dx, self.geometry.dy, self.geometry.dt
        n_x, n_y = self.geometry.n_x, self.geometry.n_y
        length_scale = self.geometry.length_scale
        self.convective_operator(conv_x=self._conv_x, conv_y=self._conv_y, sf=sf)
        u_dim = u * self.parameters.delta_u + self.parameters.u_ref
        delta = get_mushy_zone_width(
            u=u_dim,
            u_pt=self.parameters.u_pt,
            h_x=dx,
            h_y=dy,
        )
        calculate_indicator_function(
            u=u_dim,
            u_pt=self.parameters.u_pt,
            eps=self.parameters.epsilon,
            delta=delta,
            result=self.c_ind,
        )
        self.c_ind *= length_scale**3 / self.parameters.v

        self.calculate_boundary_conditions(
            sf=sf,
            top_bc=self.top_bc,
            right_bc=self.right_bc,
            bottom_bc=self.bottom_bc,
            left_bc=self.left_bc,
            order=self.bc_order,
            dx=dx / length_scale,
            dy=dy / length_scale,
        )

        self._compute_sweep_x_coeff(
            w=w,
            sf=sf,
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_ind=self.c_ind,
            dx=dx / length_scale,
            dy=dy / length_scale,
            dt=dt * self.parameters.v / length_scale,
            u_pt_ref=self.parameters.u_pt_ref,
            delta_u=self.parameters.delta_u,
            reynolds_number=self.parameters.reynolds_number,
            grashof_number=self.parameters.grashof_number,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
        )

        self._apply_boundary_conditions_x(time=time)

        self._new_w = np.copy(w)

        self._solve_sweep_x(
            n=n_y,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            result=self._new_w,
        )

        self._compute_sweep_y_coeff(
            w=self._new_w,
            sf=sf,
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_ind=self.c_ind,
            dx=dx / length_scale,
            dy=dy / length_scale,
            dt=dt * self.parameters.v / length_scale,
            u_pt_ref=self.parameters.u_pt_ref,
            delta_u=self.parameters.delta_u,
            reynolds_number=self.parameters.reynolds_number,
            grashof_number=self.parameters.grashof_number,
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
        )

        self._apply_boundary_conditions_y(time=time)

        self._solve_sweep_y(
            n=n_x,
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            result=self._new_w,
        )

        return self._new_w
