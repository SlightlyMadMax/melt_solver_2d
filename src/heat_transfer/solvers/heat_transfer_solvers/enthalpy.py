import math

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.core.boundary_conditions import BoundaryConditionType
from src.core.geometry import DomainGeometry
from src.heat_transfer.solvers.heat_transfer_solvers.base import (
    ImplicitHeatTransferSolver,
)
from src.parameters.material_properties import MaterialProperties


class EnthalpySolver(ImplicitHeatTransferSolver):
    @staticmethod
    @njit
    def _compute_sweep_x_coeff(
        u: NDArray[np.float64],
        u_iter: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_eff_old: NDArray[np.float64],
        c_eff_new: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        s_old: NDArray[np.float64],
        s_new: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        peclet_number: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u.shape
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy
        inv_peclet_number = 1.0 / peclet_number
        dt_half = 0.5 * dt

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff_new = 1.0 / c_eff_new[j, i]
                k_ip1j = 0.5 * (k_eff[j, i] + k_eff[j, i + 1])
                k_im1j = 0.5 * (k_eff[j, i] + k_eff[j, i - 1])
                k_ijp1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                # Coefficient at T_{i + 1, j}^{n + 1/2}
                a[j, i] = (
                    dt_half
                    * inv_c_eff_new
                    * (
                        c_eff_new[j, i + 1] * conv_x[j, i, 0]
                        - k_ip1j * inv_peclet_number * inv_dx2
                    )
                )

                # Coefficient at T_{i, j}^{n + 1/2}
                b[j, i] = 1.0 + dt_half * inv_c_eff_new * (
                    conv_x[j, i, 1] + (k_ip1j + k_im1j) * inv_peclet_number * inv_dx2
                )

                # Coefficient at T_{i - 1, j}^{n + 1/2}
                c[j, i] = (
                    dt_half
                    * inv_c_eff_new
                    * (
                        c_eff_new[j, i - 1] * conv_x[j, i, 2]
                        - k_im1j * inv_peclet_number * inv_dx2
                    )
                )

                # Right-hand side of the equation
                rhs[j, i] = (
                    u[j, i]
                    + 0.5 * (s_new[j, i] - s_old[j, i])
                    + 0.5
                    * inv_c_eff_new
                    * (s_new[j, i] + u_iter[j, i])
                    * (c_eff_new[j, i] - c_eff_old[j, i])
                    + dt_half
                    * inv_c_eff_new
                    * (
                        inv_dy2
                        * inv_peclet_number
                        * (
                            k_ijp1 * (u[j + 1, i] - u[j, i])
                            - k_ijm1 * (u[j, i] - u[j - 1, i])
                        )
                        - (
                            c_eff_new[j + 1, i] * conv_y[j, i, 0] * u[j + 1, i]
                            + conv_y[j, i, 1] * u[j, i]
                            + c_eff_new[j - 1, i] * conv_y[j, i, 2] * u[j - 1, i]
                        )
                    )
                )

    def _apply_boundary_conditions_x(
        self,
        time: float,
        delta: NDArray[np.float64],
    ) -> None:
        done = self._apply_standard_bc(
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            bc=self.bcs.left,
            side=0,
            time=time,
            k_eff_slice=self._k_eff[:, 0],
        )
        if not done and self.bcs.left.boundary_type == BoundaryConditionType.NEUMANN:
            self._apply_second_order_left(time, delta)

        done = self._apply_standard_bc(
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            bc=self.bcs.right,
            side=1,
            time=time,
            k_eff_slice=self._k_eff[:, -1],
        )
        if not done and self.bcs.right.boundary_type == BoundaryConditionType.NEUMANN:
            self._apply_second_order_right(time, delta)

    @staticmethod
    @njit
    def _compute_sweep_y_coeff(
        u: NDArray[np.float64],
        u_iter: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_eff_old: NDArray[np.float64],
        c_eff_new: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        s_old: NDArray[np.float64],
        s_new: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        peclet_number: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u.shape
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy
        inv_peclet_number = 1.0 / peclet_number
        dt_half = 0.5 * dt

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff_new = 1.0 / c_eff_new[j, i]
                k_i1pj = 0.5 * (k_eff[j, i] + k_eff[j, i + 1])
                k_im1j = 0.5 * (k_eff[j, i] + k_eff[j, i - 1])
                k_ijp1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                # Coefficient at T_{i, j + 1}^{n + 1}
                a[i, j] = (
                    dt_half
                    * inv_c_eff_new
                    * (
                        c_eff_new[j + 1, i] * conv_y[j, i, 0]
                        - k_ijp1 * inv_peclet_number * inv_dy2
                    )
                )

                # Coefficient at T_{i, j}^{n + 1}
                b[i, j] = 1.0 + dt_half * inv_c_eff_new * (
                    conv_y[j, i, 1] + (k_ijp1 + k_ijm1) * inv_peclet_number * inv_dy2
                )

                # Coefficient at T_{i, j - 1}^{n + 1}
                c[i, j] = (
                    dt_half
                    * inv_c_eff_new
                    * (
                        c_eff_new[j - 1, i] * conv_y[j, i, 2]
                        - k_ijm1 * inv_peclet_number * inv_dy2
                    )
                )

                # Right-hand side of the equation
                rhs[i, j] = (
                    u[j, i]
                    + 0.5 * (s_new[j, i] - s_old[j, i])
                    + 0.5
                    * inv_c_eff_new
                    * (s_new[j, i] + u_iter[j, i])
                    * (c_eff_new[j, i] - c_eff_old[j, i])
                    + dt_half
                    * inv_c_eff_new
                    * (
                        inv_dx2
                        * inv_peclet_number
                        * (
                            k_i1pj * (u[j, i + 1] - u[j, i])
                            - k_im1j * (u[j, i] - u[j, i - 1])
                        )
                        - (
                            c_eff_new[j, i + 1] * conv_x[j, i, 0] * u[j, i + 1]
                            + conv_x[j, i, 1] * u[j, i]
                            + c_eff_new[j, i - 1] * conv_x[j, i, 2] * u[j, i - 1]
                        )
                    )
                )

    def _apply_boundary_conditions_y(
        self, time: float, delta: NDArray[np.float64]
    ) -> None:
        done = self._apply_standard_bc(
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            bc=self.bcs.bottom,
            side=0,
            time=time,
            k_eff_slice=self._k_eff[0, :],
        )
        if not done and self.bcs.bottom.boundary_type == BoundaryConditionType.NEUMANN:
            self._apply_second_order_bottom(time, delta)

        done = self._apply_standard_bc(
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            bc=self.bcs.top,
            side=1,
            time=time,
            k_eff_slice=self._k_eff[-1, :],
        )
        if not done and self.bcs.top.boundary_type == BoundaryConditionType.NEUMANN:
            self._apply_second_order_top(time, delta)

    def calculate_source_term(
        self, u: NDArray[np.float64], u_0: float, delta: float, stefan_number: float
    ):
        diff = u - u_0
        if abs(diff) < delta:
            return 0.5 * (1.0 + math.tanh(3.0 * diff / math.sqrt(delta**2 - diff**2)))
        return 1.0 / stefan_number if u > u_0 else 0.0

    def solve_linear(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        delta: float,
        time: float = 0.0,
    ) -> None:
        geometry: DomainGeometry = self.cfg.geometry
        props: MaterialProperties = self.cfg.material_props
        n_x, n_y = geometry.n_x, geometry.n_y
        dx, dy, dt = geometry.dx, geometry.dy, geometry.dt
        dx_scaled = dx / self.cfg.l
        dy_scaled = dy / self.cfg.l
        dt_scaled = dt * self.cfg.v / self.cfg.l

        self.convective_operator(
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            sf=sf,
            u=u * self.cfg.delta_u + self.cfg.u_ref,
            u_pt=self.cfg.material_props.u_pt,
        )
        u_dim = self._iter_u * self.cfg.delta_u + self.cfg.u_ref

        self.compute_effective_properties(
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            u=self._iter_u,
            u_pt_non_dim=self.cfg.u_pt_non_dim,
            c_ref=self.cfg.volumetric_heat_capacity_ref,
            c_solid=props.volumetric_heat_capacity_solid,
            c_liquid=props.volumetric_heat_capacity_liquid,
            l_solid=props.volumetric_latent_heat,
            k_ref=self.cfg.thermal_conductivity_ref,
            k_solid=props.thermal_conductivity_solid,
            k_liquid=props.thermal_conductivity_liquid,
            delta=delta,
        )

        self._compute_sweep_x_coeff(
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dx=dx_scaled,
            dy=dy_scaled,
            dt=dt_scaled,
            peclet_number=self.cfg.peclet_number,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
        )

        self._new_u = np.copy(u)

        self._apply_boundary_conditions_x(time=time, delta=delta)

        self._solve_sweep_x(
            n=n_y,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            result=self._new_u,
        )

        self._compute_sweep_y_coeff(
            u=self._new_u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dx=dx_scaled,
            dy=dy_scaled,
            dt=dt_scaled,
            peclet_number=self.cfg.peclet_number,
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
        )

        self._apply_boundary_conditions_y(time=time, delta=delta)

        self._solve_sweep_y(
            n=n_x,
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            result=self._new_u,
        )
