import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.core.boundary_conditions import BoundaryConditionType
from src.heat_transfer.coefficient_smoothing.mushy_zone import get_mushy_zone_width
from src.heat_transfer.solvers.heat_transfer_solvers.base import (
    ImplicitHeatTransferSolver,
)
from src.heat_transfer.solvers.heat_transfer_solvers.registry import (
    HeatTransferSolverName,
    register_solver,
)


@register_solver(HeatTransferSolverName.DOUGLAS_RACHFORD)
class DouglasRachfordSolver(ImplicitHeatTransferSolver):
    @staticmethod
    @njit
    def _compute_sweep_x_coeff(
        u: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        peclet_number: float,
        rhs: NDArray[np.float64],
        a_x: NDArray[np.float64],
        b_x: NDArray[np.float64],
        c_x: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u.shape
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy
        inv_peclet_number = 1.0 / peclet_number

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_i1j = 0.5 * (k_eff[j, i] + k_eff[j, i + 1])
                k_im1j = 0.5 * (k_eff[j, i] + k_eff[j, i - 1])
                k_ij1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                # Coefficient at T_{i + 1, j}^{n + 1/2}
                a_x[j, i] = dt * (
                    conv_x[j, i, 0] - k_i1j * inv_peclet_number * inv_c_eff * inv_dx2
                )

                # Coefficient at T_{i, j}^{n + 1/2}
                b_x[j, i] = 1.0 + dt * (
                    conv_x[j, i, 1]
                    + (k_i1j + k_im1j) * inv_peclet_number * inv_c_eff * inv_dx2
                )

                # Coefficient at T_{i - 1, j}^{n + 1/2}
                c_x[j, i] = dt * (
                    conv_x[j, i, 2] - k_im1j * inv_peclet_number * inv_c_eff * inv_dx2
                )

                rhs[j, i] = u[j, i] + dt * inv_c_eff * (
                    inv_dy2
                    * inv_peclet_number
                    * (
                        k_ij1 * (u[j + 1, i] - u[j, i])
                        - k_ijm1 * (u[j, i] - u[j - 1, i])
                    )
                    - (
                        conv_y[j, i, 0] * u[j + 1, i]
                        + conv_y[j, i, 1] * u[j, i]
                        + conv_y[j, i, 2] * u[j - 1, i]
                    )
                )

    def _apply_boundary_conditions_x(self, time: float) -> None:
        if self.bcs.left.boundary_type == BoundaryConditionType.DIRICHLET:
            self.apply_dirichlet(
                a=self._a_x,
                b=self._b_x,
                c=self._c_x,
                rhs=self._rhs_x,
                value=self.bcs.left.get_value(t=time),
                side=0,
            )
        elif self.bcs.left.boundary_type == BoundaryConditionType.NEUMANN:
            k = self._k_eff[:, 0] * self.parameters.thermal_conductivity_ref
            self.apply_neumann_first_order(
                a=self._a_x,
                b=self._b_x,
                c=self._c_x,
                rhs=self._rhs_x,
                flux=self.bcs.left.get_flux(t=time) / k,
                side=0,
            )
        else:
            raise NotImplementedError("Boundary condition type not implemented")

        if self.bcs.right.boundary_type == BoundaryConditionType.DIRICHLET:
            self.apply_dirichlet(
                a=self._a_x,
                b=self._b_x,
                c=self._c_x,
                rhs=self._rhs_x,
                value=self.bcs.right.get_value(t=time),
                side=1,
            )
        elif self.bcs.right.boundary_type == BoundaryConditionType.NEUMANN:
            k = self._k_eff[:, -1] * self.parameters.thermal_conductivity_ref
            self.apply_neumann_first_order(
                a=self._a_x,
                b=self._b_x,
                c=self._c_x,
                rhs=self._rhs_x,
                flux=self.bcs.right.get_flux(t=time) / k,
                side=1,
            )
        else:
            raise NotImplementedError("Boundary condition type not implemented")

    @staticmethod
    @njit
    def _compute_sweep_y_coeff(
        u_old: NDArray[np.float64],
        u_prev: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        dy: float,
        dt: float,
        peclet_number: float,
        a_y: NDArray[np.float64],
        b_y: NDArray[np.float64],
        c_y: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> None:
        n_y, n_x = u_old.shape
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy
        inv_peclet_number = 1.0 / peclet_number

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_ij1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                # Coefficient at T_{i, j + 1}^{n + 1}
                a_y[i, j] = dt * (
                    conv_y[j, i, 0] - k_ij1 * inv_peclet_number * inv_c_eff * inv_dy2
                )

                # Coefficient at T_{i, j}^{n + 1}
                b_y[i, j] = 1.0 + dt * (
                    conv_y[j, i, 1]
                    + (k_ij1 + k_ijm1) * inv_peclet_number * inv_c_eff * inv_dy2
                )

                # Coefficient at T_{i, j - 1}^{n + 1}
                c_y[i, j] = dt * (
                    conv_y[j, i, 2] - k_ijm1 * inv_peclet_number * inv_c_eff * inv_dy2
                )

                # Right-hand side of the equation
                rhs[i, j] = u_prev[j, i] - dt * inv_c_eff * (
                    inv_dy2
                    * inv_peclet_number
                    * (
                        k_ij1 * (u_old[j + 1, i] - u_old[j, i])
                        - k_ijm1 * (u_old[j, i] - u_old[j - 1, i])
                    )
                    - (
                        conv_y[j, i, 0] * u_old[j + 1, i]
                        + conv_y[j, i, 1] * u_old[j, i]
                        + conv_y[j, i, 2] * u_old[j - 1, i]
                    )
                )

    def _apply_boundary_conditions_y(self, time: float) -> None:
        if self.bcs.bottom.boundary_type == BoundaryConditionType.DIRICHLET:
            self.apply_dirichlet(
                a=self._a_y,
                b=self._b_y,
                c=self._c_y,
                rhs=self._rhs_y,
                value=self.bcs.bottom.get_value(t=time),
                side=0,
            )
        elif self.bcs.bottom.boundary_type == BoundaryConditionType.NEUMANN:
            k = self._k_eff[0, :] * self.parameters.thermal_conductivity_ref
            self.apply_neumann_first_order(
                a=self._a_y,
                b=self._b_y,
                c=self._c_y,
                rhs=self._rhs_y,
                flux=self.bcs.bottom.get_flux(t=time) / k,
                side=0,
            )
        else:
            raise NotImplementedError("Boundary condition type not implemented")

        if self.bcs.top.boundary_type == BoundaryConditionType.DIRICHLET:
            self.apply_dirichlet(
                a=self._a_y,
                b=self._b_y,
                c=self._c_y,
                rhs=self._rhs_y,
                value=self.bcs.top.get_value(t=time),
                side=1,
            )
        elif self.bcs.top.boundary_type == BoundaryConditionType.NEUMANN:
            k = self._k_eff[-1, :] * self.parameters.thermal_conductivity_ref
            self.apply_neumann_first_order(
                a=self._a_y,
                b=self._b_y,
                c=self._c_y,
                rhs=self._rhs_y,
                flux=self.bcs.top.get_flux(t=time) / k,
                side=1,
            )
        else:
            raise NotImplementedError("Boundary condition type not implemented")

    def solve_linear(
        self, u: NDArray[np.float64], sf: NDArray[np.float64], time: float = 0.0
    ) -> None:
        dx, dy = self.geometry.dx, self.geometry.dy
        n_x, n_y = self.geometry.n_x, self.geometry.n_y
        self.convective_operator(
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            sf=sf,
            u=u * self.parameters.delta_u + self.parameters.u_ref,
            u_pt=self.parameters.u_pt,
        )
        dim_u = self._iter_u * self.parameters.delta_u + self.parameters.u_ref
        delta = get_mushy_zone_width(
            u=dim_u,
            u_pt=self.parameters.u_pt,
            h_x=dx,
            h_y=dy,
        )

        self.compute_effective_properties(
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            u=self._iter_u,
            u_ref=self.parameters.u_ref,
            u_pt=self.parameters.u_pt,
            delta_u=self.parameters.delta_u,
            c_ref=self.parameters.volumetric_heat_capacity_ref,
            c_solid=self.parameters.volumetric_heat_capacity_solid,
            c_liquid=self.parameters.volumetric_heat_capacity_liquid,
            l_solid=self.parameters.volumetric_latent_heat,
            k_ref=self.parameters.thermal_conductivity_ref,
            k_solid=self.parameters.thermal_conductivity_solid,
            k_liquid=self.parameters.thermal_conductivity_liquid,
            delta=delta,
        )

        self._compute_sweep_x_coeff(
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
            dt=self.geometry.dt * self.parameters.v / self.geometry.length_scale,
            peclet_number=self.parameters.peclet_number,
            a_x=self._a_x,
            b_x=self._b_x,
            c_x=self._c_x,
            rhs=self._rhs_x,
        )

        self._apply_boundary_conditions_x(time=time)

        self._new_u = np.copy(u)

        self._solve_sweep_x(
            n_y=n_y,
            a_x=self._a_x,
            b_x=self._b_x,
            c_x=self._c_x,
            rhs_x=self._rhs_x,
            result=self._new_u,
        )

        self._compute_sweep_y_coeff(
            u_old=u,
            u_prev=self._new_u,
            conv_y=self._conv_y,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            dy=dy / self.geometry.length_scale,
            dt=self.geometry.dt * self.parameters.v / self.geometry.length_scale,
            peclet_number=self.parameters.peclet_number,
            a_y=self._a_y,
            b_y=self._b_y,
            c_y=self._c_y,
            rhs=self._rhs_y,
        )

        self._apply_boundary_conditions_y(time=time)

        self._solve_sweep_y(
            n_x=n_x,
            a_y=self._a_y,
            b_y=self._b_y,
            c_y=self._c_y,
            rhs_y=self._rhs_y,
            result=self._new_u,
        )
