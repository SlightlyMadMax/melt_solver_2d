import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.core.boundary_conditions import BoundaryConditionType
from src.heat_transfer.coefficient_smoothing.delta import get_max_delta
from src.heat_transfer.solvers.heat_transfer_solvers.base import (
    ExplicitHeatTransferSolver,
)
from src.heat_transfer.solvers.heat_transfer_solvers.registry import (
    HeatTransferSolverName,
    register_solver,
)


@register_solver(HeatTransferSolverName.EXPLICIT)
class ExplicitHeatSolver(ExplicitHeatTransferSolver):
    @staticmethod
    @njit
    def _compute_temperature(
        u: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        result: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        peclet_number: float,
        right_value: NDArray[np.float64] = None,
        left_value: NDArray[np.float64] = None,
    ) -> NDArray[np.float64]:
        n_y, n_x = u.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy
        inv_peclet_number = 1.0 / peclet_number

        result[0, :] = result[1, :]  # Adiabatic top wall
        result[-1, :] = result[-2, :]  # Adiabatic bottom wall
        result[:, 0] = left_value  # Cold right wall
        result[:, -1] = right_value  # Hot left wall

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]
                k_i1j = 0.5 * (k_eff[j, i] + k_eff[j, i + 1])
                k_im1j = 0.5 * (k_eff[j, i] + k_eff[j, i - 1])
                k_ij1 = 0.5 * (k_eff[j, i] + k_eff[j + 1, i])
                k_ijm1 = 0.5 * (k_eff[j, i] + k_eff[j - 1, i])

                advection_x = (
                    conv_x[j, i, 0] * u[j, i + 1]
                    + conv_x[j, i, 1] * u[j, i]
                    + conv_x[j, i, 2] * u[j, i - 1]
                )

                advection_y = (
                    conv_y[j, i, 0] * u[j + 1, i]
                    + conv_y[j, i, 1] * u[j, i]
                    + conv_y[j, i, 2] * u[j - 1, i]
                )

                diffusion = (
                    inv_c_eff
                    * inv_peclet_number
                    * (
                        inv_dx
                        * (
                            k_i1j * inv_dx * (u[j, i + 1] - u[j, i])
                            - k_im1j * inv_dx * (u[j, i] - u[j, i - 1])
                        )
                        + inv_dy
                        * (
                            k_ij1 * inv_dy * (u[j + 1, i] - u[j, i])
                            - k_ijm1 * inv_dy * (u[j - 1, i] - u[j, i])
                        )
                    )
                )

                result[j, i] = u[j, i] + dt * (-advection_x - advection_y + diffusion)

        return result

    def solve_linear(
        self, u: NDArray[np.float64], sf: NDArray[np.float64], time: float = 0.0
    ) -> None:
        convection_x, convection_y = self.convective_operator(
            sf=sf,
            u=u * self.parameters.delta_u + self.parameters.u_ref,
            u_pt=self.parameters.u_pt,
        )
        delta = (
            self.parameters.delta
            if self.fixed_delta
            else get_max_delta(
                u=self._iter_u * self.parameters.delta_u + self.parameters.u_ref,
                u_pt=self.parameters.u_pt,
            )
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

        self._compute_temperature(
            u=u,
            conv_x=convection_x,
            conv_y=convection_y,
            result=self._new_u,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
            dt=self.geometry.dt * self.parameters.v / self.geometry.length_scale,
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            peclet_number=self.parameters.peclet_number,
            right_value=(
                self.bcs.right.get_value(t=time)
                if self.bcs.right.boundary_type == BoundaryConditionType.DIRICHLET
                else None
            ),
            left_value=(
                self.bcs.left.get_value(t=time)
                if self.bcs.left.boundary_type == BoundaryConditionType.DIRICHLET
                else None
            ),
        )
