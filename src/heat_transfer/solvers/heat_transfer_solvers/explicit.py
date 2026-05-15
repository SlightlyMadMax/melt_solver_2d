import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.core.boundary_conditions import BoundaryConditionType
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import BaseHeatSolver
from src.heat_transfer.solvers.heat_transfer_solvers.registry import (
    register_solver,
    HeatTransferSolverName,
)


@register_solver(HeatTransferSolverName.EXPLICIT)
class ExplicitHeatSolver(BaseHeatSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    @njit
    def _compute_temperature(
        u: NDArray[np.float64],
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        corr_x: NDArray[np.float64],
        corr_y: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        k_x: NDArray[np.float64],
        k_y: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        pe: float,
        result: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n_y, n_x = u.shape
        inv_pe_dx2 = 1.0 / (pe * dx * dx)
        inv_pe_dy2 = 1.0 / (pe * dy * dy)

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c_eff = 1.0 / c_eff[j, i]

                convection_x = (
                    conv_x[j, i, 0] * u[j, i + 1]
                    + conv_x[j, i, 1] * u[j, i]
                    + conv_x[j, i, 2] * u[j, i - 1]
                )
                convection_y = (
                    conv_y[j, i, 0] * u[j + 1, i]
                    + conv_y[j, i, 1] * u[j, i]
                    + conv_y[j, i, 2] * u[j - 1, i]
                )

                convection = convection_x + convection_y + corr_x[j, i] + corr_y[j, i]

                conduction_x = (
                    inv_c_eff
                    * inv_pe_dx2
                    * (
                        k_x[j, i + 1] * (u[j, i + 1] - u[j, i])
                        - k_x[j, i] * (u[j, i] - u[j, i - 1])
                    )
                )
                conduction_y = (
                    inv_c_eff
                    * inv_pe_dy2
                    * (
                        k_y[j + 1, i] * (u[j + 1, i] - u[j, i])
                        - k_y[j, i] * (u[j, i] - u[j - 1, i])
                    )
                )
                conduction = conduction_x + conduction_y

                result[j, i] = u[j, i] + dt * (conduction - convection)

        return result

    def apply_boundary_conditions(
        self,
        u: NDArray[np.float64],
        time: float,
    ) -> None:
        """
        Apply boundary conditions in-place to u.
        """
        dx, dy, _ = self.cfg.scaled_grid_steps
        bcs = self.bcs

        # --- TOP (j = ny - 1) ---
        bc = bcs.top
        if bc.boundary_type == BoundaryConditionType.DIRICHLET:
            u[-1, :] = bc.get_value(time)

        elif bc.boundary_type == BoundaryConditionType.NEUMANN:
            q = bc.get_flux(time)
            # du/dy = q  ->  (u_N - u_{N-1}) / dy = q
            u[-1, :] = u[-2, :] + dy * q

        # --- BOTTOM (j = 0) ---
        bc = bcs.bottom
        if bc.boundary_type == BoundaryConditionType.DIRICHLET:
            u[0, :] = bc.get_value(time)

        elif bc.boundary_type == BoundaryConditionType.NEUMANN:
            q = bc.get_flux(time)
            # du/dy = q  ->  (u_1 - u_0) / dy = q
            u[0, :] = u[1, :] - dy * q

        # --- RIGHT (i = nx - 1) ---
        bc = bcs.right
        if bc.boundary_type == BoundaryConditionType.DIRICHLET:
            u[:, -1] = bc.get_value(time)

        elif bc.boundary_type == BoundaryConditionType.NEUMANN:
            q = bc.get_flux(time)
            # du/dx = q
            u[:, -1] = u[:, -2] + dx * q

        # --- LEFT (i = 0) ---
        bc = bcs.left
        if bc.boundary_type == BoundaryConditionType.DIRICHLET:
            u[:, 0] = bc.get_value(time)

        elif bc.boundary_type == BoundaryConditionType.NEUMANN:
            q = bc.get_flux(time)
            u[:, 0] = u[:, 1] - dx * q

    def solve(
        self,
        u: NDArray[np.float64],
        delta: float,
        sf: NDArray[np.float64],
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        dx_scaled, dy_scaled, dt_scaled = self.cfg.scaled_grid_steps

        self.compute_effective_properties(u=u, delta=delta)

        self.convective_operator(
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            correction_x=self._correction_x,
            correction_y=self._correction_y,
            convected_quantity=u,
            sf=sf,
        )

        self._u_new[:, :] = u

        self._compute_temperature(
            u=u,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            corr_x=self._correction_x,
            corr_y=self._correction_y,
            c_eff=self._c_eff,
            k_x=self._k_x,
            k_y=self._k_y,
            dx=dx_scaled,
            dy=dy_scaled,
            dt=dt_scaled,
            pe=self.cfg.peclet_number,
            result=self._u_new,
        )

        self.apply_boundary_conditions(self._u_new, time)

        return self._u_new
