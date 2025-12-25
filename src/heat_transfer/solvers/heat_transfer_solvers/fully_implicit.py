from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from src.core.boundary_conditions import BoundaryConditionType
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import BaseHeatSolver
from src.heat_transfer.solvers.heat_transfer_solvers.registry import (
    register_solver,
    HeatTransferSolverName,
)


@register_solver(HeatTransferSolverName.FULLY_IMPLICIT)
class FullyImplicitSolver(BaseHeatSolver):

    def build_system(
        self, u_old: np.ndarray, time: float
    ) -> Tuple[csr_matrix, np.ndarray]:
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        dx, dy, tau = self.cfg.scaled_grid_steps
        total_points = n_y * n_x
        inv_tau = 1.0 / tau
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        inv_pe = 1.0 / self.cfg.peclet_number

        # Initialize sparse matrix (using LIL for efficient construction)
        A = lil_matrix((total_points, total_points))
        b = np.zeros(total_points)

        # Helper function to get linear index
        def linear_idx(j, i):
            return j * n_x + i

        # Build matrix for interior points
        for j in range(n_y):
            for i in range(n_x):
                idx = linear_idx(j, i)

                # Coefficients from time derivative and convection
                coeff_center = self._c_eff[j, i] * (
                    inv_tau + self._conv_x[j, i, 1] + self._conv_y[j, i, 1]
                )

                # Coefficients from diffusion in x-direction
                k_x_plus = self._k_x[j, i + 1]  # k_{j, i+1/2}
                k_x_minus = self._k_x[j, i]  # k_{j, i-1/2}

                coeff_center += inv_pe * inv_dx2 * (k_x_plus + k_x_minus)

                # Coefficients from diffusion in y-direction
                k_y_plus = self._k_y[j + 1, i]  # k_{j+1/2, i}
                k_y_minus = self._k_y[j, i]  # k_{j-1/2, i}

                coeff_center += inv_pe * inv_dy2 * (k_y_plus + k_y_minus)

                # Set center coefficient
                A[idx, idx] = coeff_center

                # Set east neighbor (i+1) if not at right boundary
                if i < n_x - 1:
                    idx_east = linear_idx(j, i + 1)
                    coeff_east = (
                        self._c_eff[j, i] * self._conv_x[j, i, 0]
                        - inv_pe * inv_dx2 * k_x_plus
                    )
                    A[idx, idx_east] = coeff_east

                # Set west neighbor (i-1) if not at left boundary
                if i > 0:
                    idx_west = linear_idx(j, i - 1)
                    coeff_west = (
                        self._c_eff[j, i] * self._conv_x[j, i, 2]
                        - inv_pe * inv_dx2 * k_x_minus
                    )
                    A[idx, idx_west] = coeff_west

                # Set north neighbor (j+1) if not at top boundary
                if j < n_y - 1:
                    idx_north = linear_idx(j + 1, i)
                    coeff_north = (
                        self._c_eff[j, i] * self._conv_y[j, i, 0]
                        - inv_pe * inv_dy2 * k_y_plus
                    )
                    A[idx, idx_north] = coeff_north

                # Set south neighbor (j-1) if not at bottom boundary
                if j > 0:
                    idx_south = linear_idx(j - 1, i)
                    coeff_south = (
                        self._c_eff[j, i] * self._conv_y[j, i, 2]
                        - inv_pe * inv_dy2 * k_y_minus
                    )
                    A[idx, idx_south] = coeff_south

                # Right-hand side from previous time step
                b[idx] = self._c_eff[j, i] * inv_tau * u_old[j, i]

        # Apply boundary conditions
        self._apply_boundary_conditions(A, b, self._k_eff, self._k_x, self._k_y, time)

        # Convert to CSR format for efficient solving
        return A.tocsr(), b

    def _apply_boundary_conditions(
        self,
        A: lil_matrix,
        b: np.ndarray,
        k: np.ndarray,
        k_x_faces: np.ndarray,
        k_y_faces: np.ndarray,
        time: float,
    ):
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        dx, dy, tau = self.cfg.scaled_grid_steps
        pe = self.cfg.peclet_number
        inv_pe = 1.0 / pe

        # Helper function to get linear index
        def linear_idx(j, i):
            return j * n_x + i

        # Apply left boundary condition
        if self.bcs.left.boundary_type == BoundaryConditionType.DIRICHLET:
            u_left = self.bcs.left.get_value(t=time)
            for j in range(n_y):
                idx = linear_idx(j, 0)
                # Set Dirichlet condition: T = T_left
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = u_left[j]

        elif self.bcs.left.boundary_type == BoundaryConditionType.NEUMANN:
            q_left = self.bcs.left.get_flux(t=time)
            inv_dx2 = 1.0 / (dx * dx)

            for j in range(n_y):
                idx = linear_idx(j, 0)

                # Modify equation for j,0 using ghost node expression
                # For Neumann: T_{-1,i} = T_{1,i} - (q_i * h_x) / k_{0,i}
                # This affects the coefficient for T_{1,i} and adds to RHS

                # Original west coefficient (for ghost node)
                coeff_ghost = -inv_pe * inv_dx2 * k_x_faces[j, 0]

                # Add contribution to T_{1,i} (east neighbor)
                idx_east = linear_idx(j, 1)
                A[idx, idx_east] += coeff_ghost

                # Add contribution to RHS from boundary flux
                b[idx] += coeff_ghost * (pe * q_left[j] * dx / k[j, 0])

        # Apply right boundary condition
        if self.bcs.right.boundary_type == BoundaryConditionType.DIRICHLET:
            u_right = self.bcs.right.get_value(t=time)

            for j in range(n_y):
                idx = linear_idx(j, n_x - 1)
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = u_right[j]

        elif self.bcs.right.boundary_type == BoundaryConditionType.NEUMANN:
            q_right = self.bcs.right.get_flux(t=time)
            inv_dx2 = 1.0 / (dx * dx)

            for j in range(n_y):
                idx = linear_idx(j, n_x - 1)

                # For right boundary: T_{n_x,i} = T_{n_x-2,i} - (q_i * h_x) / k_{n_x-1,i}
                coeff_ghost = -inv_pe * inv_dx2 * k_x_faces[j, n_x]

                # Add contribution to T_{n_x-2,i} (west neighbor of west neighbor)
                idx_west_west = linear_idx(j, n_x - 2)
                A[idx, idx_west_west] += coeff_ghost

                # Add contribution to RHS
                b[idx] += coeff_ghost * (pe * q_right[j] * dx / k[j, n_x - 1])

        # Apply bottom boundary condition
        if self.bcs.bottom.boundary_type == BoundaryConditionType.DIRICHLET:
            u_bottom = self.bcs.bottom.get_value(t=time)

            for i in range(n_x):
                idx = linear_idx(0, i)
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = u_bottom[i]

        elif self.bcs.bottom.boundary_type == BoundaryConditionType.NEUMANN:
            q_bottom = self.bcs.bottom.get_flux(t=time)
            inv_dy2 = 1.0 / (dy * dy)

            for i in range(n_x):
                idx = linear_idx(0, i)

                # For bottom boundary: T_{-1,i} = T_{1,i} - (q_i * h_y) / k_{0,i}
                coeff_ghost = -inv_pe * inv_dy2 * k_y_faces[0, i]

                # Add contribution to T_{1,i} (north neighbor)
                idx_north = linear_idx(1, i)
                A[idx, idx_north] += coeff_ghost

                # Add contribution to RHS
                b[idx] += coeff_ghost * (pe * q_bottom[i] * dy / k[0, i])

        # Apply top boundary condition
        if self.bcs.top.boundary_type == BoundaryConditionType.DIRICHLET:
            u_top = self.bcs.top.get_value(t=time)

            for i in range(n_x):
                idx = linear_idx(n_y - 1, i)
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = u_top[i]

        elif self.bcs.top.boundary_type == BoundaryConditionType.NEUMANN:
            q_top = self.bcs.top.get_flux(t=time)
            inv_dy2 = 1.0 / (dy * dy)

            for i in range(n_x):
                idx = linear_idx(n_y - 1, i)

                # For top boundary: T_{n_y,i} = T_{n_y-2,i} - (q_i * h_y) / k_{n_y-1,i}
                coeff_ghost = -inv_pe * inv_dy2 * k_y_faces[n_y, i]

                # Add contribution to T_{n_y-2,i} (south neighbor of south neighbor)
                idx_south_south = linear_idx(n_y - 2, i)
                A[idx, idx_south_south] += coeff_ghost

                # Add contribution to RHS
                b[idx] += coeff_ghost * (pe * q_top[i] * dy / k[n_y - 1, i])

    def solve(
        self,
        u: np.ndarray,
        delta: float,
        sf: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x

        self.compute_effective_properties(u=u, delta=delta)

        self.convective_operator(
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            correction_x=self._correction_x,
            correction_y=self._correction_y,
            convected_quantity=u,
            sf=sf,
        )

        # Build linear system
        A, b = self.build_system(u_old=u, time=time)

        # Solve sparse linear system
        u_flat = spsolve(A, b)

        # Reshape to 2D
        u_new = u_flat.reshape((n_y, n_x))

        return u_new
