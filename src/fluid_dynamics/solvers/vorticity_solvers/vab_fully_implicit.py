import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from src.convective_operators import EffectiveSFTransportOperator
from src.core.geometry import DomainGeometry
from src.core.solvers.base_solver import BaseSolver
from src.fluid_dynamics.utils import calculate_indicator_function
from src.heat_transfer.coefficient_smoothing.mushy_zone import get_mushy_zone_width
from src.parameters.fluid import FluidParameters


class VabFullyImplicitScheme(BaseSolver):
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        *args,
        **kwargs,
    ):
        super().__init__(geometry=geometry)

        self.parameters = parameters
        self.convective_operator = EffectiveSFTransportOperator(geometry=geometry)

        # Pre-allocate some arrays that will be used in the calculations
        self._new_w: NDArray[np.float64] = np.zeros(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._conv_x: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x, 3)
        )
        self._conv_y: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x, 3)
        )
        self.c_ind: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self.rho = self.calculate_rho()

    def calculate_rho(self):
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        dy, dx = (
            self.geometry.dy / self.parameters.l,
            self.geometry.dx / self.parameters.l,
        )

        rho = np.zeros((n_y, n_x))

        rho[2 : n_y - 2, 1] = 2 * dx**-4
        rho[2 : n_y - 2, n_x - 2] = 2 * dx**-4

        rho[1, 2 : n_x - 2] = 2 * dy**-4
        rho[n_y - 2, 2 : n_x - 2] = 2 * dy**-4

        rho[1, 1] = 2 * (dx**-4 + dy**-4)
        rho[1, n_x - 2] = 2 * (dx**-4 + dy**-4)
        rho[n_y - 2, 1] = 2 * (dx**-4 + dy**-4)
        rho[n_y - 2, n_x - 2] = 2 * (dx**-4 + dy**-4)

        return rho

    def solve(
        self,
        w: NDArray[np.float64],
        conv_w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        self.convective_operator(w=conv_w, conv_x=self._conv_x, conv_y=self._conv_y)
        u_dim = u * self.parameters.delta_u + self.parameters.u_ref
        delta = get_mushy_zone_width(
            u=u_dim,
            u_pt=self.parameters.u_pt,
            h_x=self.geometry.dx,
            h_y=self.geometry.dy,
        )
        calculate_indicator_function(
            u=u_dim,
            u_pt=self.parameters.u_pt,
            eps=self.parameters.epsilon,
            delta=delta,
            result=self.c_ind,
        )
        self.c_ind *= self.geometry.length_scale**3 / self.parameters.v
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        inner_n_y, inner_n_x = n_y - 2, n_x - 2
        size = inner_n_x * inner_n_y

        dx = self.geometry.dx / self.geometry.length_scale
        dy = self.geometry.dy / self.geometry.length_scale
        inv_dx = 1.0 / dx

        inv_re = 1.0 / self.parameters.reynolds_number
        inv_re2 = inv_re * inv_re

        tau = self.geometry.dt * self.parameters.v / self.geometry.length_scale

        gr = np.where(
            u * self.parameters.delta_u - self.parameters.u_pt_ref < 0.0,
            0.0,
            self.parameters.grashof_number,
        )

        main_diag = -2 / dx**2 - 2 / dy**2
        off_diag_x = 1 / dx**2
        off_diag_y = 1 / dy**2

        diagonals = [
            np.full(size, inv_re * main_diag),  # main
            np.full(size - 1, inv_re * off_diag_x),  # left/right
            np.full(size - 1, inv_re * off_diag_x),
            np.full(size - inner_n_x, inv_re * off_diag_y),  # top/bottom
            np.full(size - inner_n_x, inv_re * off_diag_y),
        ]
        offsets = [0, -1, 1, -inner_n_x, inner_n_x]

        # Correct for block structure in x-direction
        for i in range(1, inner_n_y):
            diagonals[1][i * inner_n_x - 1] = 0.0  # left of block
            diagonals[2][i * inner_n_x - 1] = 0.0  # right of block

        A = diags(diagonals, offsets, shape=(size, size), format="csr")

        rhs = np.zeros(size)
        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                idx = (j - 1) * inner_n_x + (i - 1)
                conv_term = (
                    self._conv_x[j, i, 0] * sf[j, i + 1]
                    + self._conv_x[j, i, 2] * sf[j, i - 1]
                    + self._conv_y[j, i, 0] * sf[j + 1, i]
                    + self._conv_y[j, i, 2] * sf[j - 1, i]
                )
                dudx = 0.5 * inv_dx * (u[j, i + 1] - u[j, i - 1])
                rhs[idx] = (
                    w[j, i] / tau
                    - conv_term
                    - (inv_re * self.rho[j, i] + self.c_ind[j, i]) * sf[j, i]
                    + gr[j, i] * inv_re2 * dudx
                )

        # Solve the system
        omega_inner_flat = spsolve(
            (1 / tau) * diags([1.0], [0], shape=(size, size)) - A, rhs
        )

        # Embed into full omega array
        self._new_w[1:-1, 1:-1] = omega_inner_flat.reshape((inner_n_y, inner_n_x))

        return self._new_w
