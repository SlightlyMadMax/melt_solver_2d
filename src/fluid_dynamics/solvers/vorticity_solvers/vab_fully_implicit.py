from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import splu

from src.convective_operators import VorticityBasedConvectiveOperator
from src.core.geometry import DomainGeometry
from src.core.solvers.base_solver import BaseSolver
from src.parameters.config import ExperimentConfig


class VabFullyImplicitScheme(BaseSolver):
    def __init__(
        self,
        cfg: ExperimentConfig,
        *args,
        **kwargs,
    ):
        super().__init__(cfg=cfg)

        self.cfg = cfg
        self.convective_operator = VorticityBasedConvectiveOperator(cfg=self.cfg)

        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        # Pre-allocate some arrays that will be used in the calculations
        self._new_w: NDArray[np.float64] = np.zeros((n_y, n_x))
        self._conv_x: NDArray[np.float64] = np.empty((n_y, n_x, 3))
        self._conv_y: NDArray[np.float64] = np.empty((n_y, n_x, 3))
        self.penalty_term: NDArray[np.float64] = np.empty((n_y, n_x))
        self._rhs: NDArray[np.float64] = np.empty((n_y - 2) * (n_x - 2))
        self._implicit_matrix: csr_matrix = self._precompute_matrix()
        self.lu = splu(self._implicit_matrix.tocsc())
        self.rho = self.calculate_rho()

    def calculate_rho(self):
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        dx_scaled, dy_scaled, _ = self.cfg.scaled_grid_steps

        rho = np.zeros((n_y, n_x))

        rho[2 : n_y - 2, 1] = 2 * dx_scaled**-4
        rho[2 : n_y - 2, n_x - 2] = 2 * dx_scaled**-4

        rho[1, 2 : n_x - 2] = 2 * dy_scaled**-4
        rho[n_y - 2, 2 : n_x - 2] = 2 * dy_scaled**-4

        rho[1, 1] = 2 * (dx_scaled**-4 + dy_scaled**-4)
        rho[1, n_x - 2] = 2 * (dx_scaled**-4 + dy_scaled**-4)
        rho[n_y - 2, 1] = 2 * (dx_scaled**-4 + dy_scaled**-4)
        rho[n_y - 2, n_x - 2] = 2 * (dx_scaled**-4 + dy_scaled**-4)

        return rho

    def _precompute_matrix(self) -> csr_matrix:
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        inner_n_y, inner_n_x = n_y - 2, n_x - 2
        size = inner_n_x * inner_n_y
        dx, dy, tau = self.cfg.scaled_grid_steps
        pr = self.cfg.prandtl_number

        main_diag = -2 / dx**2 - 2 / dy**2
        off_diag_x = 1 / dx**2
        off_diag_y = 1 / dy**2

        main_diag = np.full(size, pr * main_diag)
        x_off_diag = np.full(size - 1, pr * off_diag_x)
        y_off_diag = np.full(size - inner_n_x, pr * off_diag_y)

        x_off_diag[np.arange(1, size) % inner_n_x == 0] = 0

        diagonals = [main_diag, x_off_diag, x_off_diag, y_off_diag, y_off_diag]
        offsets = [0, -1, 1, -inner_n_x, inner_n_x]

        A = diags(diagonals, offsets, shape=(size, size), format="csr")

        return (1 / tau) * diags([1.0], [0], shape=(size, size)) - A

    def _calculate_penalty_term_coeff(self, u: np.ndarray, delta: float) -> None:
        u_pt = self.cfg.u_pt_nd
        eps = self.cfg.epsilon
        inv_eps2 = 1.0 / (eps * eps)
        diff_u = u - u_pt

        # --- Variant 1: sharp step ----------------------
        # self.penalty_term[:, :] = np.where(u <= u_pt, inv_eps2, 0.0)

        # --- Variant 2: error‐function form -------------------
        # f_l = 0.5 * (1.0 + erf(diff_u / (np.sqrt(2.0) * delta)))
        # self.penalty_term[:, :] = inv_eps2 * (1.0 - f_l) ** 2 / (f_l**3 + 1e-6)
        # self.penalty_term[:, :] = 0.5 * inv_eps2 * (1.0 - erf(diff_u / (np.sqrt(2.0) * delta)))

        # --- Variant 3: hyperbolic‐tangent form ---------------
        self.penalty_term[:, :] = 0.5 * inv_eps2 * (1.0 - np.tanh(diff_u / delta))

        # --- Variant 4: exponential form (one-sided smoothing) ----------------------
        # exp_term = np.exp((delta - diff_u) / delta)
        # temp = inv_eps2 * 0.5 * (2.0 + exp_term / (0.5 - exp_term))
        # self.penalty_term[:, :] = np.where(diff_u <= 0, temp, 0.0)

    def solve(
        self,
        w: NDArray[np.float64],
        conv_w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        delta: Optional[float] = None,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        geometry: DomainGeometry = self.cfg.geometry
        n_y, n_x = geometry.n_y, geometry.n_x
        dx, dy, tau = self.cfg.scaled_grid_steps
        pr = self.cfg.prandtl_number
        ra = self.cfg.rayleigh_number
        inv_dx = 1.0 / dx
        inner_n_y, inner_n_x = n_y - 2, n_x - 2

        self.convective_operator(w=conv_w, conv_x=self._conv_x, conv_y=self._conv_y)
        u_dim = u * self.cfg.delta_u + self.cfg.u_ref
        self._calculate_penalty_term_coeff(u=u_dim, delta=delta or self.cfg.delta_nd)

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
                self._rhs[idx] = (
                    w[j, i] / tau
                    - conv_term
                    - (pr * self.rho[j, i] + self.penalty_term[j, i]) * sf[j, i]
                    + pr * ra * dudx
                )

        omega_inner_flat = self.lu.solve(self._rhs)

        self._new_w[1:-1, 1:-1] = omega_inner_flat.reshape((inner_n_y, inner_n_x))

        return self._new_w
