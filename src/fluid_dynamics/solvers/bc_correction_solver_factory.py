import numpy as np

from typing import Tuple
from numpy.typing import NDArray
from scipy import sparse


from src.convective_operators import (
    ConvectiveTermForm,
    StreamFunctionBasedConvectiveOperator,
)
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.solvers.stream_function_solvers import (
    StreamFunctionSolverRegistry,
    StreamFunctionSolverName,
)
from src.fluid_dynamics.solvers.vorticity_solvers import (
    VorticitySolverRegistry,
    VorticitySolverName,
)
from src.fluid_dynamics.utils import calculate_vorticity_from_sf
from src.parameters.config import ExperimentConfig


class BCCorrectionNVSolver:
    def __init__(
        self,
        cfg: ExperimentConfig,
        sf_bcs: BoundaryConditions,
        sf_max_iters: int = 10000,
        sf_tolerance: float = 1e-6,
        convective_term_form: ConvectiveTermForm = ConvectiveTermForm.DIVERGENT_CENTRAL,
        vorticity_solver_name: VorticitySolverName = VorticitySolverName.PEACEMAN_RACHFORD,
        stream_function_solver_name: StreamFunctionSolverName = StreamFunctionSolverName.AMG,
        vorticity_bc_order: int = 1,
    ):
        self.cfg = cfg
        self.vorticity_bc_order = vorticity_bc_order
        self.convective_operator = StreamFunctionBasedConvectiveOperator(
            cfg=cfg, form=convective_term_form
        )
        n_y, n_x = cfg.geometry.n_y, cfg.geometry.n_x

        vorticity_solver_class = VorticitySolverRegistry.get_solver_class(
            solver_name=vorticity_solver_name
        )
        stream_function_solver_class = StreamFunctionSolverRegistry.get_solver_class(
            solver_name=stream_function_solver_name
        )

        self.vorticity_solver = vorticity_solver_class(
            cfg=cfg,
            convective_operator=self.convective_operator,
            bc_order=vorticity_bc_order,
        )
        self.stream_function_solver = stream_function_solver_class(
            cfg=cfg,
            bcs=sf_bcs,
            max_iters=sf_max_iters,
            stopping_criteria=sf_tolerance,
        )

        self._vorticity: NDArray[np.float64] = np.empty((n_y, n_x))
        self._stream_function: NDArray[np.float64] = np.empty((n_y, n_x))
        self.rho = self.calculate_rho_first_order()

    def calculate_rho_first_order(self):
        geometry: DomainGeometry = self.cfg.geometry
        n_y, n_x = geometry.n_y, geometry.n_x
        dx, dy, _ = self.cfg.scaled_grid_steps

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

    def apply_rho_to_psi_second_order(self, psi: np.ndarray) -> np.ndarray:
        geometry: DomainGeometry = self.cfg.geometry
        n_y, n_x = geometry.n_y, geometry.n_x
        dx, dy, _ = self.cfg.scaled_grid_steps

        res = np.zeros_like(psi)

        j_slice = slice(2, n_y - 2)
        i_left = 1
        res[j_slice, i_left] = 4.0 * psi[j_slice, i_left] / dx**4 - psi[
            j_slice, i_left + 1
        ] / (2.0 * dx**4)

        # right side: i = n_x - 2, neighbor is n_x - 3
        i_right = n_x - 2
        res[j_slice, i_right] = 4.0 * psi[j_slice, i_right] / dx**4 - psi[
            j_slice, i_right - 1
        ] / (2.0 * dx**4)

        # top side: j = 1, i = 2 .. n_x-3
        i_slice = slice(2, n_x - 2)
        j_top = 1
        res[j_top, i_slice] = 4.0 * psi[j_top, i_slice] / dy**4 - psi[
            j_top + 1, i_slice
        ] / (2.0 * dy**4)

        # bottom side: j = n_y - 2, neighbor is n_y - 3
        j_bot = n_y - 2
        res[j_bot, i_slice] = 4.0 * psi[j_bot, i_slice] / dy**4 - psi[
            j_bot - 1, i_slice
        ] / (2.0 * dy**4)

        # corners: combine x- and y- contributions
        # top-left (j=1, i=1)
        res[j_top, i_left] = (
            4.0 * psi[j_top, i_left] / dx**4
            - psi[j_top, i_left + 1] / (2.0 * dx**4)
            + 4.0 * psi[j_top, i_left] / dy**4
            - psi[j_top + 1, i_left] / (2.0 * dy**4)
        )

        # top-right (j=1, i=n_x-2)
        res[j_top, i_right] = (
            4.0 * psi[j_top, i_right] / dx**4
            - psi[j_top, i_right - 1] / (2.0 * dx**4)
            + 4.0 * psi[j_top, i_right] / dy**4
            - psi[j_top + 1, i_right] / (2.0 * dy**4)
        )

        # bottom-left (j=n_y-2, i=1)
        res[j_bot, i_left] = (
            4.0 * psi[j_bot, i_left] / dx**4
            - psi[j_bot, i_left + 1] / (2.0 * dx**4)
            + 4.0 * psi[j_bot, i_left] / dy**4
            - psi[j_bot - 1, i_left] / (2.0 * dy**4)
        )

        # bottom-right (j=n_y-2, i=n_x-2)
        res[j_bot, i_right] = (
            4.0 * psi[j_bot, i_right] / dx**4
            - psi[j_bot, i_right - 1] / (2.0 * dx**4)
            + 4.0 * psi[j_bot, i_right] / dy**4
            - psi[j_bot - 1, i_right] / (2.0 * dy**4)
        )

        return res

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        delta: float,
        time: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._vorticity[:, :] = w

        self._solve_vorticity(
            old_vorticity=self._vorticity,
            stream_function=sf,
            temperature=u,
            delta=delta,
            time=time,
        )
        self._solve_stream_function(
            sf_old=sf,
            vorticity=self._vorticity,
            time=time,
        )

        calculate_vorticity_from_sf(
            sf=self._stream_function, result=self._vorticity, cfg=self.cfg
        )
        return self._stream_function, self._vorticity

    def _solve_vorticity(
        self,
        old_vorticity: np.ndarray,
        stream_function: np.ndarray,
        temperature: np.ndarray,
        delta: float,
        time: float,
    ) -> None:
        self._vorticity[:, :] = self.vorticity_solver.solve(
            w=old_vorticity,
            sf=stream_function,
            u=temperature,
            delta=delta,
            time=time,
        )

    def _solve_stream_function(
        self,
        sf_old: np.ndarray,
        vorticity: np.ndarray,
        time: float,
    ) -> None:
        b = self._construct_rhs(
            vorticity=vorticity,
            sf_old=sf_old,
            px_half=self.vorticity_solver.px_half,
            py_half=self.vorticity_solver.py_half,
        )
        A = self._construct_matrix(
            px_half=self.vorticity_solver.px_half,
            py_half=self.vorticity_solver.py_half,
        )
        self._stream_function[:, :] = self.stream_function_solver.solve(
            initial_guess=sf_old,
            A=A,
            b_flat=b,
            time=time,
        )
        # psi_vec = self._stream_function[1:-1, 1:-1].ravel()
        # residual = (-A).dot(psi_vec) - (-b).ravel()
        # print("‖residual‖₂:", np.linalg.norm(residual, 2))

    def _construct_rhs(
        self,
        vorticity: np.ndarray,
        sf_old: np.ndarray,
        px_half: np.ndarray,
        py_half: np.ndarray,
    ) -> np.ndarray:
        dx, dy, tau = self.cfg.scaled_grid_steps
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        inv_re = 1.0 / self.cfg.reynolds_number

        # interior (i = 1..n_x-2, j = 1..n_y-2)
        psi = sf_old[1:-1, 1:-1]  # shape (n_y-2, n_x-2)
        w = vorticity[1:-1, 1:-1]
        if self.vorticity_bc_order == 1:
            r = self.rho[1:-1, 1:-1] * psi
        else:  # second order bc
            r = self.apply_rho_to_psi_second_order(sf_old)[1:-1, 1:-1]

        # X-direction: px_half has shape (n_y, n_x-1)
        # we need px_half[j, i] and px_half[j, i-1] for i=1..n_x-2, j=1..n_y-2
        px_i = px_half[1:-1, 1:]  # selects columns 1..(n_x-2) -> shape (n_y-2, n_x-2)
        px_im1 = px_half[
            1:-1, :-1
        ]  # selects columns 0..(n_x-3) -> shape (n_y-2, n_x-2)

        sf_x_fwd = sf_old[1:-1, 2:]  # sf[j, i+1]
        sf_x = psi  # sf[j, i]
        sf_x_bak = sf_old[1:-1, 0:-2]  # sf[j, i-1]

        term_x = px_i * (sf_x_fwd - sf_x) - px_im1 * (sf_x - sf_x_bak)

        # Y-direction: py_half has shape (n_y-1, n_x)
        # we need py_half[j, i] and py_half[j-1, i] for j=1..n_y-2, i=1..n_x-2
        py_j = py_half[1:, 1:-1]  # rows 1..(n_y-2), cols 1..(n_x-2) -> (n_y-2, n_x-2)
        py_jm1 = py_half[:-1, 1:-1]  # rows 0..(n_y-3), cols 1..(n_x-2)

        sf_y_fwd = sf_old[2:, 1:-1]  # sf[j+1, i]
        sf_y = psi  # sf[j, i]
        sf_y_bak = sf_old[0:-2, 1:-1]  # sf[j-1, i]

        term_y = py_j * (sf_y_fwd - sf_y) - py_jm1 * (sf_y - sf_y_bak)

        c_inner = -inv_dx2 * term_x - inv_dy2 * term_y

        b_int = -w - 0.5 * tau * (c_inner + inv_re * r)

        return b_int.ravel()

    def _construct_matrix(self, px_half: np.ndarray, py_half: np.ndarray):
        geometry: DomainGeometry = self.cfg.geometry
        n_y, n_x = geometry.n_y, geometry.n_x
        dx, dy, tau = self.cfg.scaled_grid_steps
        inv_re = 1.0 / self.cfg.reynolds_number

        inner_n_y, inner_n_x = n_y - 2, n_x - 2
        size = inner_n_x * inner_n_y

        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        tau_half = 0.5 * tau

        rho_inner = self.rho[1:-1, 1:-1]
        rho_term_flat = (tau_half * inv_re * rho_inner).ravel()

        p_e = px_half[1:-1, 1:]
        p_w = px_half[1:-1, :-1]
        p_n = py_half[1:, 1:-1]
        p_s = py_half[:-1, 1:-1]

        a_e = p_e * inv_dx2
        a_w = p_w * inv_dx2
        a_n = p_n * inv_dy2
        a_s = p_s * inv_dy2

        sum_neighbors = a_e + a_w + a_n + a_s

        lam_main = -2.0 * inv_dx2 - 2.0 * inv_dy2
        main_diag = np.full(size, lam_main)

        main_diag -= (tau_half * sum_neighbors).ravel()

        main_diag -= rho_term_flat

        base_side = np.zeros(size - 1)
        base_updown = np.zeros(size - inner_n_x)

        base = 0
        for r in range(inner_n_y):
            if inner_n_x > 1:
                idx = base + np.arange(inner_n_x - 1)
                base_side[idx] = inv_dx2 + tau_half * a_e[r, :-1]
            base += inner_n_x

        base = 0
        for r in range(inner_n_y - 1):
            idx = base + np.arange(inner_n_x)
            base_updown[idx] = inv_dy2 + tau_half * a_n[r, :]
            base += inner_n_x

        if self.vorticity_bc_order == 1:
            diagonals = [main_diag, base_side, base_side, base_updown, base_updown]
        else:  # second order bc
            lower_side = base_side.copy()  # offset -1
            upper_side = base_side.copy()  # offset +1
            lower_ud = base_updown.copy()  # offset -inner_n_x
            upper_ud = base_updown.copy()  # offset +inner_n_x

            delta_diag_x = tau_half * inv_re * (2.0 / (dx**4))
            delta_off_x = tau_half * inv_re * (1.0 / (2.0 * dx**4))

            delta_diag_y = tau_half * inv_re * (2.0 / (dy**4))
            delta_off_y = tau_half * inv_re * (1.0 / (2.0 * dy**4))

            def idx_from_rc(r, c):
                return r * inner_n_x + c

            # Left boundary (inner c == 0): change diag and add coupling to east (c==1)
            if inner_n_x >= 2:
                for r in range(0, inner_n_y):
                    center = idx_from_rc(r, 0)
                    main_diag[center] -= delta_diag_x
                    upper_side[center] += delta_off_x

            # Right boundary (inner c == inner_n_x-1): coupling to west
            if inner_n_x >= 2:
                for r in range(0, inner_n_y):
                    center = idx_from_rc(r, inner_n_x - 1)
                    main_diag[center] -= delta_diag_x
                    # coupling center -> west is stored in lower_side at index (center-1)
                    if center - 1 >= 0:
                        lower_side[center - 1] += delta_off_x

            # Top boundary (inner r == 0): change diag and add coupling to south (r==1)
            if inner_n_y >= 2:
                for c in range(0, inner_n_x):
                    center = idx_from_rc(0, c)
                    main_diag[center] -= delta_diag_y
                    # coupling center -> south is an *upper* updown entry at position `center`
                    upper_ud[center] += delta_off_y

            # Bottom boundary (inner r == inner_n_y-1): coupling to north
            if inner_n_y >= 2:
                for c in range(0, inner_n_x):
                    center = idx_from_rc(inner_n_y - 1, c)
                    main_diag[center] -= delta_diag_y
                    # coupling center -> north is stored in lower_ud at index (center - inner_n_x)
                    pos = center - inner_n_x
                    if pos >= 0:
                        lower_ud[pos] += delta_off_y

            diagonals = [main_diag, lower_side, upper_side, lower_ud, upper_ud]

        offsets = [0, -1, 1, -inner_n_x, inner_n_x]

        m = sparse.diags(diagonals, offsets, shape=(size, size), format="csr")
        return m
