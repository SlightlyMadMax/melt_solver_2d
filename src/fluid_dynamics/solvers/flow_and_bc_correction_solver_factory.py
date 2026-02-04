import numpy as np

from typing import Tuple
from numpy.typing import NDArray
from scipy import sparse


from src.convective_operators import VorticityBasedConvectiveOperator
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.solvers.stream_function_solvers import *
from src.fluid_dynamics.solvers.vorticity_solvers import *
from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import PenaltyTermForm
from src.fluid_dynamics.utils import calculate_vorticity_from_sf
from src.parameters.config import ExperimentConfig


class FlowCorrectionNVSolver:
    def __init__(
        self,
        cfg: ExperimentConfig,
        sf_bcs: BoundaryConditions,
        sf_max_iters: int = 10000,
        sf_tolerance: float = 1e-6,
        penalty_term_form: PenaltyTermForm = PenaltyTermForm.LINEAR,
    ):
        self.cfg = cfg
        self.convective_operator = VorticityBasedConvectiveOperator(cfg=self.cfg)
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x

        nonlinearity_predictor_class = VorticitySolverRegistry.get_solver_class(
            solver_name=VorticitySolverName.EXPLICIT
        )
        vorticity_solver_class = VorticitySolverRegistry.get_solver_class(
            solver_name=VorticitySolverName.VABISHCHEVICH
        )
        stream_function_solver_class = StreamFunctionSolverRegistry.get_solver_class(
            solver_name=StreamFunctionSolverName.AMG
        )

        self.nonlinearity_predictor = nonlinearity_predictor_class(
            cfg=cfg,
            convective_operator=self.convective_operator,
            bc_order=1,
            penalty_term_form=penalty_term_form,
        )
        self.vorticity_solver = vorticity_solver_class(
            cfg=cfg,
            convective_operator=self.convective_operator,
            bc_order=1,
            penalty_term_form=penalty_term_form,
        )
        self.stream_function_solver = stream_function_solver_class(
            cfg=cfg,
            bcs=sf_bcs,
            max_iters=sf_max_iters,
            stopping_criteria=sf_tolerance,
        )

        self._vorticity: NDArray[np.float64] = np.empty((n_y, n_x))
        self._temp_vorticity: NDArray[np.float64] = np.empty((n_y, n_x))
        self._stream_function: NDArray[np.float64] = np.empty((n_y, n_x))
        self._conv_x: NDArray[np.float64] = np.empty((n_y, n_x, 3))
        self._conv_y: NDArray[np.float64] = np.empty((n_y, n_x, 3))
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

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        delta: float,
        time: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        old_vorticity = np.copy(w)

        self._predict_vorticity(
            old_vorticity=old_vorticity,
            stream_function=sf,
            temperature=u,
            delta=delta,
            time=time,
        )
        self._solve_vorticity(
            old_vorticity=old_vorticity,
            conv_vorticity=self._temp_vorticity,
            stream_function=sf,
            temperature=u,
            delta=delta,
            time=time,
        )
        self._solve_stream_function(
            sf_old=sf,
            vorticity=self._temp_vorticity,
            conv_vorticity=old_vorticity,
            time=time,
        )
        calculate_vorticity_from_sf(
            sf=self._stream_function, result=self._vorticity, cfg=self.cfg
        )
        return self._stream_function, self._vorticity

    def _predict_vorticity(
        self,
        old_vorticity: np.ndarray,
        stream_function: np.ndarray,
        temperature: np.ndarray,
        delta: float,
        time: float,
    ) -> None:
        self._temp_vorticity[:, :] = self.nonlinearity_predictor.solve(
            w=old_vorticity,
            sf=stream_function,
            u=temperature,
            delta=delta,
            time=time,
        )

    def _solve_vorticity(
        self,
        old_vorticity: np.ndarray,
        conv_vorticity: np.ndarray,
        stream_function: np.ndarray,
        temperature: np.ndarray,
        delta: float,
        time: float,
    ) -> None:
        self._temp_vorticity[:, :] = self.vorticity_solver.solve(
            w=old_vorticity,
            conv_w=conv_vorticity,
            sf=stream_function,
            u=temperature,
            delta=delta,
            time=time,
        )

    def _solve_stream_function(
        self,
        sf_old: np.ndarray,
        vorticity: np.ndarray,
        conv_vorticity: np.ndarray,
        time: float,
    ) -> None:
        self.convective_operator(
            w=conv_vorticity, conv_x=self._conv_x, conv_y=self._conv_y
        )
        b = self._construct_rhs(
            vorticity=vorticity,
            sf_old=sf_old,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            px_half=self.vorticity_solver.px_half,
            py_half=self.vorticity_solver.py_half,
        )
        A = self._construct_matrix(
            conv_x=self._conv_x,
            conv_y=self._conv_y,
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
        conv_x: np.ndarray,
        conv_y: np.ndarray,
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
        r = self.rho[1:-1, 1:-1]

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

        conv = (
            conv_x[1:-1, 1:-1, 0] * sf_old[1:-1, 2:]
            + conv_x[1:-1, 1:-1, 2] * sf_old[1:-1, :-2]
            + conv_y[1:-1, 1:-1, 0] * sf_old[2:, 1:-1]
            + conv_y[1:-1, 1:-1, 2] * sf_old[:-2, 1:-1]
        )

        b_int = -w - 0.5 * tau * (c_inner + inv_re * r * psi + conv)

        return b_int.ravel()

    def _construct_matrix(
        self,
        conv_x: np.ndarray,
        conv_y: np.ndarray,
        px_half: np.ndarray,
        py_half: np.ndarray,
    ):
        geometry: DomainGeometry = self.cfg.geometry
        n_y, n_x = geometry.n_y, geometry.n_x
        dx, dy, tau = self.cfg.scaled_grid_steps
        inv_re = 1.0 / self.cfg.reynolds_number

        inner_n_y, inner_n_x = n_y - 2, n_x - 2
        size = inner_n_x * inner_n_y

        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        tau_half = 0.5 * tau

        p_e = px_half[1:-1, 1:]  # east half-link for inner points
        p_w = px_half[1:-1, :-1]  # west half-link
        p_n = py_half[1:, 1:-1]  # north half-link
        p_s = py_half[:-1, 1:-1]  # south half-link

        a_e = p_e * inv_dx2
        a_w = p_w * inv_dx2
        a_n = p_n * inv_dy2
        a_s = p_s * inv_dy2

        sum_neighbors = a_e + a_w + a_n + a_s  # shape (inner_n_y, inner_n_x)

        rho_inner = self.rho[1:-1, 1:-1]
        rho_term_flat = (tau_half * inv_re * rho_inner).ravel()

        conv_x_inner = conv_x[1:-1, 1:-1, :]
        conv_y_inner = conv_y[1:-1, 1:-1, :]

        conv_x_east_flat = conv_x_inner[:, :, 0].flatten()
        conv_x_west_flat = conv_x_inner[:, :, 2].flatten()  # conv at west face
        conv_y_north_flat = conv_y_inner[:, :, 0].flatten()  # conv at north face
        conv_y_south_flat = conv_y_inner[:, :, 2].flatten()  # conv at south face

        lam_main = -2.0 * inv_dx2 - 2.0 * inv_dy2
        main_diag = np.full(size, lam_main)

        main_diag -= (tau_half * sum_neighbors).ravel()

        main_diag -= rho_term_flat

        side_diag_east = np.zeros(size - 1)  # +1 offset
        side_diag_west = np.zeros(size - 1)  # -1 offset

        east_mask = (np.arange(size) % inner_n_x) < (inner_n_x - 1)
        east_idx = np.where(east_mask)[0]
        a_e_flat = a_e.flatten()
        a_w_flat = a_w.flatten()

        side_diag_east[east_idx] = (inv_dx2 + tau_half * a_e_flat[east_idx]) - (
            tau_half * conv_x_east_flat[east_idx]
        )

        west_valid_in_diag = (np.arange(1, size) % inner_n_x) != 0
        west_valid_indices = np.where(west_valid_in_diag)[0]

        side_diag_west[west_valid_indices] = (
            inv_dx2 + tau_half * a_w_flat[np.arange(1, size)[west_valid_in_diag]]
        ) - (tau_half * conv_x_west_flat[np.arange(1, size)[west_valid_in_diag]])

        up_down_len = size - inner_n_x
        up_down_diag_north = np.zeros(up_down_len)  # +inner_n_x offset (north)
        up_down_diag_south = np.zeros(up_down_len)  # -inner_n_x offset (south)

        if inner_n_y > 1:
            base = 0
            pos = 0
            for r in range(inner_n_y - 1):
                idx = base + np.arange(inner_n_x)
                up_down_diag_north[pos : pos + inner_n_x] = (
                    inv_dy2 + tau_half * a_n[r, :]
                ) - (tau_half * conv_y_north_flat[idx])
                up_down_diag_south[pos : pos + inner_n_x] = (
                    inv_dy2 + tau_half * a_s[r, :]
                ) - (tau_half * conv_y_south_flat[idx + inner_n_x])
                base += inner_n_x
                pos += inner_n_x

        diagonals = [
            main_diag,
            side_diag_west,
            side_diag_east,
            up_down_diag_south,
            up_down_diag_north,
        ]
        offsets = [0, -1, 1, -inner_n_x, inner_n_x]

        m = sparse.diags(diagonals, offsets, shape=(size, size), format="csr")
        return m
