import numpy as np

from typing import Tuple
from numpy.typing import NDArray
from scipy import sparse


from src.convective_operators import VorticityBasedConvectiveOperator
from src.core.boundary_conditions import BoundaryConditions
from src.fluid_dynamics.solvers.stream_function_solvers import *
from src.fluid_dynamics.solvers.vorticity_solvers import *
from src.fluid_dynamics.utils import calculate_vorticity_from_sf
from src.parameters.config import ExperimentConfig


class FlowCorrectionNVSolver:
    def __init__(
        self,
        cfg: ExperimentConfig,
        sf_bcs: BoundaryConditions,
        sf_max_iters: int = 10000,
        sf_tolerance: float = 1e-6,
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
        )
        self.vorticity_solver = vorticity_solver_class(
            cfg=cfg,
            convective_operator=self.convective_operator,
            bc_order=1,
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
        dy_scaled, dx_scaled = (
            self.cfg.geometry.dy / self.cfg.l,
            self.cfg.geometry.dx / self.cfg.l,
        )

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
        penalty_term = self.vorticity_solver.penalty_term
        self.convective_operator(
            w=conv_vorticity, conv_x=self._conv_x, conv_y=self._conv_y
        )
        b = self._construct_rhs(
            vorticity=vorticity,
            sf_old=sf_old,
            penalty_term=penalty_term,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
        )
        A = self._construct_matrix(
            penalty_term=penalty_term,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
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
        penalty_term: np.ndarray,
        conv_x: np.ndarray,
        conv_y: np.ndarray,
    ) -> np.ndarray:
        _, _, tau = self.cfg.scaled_grid_steps
        inv_re = 1.0 / self.cfg.reynolds_number

        psi = sf_old[1:-1, 1:-1]
        w = vorticity[1:-1, 1:-1]
        r = self.rho[1:-1, 1:-1]
        c = penalty_term[1:-1, 1:-1]

        conv = (
            conv_x[1:-1, 1:-1, 0] * sf_old[1:-1, 2:]
            + conv_x[1:-1, 1:-1, 2] * sf_old[1:-1, :-2]
            + conv_y[1:-1, 1:-1, 0] * sf_old[2:, 1:-1]
            + conv_y[1:-1, 1:-1, 2] * sf_old[:-2, 1:-1]
        )

        b_int = -w - 0.5 * tau * ((c + inv_re * r) * psi + conv)

        return b_int.ravel()

    def _construct_matrix(
        self,
        penalty_term: np.ndarray,
        conv_x: np.ndarray,
        conv_y: np.ndarray,
    ):
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        dx, dy, tau = self.cfg.scaled_grid_steps
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        tau_half = 0.5 * tau
        inv_re = 1.0 / self.cfg.reynolds_number

        inner_n_y, inner_n_x = n_y - 2, n_x - 2
        size = inner_n_x * inner_n_y

        c = tau_half * (penalty_term + inv_re * self.rho)
        c_inner_flat = c[1:-1, 1:-1].flatten()

        # Extract convective coefficients for inner grid
        conv_x_inner = conv_x[1:-1, 1:-1, :]
        conv_y_inner = conv_y[1:-1, 1:-1, :]

        conv_x_east_flat = conv_x_inner[:, :, 0].flatten()
        conv_x_west_flat = conv_x_inner[:, :, 2].flatten()
        conv_y_north_flat = conv_y_inner[:, :, 0].flatten()
        conv_y_south_flat = conv_y_inner[:, :, 2].flatten()

        # Main diagonal: Δψ - c ψ
        main_diag = -2.0 * inv_dx2 - 2.0 * inv_dy2 - c_inner_flat

        # East diagonal (offset +1)
        east_mask = (np.arange(size) % inner_n_x) < (inner_n_x - 1)
        east_valid_indices = np.where(east_mask)[0]
        east_diag = np.zeros(size - 1)
        east_diag[east_valid_indices] = (
            inv_dx2 - tau_half * conv_x_east_flat[east_valid_indices]
        )

        # West diagonal (offset -1)
        west_valid_in_diag = (np.arange(1, size) % inner_n_x) != 0
        west_valid_indices = np.where(west_valid_in_diag)[0]
        west_diag = np.zeros(size - 1)
        west_diag[west_valid_indices] = (
            inv_dx2
            - tau_half * conv_x_west_flat[np.arange(1, size)[west_valid_in_diag]]
        )

        # North diagonal (offset +inner_n_x)
        north_diag = inv_dy2 - tau_half * conv_y_north_flat[: (size - inner_n_x)]

        # South diagonal (offset -inner_n_x)
        south_diag = inv_dy2 - tau_half * conv_y_south_flat[inner_n_x:]

        diagonals = [main_diag, west_diag, east_diag, north_diag, south_diag]
        offsets = [0, -1, 1, inner_n_x, -inner_n_x]

        m = sparse.diags(diagonals, offsets, shape=(size, size), format="csr")

        return m
