import numpy as np

from typing import Tuple
from numpy.typing import NDArray
from scipy import sparse


from src.convective_operators import (
    ConvectiveTermForm,
    VorticityTransportOperator,
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
        vorticity_bc_order: int = 1,
    ):
        self.cfg = cfg
        self.convective_operator = VorticityTransportOperator(
            cfg=cfg, form=convective_term_form
        )
        n_y, n_x = cfg.geometry.n_y, cfg.geometry.n_x

        vorticity_solver_class = VorticitySolverRegistry.get_solver_class(
            solver_name=VorticitySolverName.PEACEMAN_RACHFORD
        )
        stream_function_solver_class = StreamFunctionSolverRegistry.get_solver_class(
            solver_name=StreamFunctionSolverName.CG
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
        self.rho = self.calculate_rho()

    def calculate_rho(self):
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
            sf=self._stream_function,
            result=self._vorticity,
            dy=self.cfg.geometry.dy / self.cfg.l,
            dx=self.cfg.geometry.dx / self.cfg.l,
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
        b = self._construct_rhs_for_cg(
            vorticity=vorticity,
            sf_old=sf_old,
            Sx_half=self.vorticity_solver.px_half,
            Sy_half=self.vorticity_solver.py_half,
        )
        A = self._construct_matrix_for_cg(
            Sx_half=self.vorticity_solver.px_half,
            Sy_half=self.vorticity_solver.py_half,
        )
        self._stream_function[:, :] = self.stream_function_solver.solve(
            initial_guess=sf_old,
            A=-A,
            b_flat=-b,
            time=time,
        )
        # psi_vec = self._stream_function[1:-1, 1:-1].ravel()
        # residual = (-A).dot(psi_vec) - (-b).ravel()
        # print("‖residual‖₂:", np.linalg.norm(residual, 2))

    def _construct_rhs_for_cg(
        self,
        vorticity: np.ndarray,
        sf_old: np.ndarray,
        Sx_half: np.ndarray,
        Sy_half: np.ndarray,
    ) -> np.ndarray:
        n_y, n_x = sf_old.shape
        dx, dy, tau = self.cfg.scaled_grid_steps
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)

        psi = sf_old[1:-1, 1:-1]
        w = vorticity[1:-1, 1:-1]
        r = self.rho[1:-1, 1:-1]

        c = np.empty((n_y, n_x))

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                p_ip1j = Sx_half[j, i]
                p_im1j = Sx_half[j, i - 1]
                p_ijp1 = Sy_half[j, i]
                p_ijm1 = Sy_half[j - 1, i]
                c[j, i] = -inv_dx2 * (
                    p_ip1j * (sf_old[j, i + 1] - sf_old[j, i])
                    - p_im1j * (sf_old[j, i] - sf_old[j, i - 1])
                ) - inv_dy2 * (
                    p_ijp1 * (sf_old[j + 1, i] - sf_old[j, i])
                    - p_ijm1 * (sf_old[j, i] - sf_old[j - 1, i])
                )

        b_int = -w - 0.5 * tau * (c[1:-1, 1:-1] + r * psi / self.cfg.reynolds_number)

        return b_int.ravel()

    def _construct_matrix_for_cg(
        self,
        Sx_half: np.ndarray,
        Sy_half: np.ndarray,
    ):
        geometry: DomainGeometry = self.cfg.geometry
        n_y, n_x = geometry.n_y, geometry.n_x
        dx, dy, tau = self.cfg.scaled_grid_steps
        re = self.cfg.reynolds_number
        dx2, dy2 = dx**2, dy**2

        inner_n_y, inner_n_x = n_y - 2, n_x - 2
        size = inner_n_x * inner_n_y

        rho_inner = self.rho[1:-1, 1:-1]
        rho_term_flat = (0.5 * tau * (rho_inner / re)).ravel()

        S_e = Sx_half[1:-1, 1:]
        S_w = Sx_half[1:-1, :-1]

        S_n = Sy_half[1:, 1:-1]
        S_s = Sy_half[:-1, 1:-1]

        aE = S_e / dx2
        aW = S_w / dx2
        aN = S_n / dy2
        aS = S_s / dy2

        sum_neighbors = aE + aW + aN + aS

        lam_main = -2.0 / dx2 - 2.0 / dy2
        main_diag = np.full(size, lam_main, dtype=float)

        main_diag -= (0.5 * tau * sum_neighbors).ravel()

        main_diag -= rho_term_flat

        side_diag = np.zeros(size - 1, dtype=float)
        up_down_diag = np.zeros(size - inner_n_x, dtype=float)

        base = 0
        for r in range(inner_n_y):
            if inner_n_x > 1:
                idx = base + np.arange(inner_n_x - 1)
                side_diag[idx] = 1.0 / dx2 + 0.5 * tau * aE[r, :-1]
            base += inner_n_x

        base = 0
        for r in range(inner_n_y - 1):
            idx = base + np.arange(inner_n_x)
            up_down_diag[idx] = 1.0 / dy2 + 0.5 * tau * aN[r, :]
            base += inner_n_x

        diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
        offsets = [0, -1, 1, -inner_n_x, inner_n_x]

        m = sparse.diags(diagonals, offsets, shape=(size, size), format="csr")
        return m
