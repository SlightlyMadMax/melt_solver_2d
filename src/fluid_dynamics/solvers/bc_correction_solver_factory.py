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
from src.fluid_dynamics.solvers.stream_function_solvers import *
from src.fluid_dynamics.solvers.vorticity_solvers import *
from src.fluid_dynamics.utils import calculate_vorticity_from_sf
from src.parameters.fluid import FluidParameters


class BCCorrectionNVSolver:
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        sf_bcs: BoundaryConditions,
        sf_max_iters: int = 10000,
        sf_tolerance: float = 1e-6,
        convective_term_form: ConvectiveTermForm = ConvectiveTermForm.DIVERGENT_CENTRAL,
    ):
        self.geometry = geometry
        self.parameters = parameters
        self.convective_operator = VorticityTransportOperator(
            geometry=geometry, form=convective_term_form
        )
        n_y, n_x = geometry.n_y, geometry.n_x

        vorticity_solver_class = VorticitySolverRegistry.get_solver_class(
            solver_name=VorticitySolverName.PEACEMAN_RACHFORD
        )
        stream_function_solver_class = StreamFunctionSolverRegistry.get_solver_class(
            solver_name=StreamFunctionSolverName.CG
        )

        self.vorticity_solver = vorticity_solver_class(
            geometry=geometry,
            parameters=parameters,
            convective_operator=self.convective_operator,
            bc_order=1,
        )
        self.stream_function_solver = stream_function_solver_class(
            geometry=geometry,
            bcs=sf_bcs,
            max_iters=sf_max_iters,
            stopping_criteria=sf_tolerance,
        )

        self._vorticity: NDArray[np.float64] = np.empty((n_y, n_x))
        self._temp_vorticity: NDArray[np.float64] = np.empty((n_y, n_x))
        self._stream_function: NDArray[np.float64] = np.empty((n_y, n_x))
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
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        delta: float,
        time: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        old_vorticity = np.copy(w)

        self._solve_vorticity(
            old_vorticity=old_vorticity,
            stream_function=sf,
            temperature=u,
            delta=delta,
            time=time,
        )
        self._solve_stream_function(
            sf_old=sf,
            vorticity=self._temp_vorticity,
            time=time,
        )

        calculate_vorticity_from_sf(
            sf=self._stream_function,
            result=self._vorticity,
            dy=self.geometry.dy / self.geometry.length_scale,
            dx=self.geometry.dx / self.geometry.length_scale,
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
        self._temp_vorticity[:, :] = self.vorticity_solver.solve(
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
        c_ind = self.vorticity_solver.c_ind
        b = self._construct_rhs_for_cg(
            geometry=self.geometry,
            parameters=self.parameters,
            vorticity=vorticity,
            sf_old=sf_old,
            rho=self.rho,
            c_ind=c_ind,
        )
        A = self._construct_matrix_for_cg(
            geometry=self.geometry,
            parameters=self.parameters,
            rho=self.rho,
            c_ind=c_ind,
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

    @staticmethod
    def _construct_rhs_for_cg(
        geometry: DomainGeometry,
        parameters: FluidParameters,
        vorticity: np.ndarray,
        sf_old: np.ndarray,
        rho: np.ndarray,
        c_ind: np.ndarray,
    ) -> np.ndarray:
        tau = geometry.dt * parameters.v / geometry.length_scale

        psi = sf_old[1:-1, 1:-1]
        w = vorticity[1:-1, 1:-1]
        r = rho[1:-1, 1:-1]
        c = c_ind[1:-1, 1:-1]

        b_int = -w - 0.5 * tau * ((c + r / parameters.reynolds_number) * psi)

        return b_int.ravel()

    @staticmethod
    def _construct_matrix_for_cg(
        geometry: DomainGeometry,
        parameters: FluidParameters,
        rho: np.ndarray,
        c_ind: np.ndarray,
    ):
        n_y, n_x = geometry.n_y, geometry.n_x
        dx = geometry.dx / geometry.length_scale
        dy = geometry.dy / geometry.length_scale
        tau = geometry.dt * parameters.v / geometry.length_scale
        re = parameters.reynolds_number
        dx2, dy2 = dx**2, dy**2

        inner_n_y, inner_n_x = n_y - 2, n_x - 2
        size = inner_n_x * inner_n_y

        c = 0.5 * tau * (c_ind + rho / re)
        c_inner_flat = c[1:-1, 1:-1].ravel()

        main_diag = -2.0 / dx2 - 2.0 / dy2 - c_inner_flat

        side_diag = np.ones(size - 1) / dx2
        side_diag[np.arange(1, size) % inner_n_x == 0] = 0

        up_down_diag = np.ones(size - inner_n_x) / dy2

        diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
        offsets = [0, -1, 1, -inner_n_x, inner_n_x]

        m = sparse.diags(
            diagonals,
            offsets,
            shape=(size, size),
            format="csr",
        )

        return m
