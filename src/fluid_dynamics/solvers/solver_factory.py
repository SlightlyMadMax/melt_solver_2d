import numpy as np

from typing import Tuple
from numpy.typing import NDArray
from scipy.sparse import diags


from src.convective_operators import (
    ConvectiveTermForm,
    ConvectiveVorticityTransportOperator,
    EffectiveSFTransportOperator,
)
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.solvers.stream_function_solvers import *
from src.fluid_dynamics.solvers.vorticity_solvers import *
from src.fluid_dynamics.utils import calculate_vorticity_from_sf
from src.parameters.fluid import FluidParameters


class IterativeNavierStokesSolver:
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        sf_bcs: BoundaryConditions,
        vorticity_solver_name: VorticitySolverName = VorticitySolverName.PEACEMAN_RACHFORD,
        stream_function_solver_name: StreamFunctionSolverName = StreamFunctionSolverName.SOR,
        convective_term_form: ConvectiveTermForm = ConvectiveTermForm.UPWIND,
        sf_max_iters: int = 1000,
        sf_stopping_criteria: float = 1e-6,
        max_iters: int = 5,
        tolerance: float = 1e-6,
        urf: float = 0.5,
        bc_order: int = 2,
    ):
        self.geometry = geometry
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.urf = urf

        if bc_order not in (1, 2):
            raise NotImplementedError(
                "Only 1st and 2nd order accuracy BCs are supported for vorticity."
            )

        self.convective_operator = ConvectiveVorticityTransportOperator(
            form=convective_term_form, geometry=geometry
        )

        vorticity_solver_class = VorticitySolverRegistry.get_solver_class(
            solver_name=vorticity_solver_name
        )
        stream_function_solver_class = StreamFunctionSolverRegistry.get_solver_class(
            solver_name=stream_function_solver_name
        )

        self.vorticity_solver = vorticity_solver_class(
            geometry=geometry,
            parameters=parameters,
            convective_operator=self.convective_operator,
            bc_order=bc_order,
            incorporated_bc=False,
        )
        self.stream_function_solver = stream_function_solver_class(
            geometry=geometry,
            bcs=sf_bcs,
            max_iters=sf_max_iters,
            stopping_criteria=sf_stopping_criteria,
        )

        self._vorticity: NDArray[np.float64] = np.empty((geometry.n_y, geometry.n_x))
        self._stream_function: NDArray[np.float64] = np.empty(
            (geometry.n_y, geometry.n_x)
        )
        self._iter_stream_function: NDArray[np.float64] = np.empty(
            (geometry.n_y, geometry.n_x)
        )

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        urf = self.urf
        last_diff = np.inf
        self._stream_function[:, :] = sf
        self._iter_stream_function[:, :] = sf

        for iteration in range(self.max_iters):
            self._solve_vorticity(
                old_vorticity=w,
                stream_function=self._iter_stream_function,
                temperature=u,
                time=time,
            )
            self._solve_stream_function(
                initial_guess=self._stream_function,
                vorticity=self._vorticity,
                time=time,
            )

            # Check for convergence
            norm_diff = np.linalg.norm(
                self._stream_function - self._iter_stream_function, ord=2
            )
            if norm_diff < self.tolerance:
                break

            # Adaptive under-relaxation parameter
            if norm_diff > last_diff:
                urf = max(urf * 0.5, 1e-4)
            last_diff = norm_diff

            # Under-relaxation
            self._iter_stream_function[:, :] = (
                urf * self._stream_function + (1 - urf) * self._iter_stream_function
            )

        return self._stream_function, self._vorticity

    def _solve_vorticity(
        self,
        old_vorticity: np.ndarray,
        stream_function: np.ndarray,
        temperature: np.ndarray,
        time: float,
    ) -> None:
        self._vorticity[:, :] = self.vorticity_solver.solve(
            w=old_vorticity,
            sf=stream_function,
            u=temperature,
            time=time,
        )

    def _solve_stream_function(
        self,
        initial_guess: np.ndarray,
        vorticity: np.ndarray,
        time: float,
    ) -> None:
        self._stream_function[:, :] = self.stream_function_solver.solve(
            initial_guess=initial_guess, rhs=-vorticity, time=time
        )


class NonIterativeNavierStokersSolver:
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        sf_bcs: BoundaryConditions,
        sf_max_iters: int = 10000,
        sf_stopping_criteria: float = 1e-6,
    ):
        self.geometry = geometry
        self.parameters = parameters

        self.convective_operator = EffectiveSFTransportOperator(geometry=geometry)

        vorticity_solver_class = VorticitySolverRegistry.get_solver_class(
            solver_name=VorticitySolverName.VABISHCHEVICH
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
            stopping_criteria=sf_stopping_criteria,
        )

        self._vorticity: NDArray[np.float64] = np.empty((geometry.n_y, geometry.n_x))
        self._temp_vorticity: NDArray[np.float64] = np.empty(
            (geometry.n_y, geometry.n_x)
        )
        self._stream_function: NDArray[np.float64] = np.empty(
            (geometry.n_y, geometry.n_x)
        )

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._temp_vorticity = self._solve_vorticity(
            old_vorticity=w,
            conv_vorticity=w,
            stream_function=sf,
            temperature=u,
            time=time,
        )
        # print(np.max(self._temp_vorticity))
        # print(np.min(self._temp_vorticity))
        self._stream_function = self._solve_stream_function(
            sf_nm1=sf,
            vorticity=self._temp_vorticity,
            conv_vorticity=w,
            time=time,
        )
        calculate_vorticity_from_sf(
            sf=self._stream_function,
            result=self._vorticity,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
        )
        # print(np.max(self._stream_function))
        # print(np.min(self._stream_function))
        # print(np.max(self._vorticity))
        # print(np.min(self._vorticity))
        # print()
        return self._stream_function, self._vorticity

    def _solve_vorticity(
        self,
        old_vorticity: np.ndarray,
        conv_vorticity: np.ndarray,
        stream_function: np.ndarray,
        temperature: np.ndarray,
        time: float,
    ) -> np.ndarray:
        return self.vorticity_solver.solve(
            w=old_vorticity,
            conv_w=conv_vorticity,
            sf=stream_function,
            u=temperature,
            time=time,
        )

    def _solve_stream_function(
        self,
        sf_nm1: np.ndarray,
        vorticity: np.ndarray,
        conv_vorticity: np.ndarray,
        time: float,
    ) -> np.ndarray:
        conv_x, conv_y = self.convective_operator(w=conv_vorticity)
        b = construct_rhs_for_cg(
            geometry=self.geometry,
            parameters=self.parameters,
            vorticity=vorticity,
            sf_nm1=sf_nm1,
            rho=self.vorticity_solver.rho,
            c_ind=self.vorticity_solver.c_ind,
            conv_x=conv_x,
            conv_y=conv_y,
        )
        A = construct_matrix_for_cg(
            geometry=self.geometry,
            parameters=self.parameters,
            rho=self.vorticity_solver.rho,
            c_ind=self.vorticity_solver.c_ind,
            conv_x=conv_x,
            conv_y=conv_y,
        )
        return self.stream_function_solver.solve(
            initial_guess=sf_nm1,
            A=-A,
            b=-b,
            time=time,
        )


def construct_rhs_for_cg(
    geometry: DomainGeometry,
    parameters: FluidParameters,
    vorticity: np.ndarray,
    sf_nm1: np.ndarray,
    rho: np.ndarray,
    c_ind: np.ndarray,
    conv_x: np.ndarray,
    conv_y: np.ndarray,
) -> np.ndarray:
    b = np.zeros_like(sf_nm1)
    tau = geometry.dt * parameters.v / geometry.length_scale
    for i in range(1, geometry.n_x - 1):
        for j in range(1, geometry.n_y - 1):
            b[j, i] = -vorticity[j, i] - 0.5 * tau * (
                (c_ind[j, i] + rho[j, i] / parameters.reynolds_number) * sf_nm1[j, i]
                - (
                    conv_x[j, i, 0] * sf_nm1[j, i + 1]
                    + conv_x[j, i, 2] * sf_nm1[j, i - 1]
                )
                - (
                    conv_y[j, i, 0] * sf_nm1[j + 1, i]
                    + conv_y[j, i, 2] * sf_nm1[j - 1, i]
                )
            )
    return b


def construct_matrix_for_cg(
    geometry: DomainGeometry,
    parameters: FluidParameters,
    rho: np.ndarray,
    c_ind: np.ndarray,
    conv_x: np.ndarray,
    conv_y: np.ndarray,
):
    n_y, n_x = geometry.n_y, geometry.n_x
    dx2 = geometry.dx**2
    dy2 = geometry.dy**2

    inner_n_y, inner_n_x = n_y - 2, n_x - 2

    tau = geometry.dt * parameters.v / geometry.length_scale
    c = 0.5 * tau * (c_ind + rho / parameters.reynolds_number)

    c_inner = c[1:-1, 1:-1]
    c_inner_flat = c_inner.flatten()

    diagonal = -2 / dx2 - 2 / dy2 - c_inner_flat

    size = inner_n_x * inner_n_y
    main_diag = np.full(size, diagonal)

    # Off-diagonal elements must be adjusted for staggered indexing
    x_off_diag_array = (1 / dx2) + conv_x[1:-1, 1:-1, 0].flatten()  # u_{i+1, j}
    x_off_diag_left_array = (1 / dx2) + conv_x[1:-1, 1:-1, 2].flatten()  # u_{i-1, j}
    y_off_diag_array = (1 / dy2) + conv_y[1:-1, 1:-1, 0].flatten()  # u_{i, j+1}
    y_off_diag_bottom_array = (1 / dy2) + conv_y[1:-1, 1:-1, 2].flatten()  # u_{i, j-1}

    for i in range(1, inner_n_y):
        x_off_diag_array[i * inner_n_x - 1] = 0
        x_off_diag_left_array[i * inner_n_x - 1] = 0

    diagonals = [
        main_diag,
        x_off_diag_left_array,
        x_off_diag_array,
        y_off_diag_bottom_array,
        y_off_diag_array,
    ]
    offsets = [0, -1, 1, -inner_n_x, inner_n_x]

    m = diags(diagonals, offsets, shape=(size, size), format="csr")

    return m
