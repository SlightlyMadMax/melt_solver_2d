import numpy as np

from typing import Tuple
from numpy.typing import NDArray
from scipy.sparse import diags


from src.convective_operators import (
    ConvectiveTermForm,
    VorticityTransportOperator,
    EffectiveSFTransportOperator,
)
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.solvers.stream_function_solvers import *
from src.fluid_dynamics.solvers.vorticity_solvers import *
from src.fluid_dynamics.solvers.vorticity_solvers.vab_fully_implicit import (
    VabFullyImplicitScheme,
)
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
        self.convective_operator = VorticityTransportOperator(
            form=convective_term_form, geometry=geometry
        )
        n_y, n_x = geometry.n_y, geometry.n_x

        if bc_order not in (1, 2):
            raise NotImplementedError(
                "Only 1st and 2nd order accuracy BCs are supported for vorticity."
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

        self._vorticity: NDArray[np.float64] = np.empty((n_y, n_x))
        self._stream_function: NDArray[np.float64] = np.empty((n_y, n_x))
        self._iter_stream_function: NDArray[np.float64] = np.empty((n_y, n_x))

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
        sf_tolerance: float = 1e-6,
    ):
        self.geometry = geometry
        self.parameters = parameters
        self.convective_operator = EffectiveSFTransportOperator(geometry=geometry)
        n_y, n_x = geometry.n_y, geometry.n_x

        nonlinearity_predictor_class = VorticitySolverRegistry.get_solver_class(
            solver_name=VorticitySolverName.EXPLICIT
        )
        # vorticity_solver_class = VorticitySolverRegistry.get_solver_class(
        #     solver_name=VorticitySolverName.VABISHCHEVICH
        # )
        stream_function_solver_class = StreamFunctionSolverRegistry.get_solver_class(
            solver_name=StreamFunctionSolverName.CG
        )

        self.nonlinearity_predictor = nonlinearity_predictor_class(
            geometry=geometry,
            parameters=parameters,
            convective_operator=self.convective_operator,
            bc_order=1,
        )
        self.vorticity_solver = VabFullyImplicitScheme(
            geometry=geometry,
            parameters=parameters,
        )
        # self.vorticity_solver = vorticity_solver_class(
        #     geometry=geometry,
        #     parameters=parameters,
        #     convective_operator=self.convective_operator,
        #     bc_order=1,
        # )
        self.stream_function_solver = stream_function_solver_class(
            geometry=geometry,
            bcs=sf_bcs,
            max_iters=sf_max_iters,
            stopping_criteria=sf_tolerance,
        )

        self._vorticity: NDArray[np.float64] = np.empty((n_y, n_x))
        self._temp_vorticity: NDArray[np.float64] = np.empty((n_y, n_x))
        self._stream_function: NDArray[np.float64] = np.empty((n_y, n_x))
        self._conv_x: NDArray[np.float64] = np.empty((n_y, n_x, 3))
        self._conv_y: NDArray[np.float64] = np.empty((n_y, n_x, 3))

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        old_vorticity = np.copy(w)
        old_sf = np.copy(sf)

        self._predict_vorticity(
            old_vorticity=old_vorticity,
            stream_function=sf,
            temperature=u,
            time=time,
        )
        self._solve_vorticity(
            old_vorticity=old_vorticity,
            conv_vorticity=self._temp_vorticity,
            stream_function=sf,
            temperature=u,
            time=time,
        )
        self._solve_stream_function(
            sf_old=sf,
            vorticity=self._temp_vorticity,
            conv_vorticity=old_vorticity,
            time=time,
        )
        self._update_vorticity(
            sf_new=self._stream_function,
            sf_old=old_sf,
            temp_vorticity=self._temp_vorticity,
        )
        return self._stream_function, self._vorticity

    def _predict_vorticity(
        self,
        old_vorticity: np.ndarray,
        stream_function: np.ndarray,
        temperature: np.ndarray,
        time: float,
    ) -> None:
        self._temp_vorticity[:, :] = self.nonlinearity_predictor.solve(
            w=old_vorticity,
            sf=stream_function,
            u=temperature,
            time=time,
        )

    def _solve_vorticity(
        self,
        old_vorticity: np.ndarray,
        conv_vorticity: np.ndarray,
        stream_function: np.ndarray,
        temperature: np.ndarray,
        time: float,
    ) -> None:
        self._temp_vorticity[:, :] = self.vorticity_solver.solve(
            w=old_vorticity,
            conv_w=conv_vorticity,
            sf=stream_function,
            u=temperature,
            time=time,
        )

    def _solve_stream_function(
        self,
        sf_old: np.ndarray,
        vorticity: np.ndarray,
        conv_vorticity: np.ndarray,
        time: float,
    ) -> None:
        c_ind = self.vorticity_solver.c_ind
        rho = self.vorticity_solver.rho
        self.convective_operator(
            w=conv_vorticity, conv_x=self._conv_x, conv_y=self._conv_y
        )
        b = construct_rhs_for_cg(
            geometry=self.geometry,
            parameters=self.parameters,
            vorticity=vorticity,
            sf_old=sf_old,
            rho=rho,
            c_ind=c_ind,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
        )
        A = construct_matrix_for_cg(
            geometry=self.geometry,
            parameters=self.parameters,
            rho=rho,
            c_ind=c_ind,
            conv_x=self._conv_x,
            conv_y=self._conv_y,
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

    def _update_vorticity(
        self,
        sf_new,
        sf_old,
        temp_vorticity,
    ):
        c_ind = self.vorticity_solver.c_ind
        rho = self.vorticity_solver.rho
        conv_x_arr = self._conv_x
        conv_y_arr = self._conv_y
        omega = self._vorticity
        tau = self.geometry.dt * self.parameters.v / self.geometry.length_scale
        inv_reynolds = 1.0 / self.parameters.reynolds_number
        inv_dx2 = 1.0 / self.geometry.dx**2
        inv_dy2 = 1.0 / self.geometry.dy**2
        dsf = sf_new - sf_old

        # Define slices for interior points (1:-1, 1:-1)
        interior = (slice(1, -1), slice(1, -1))

        dsf_right = dsf[interior[0], 2:]  # dsf[j, i+1]
        dsf_left = dsf[interior[0], :-2]  # dsf[j, i-1]
        dsf_up = dsf[2:, interior[1]]  # dsf[j+1, i]
        dsf_down = dsf[:-2, interior[1]]  # dsf[j-1, i]

        conv_x_0 = conv_x_arr[interior[0], interior[1], 0]
        conv_x_2 = conv_x_arr[interior[0], interior[1], 2]
        conv_y_0 = conv_y_arr[interior[0], interior[1], 0]
        conv_y_2 = conv_y_arr[interior[0], interior[1], 2]

        term_x0 = conv_x_0 * dsf_right
        term_x2 = conv_x_2 * dsf_left
        term_y0 = conv_y_0 * dsf_up
        term_y2 = conv_y_2 * dsf_down
        Vphi = term_x0 + term_x2 + term_y0 + term_y2

        # Compute combined Linv and Cphi terms
        linv_cphi = (inv_reynolds * rho[interior] + c_ind[interior]) * dsf[interior]

        # Update vorticity for interior points
        omega[interior] = temp_vorticity[interior] + tau * (Vphi + linv_cphi)

        # Apply boundary conditions
        omega[-1, :] = -2.0 * sf_new[-2, :] * inv_dy2  # Top boundary
        omega[:, -1] = -2.0 * sf_new[:, -2] * inv_dx2  # Right boundary
        omega[0, :] = -2.0 * sf_new[1, :] * inv_dy2  # Bottom boundary
        omega[:, 0] = -2.0 * sf_new[:, 1] * inv_dx2  # Left boundary


def construct_rhs_for_cg(
    geometry: DomainGeometry,
    parameters: FluidParameters,
    vorticity: np.ndarray,
    sf_old: np.ndarray,
    rho: np.ndarray,
    c_ind: np.ndarray,
    conv_x: np.ndarray,
    conv_y: np.ndarray,
) -> np.ndarray:
    tau = geometry.dt * parameters.v / geometry.length_scale

    psi = sf_old[1:-1, 1:-1]
    w = vorticity[1:-1, 1:-1]
    r = rho[1:-1, 1:-1]
    c = c_ind[1:-1, 1:-1]

    conv = (
        conv_x[1:-1, 1:-1, 0] * sf_old[1:-1, 2:]
        + conv_x[1:-1, 1:-1, 2] * sf_old[1:-1, :-2]
        + conv_y[1:-1, 1:-1, 0] * sf_old[2:, 1:-1]
        + conv_y[1:-1, 1:-1, 2] * sf_old[:-2, 1:-1]
    )

    b_int = -w - tau * ((c + r / parameters.reynolds_number) * psi + conv)

    return b_int.ravel()


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
    size = inner_n_x * inner_n_y

    tau = geometry.dt * parameters.v / geometry.length_scale
    c = tau * (c_ind + rho / parameters.reynolds_number)
    c_inner_flat = c[1:-1, 1:-1].flatten()

    # Main diagonal: Δψ - c ψ
    main_diag = -2.0 / dx2 - 2.0 / dy2 - c_inner_flat

    # Off-diagonals for interior stencil + convection
    x_off_right = (1.0 / dx2 - tau * conv_x[1:-1, 1:-1, 0]).flatten()
    x_off_left = (1.0 / dx2 - tau * conv_x[1:-1, 1:-1, 2]).flatten()
    y_off_up = (1.0 / dy2 - tau * conv_y[1:-1, 1:-1, 0]).flatten()
    y_off_down = (1.0 / dy2 - tau * conv_y[1:-1, 1:-1, 2]).flatten()

    for row in range(1, inner_n_y):
        x_off_right[row * inner_n_x - 1] = 0.0
        x_off_left[row * inner_n_x] = 0.0

    diagonals = [
        main_diag,
        x_off_left,
        x_off_right,
        y_off_down,
        y_off_up,
    ]
    offsets = [0, -1, 1, -inner_n_x, inner_n_x]

    return diags(diagonals, offsets, shape=(size, size), format="csr")
