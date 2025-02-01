from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryConditions
from src.convective_operator import ConvectiveTermForm, ConvectionOperator
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.solvers.stream_function_solvers import *
from src.fluid_dynamics.solvers.vorticity_solvers import *
from src.fluid_dynamics.utils import calculate_vorticity_from_sf
from src.geometry import DomainGeometry


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
        implicit_lin_max_iters: int = 5,
        implicit_lin_stopping_criteria: float = 1e-6,
        implicit_lin_urf: float = 0.5,
        bc_order: int = 2,
    ):
        self.geometry = geometry
        self.implicit_lin_max_iters = implicit_lin_max_iters
        self.implicit_lin_stopping_criteria = implicit_lin_stopping_criteria
        self.implicit_lin_urf = implicit_lin_urf

        if bc_order not in (1, 2):
            raise NotImplementedError(
                "Only 1st and 2nd order accuracy BCs are supported for vorticity."
            )

        self.convective_operator = ConvectionOperator(
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
        self._temp_stream_function: NDArray[np.float64] = np.empty(
            (geometry.n_y, geometry.n_x)
        )

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha = self.implicit_lin_urf
        self._stream_function = np.copy(sf)

        for iteration in range(self.implicit_lin_max_iters):
            self._vorticity = self._solve_vorticity(
                old_vorticity=w,
                stream_function=self._stream_function,
                temperature=u,
                time=time,
            )
            self._temp_stream_function = self._solve_stream_function(
                initial_guess=self._stream_function,
                vorticity=self._vorticity,
                time=time,
            )
            diff = np.linalg.norm(
                self._temp_stream_function - self._stream_function, ord=2
            )
            self._stream_function = self._stream_function + alpha * (
                self._temp_stream_function - self._stream_function
            )
            if diff < self.implicit_lin_stopping_criteria:
                break

        return self._stream_function, self._vorticity

    def _solve_vorticity(
        self,
        old_vorticity: np.ndarray,
        stream_function: np.ndarray,
        temperature: np.ndarray,
        time: float,
    ) -> np.ndarray:
        return self.vorticity_solver.solve(
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
    ) -> np.ndarray:
        return self.stream_function_solver.solve(
            initial_guess=initial_guess, rhs=-vorticity, time=time
        )


class NonIterativeNavierStokersSolver:
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        sf_bcs: BoundaryConditions,
        sf_max_iters: int = 1000,
        sf_stopping_criteria: float = 1e-6,
    ):
        self.geometry = geometry
        self.parameters = parameters

        self.convective_operator = ConvectionOperator(
            form=ConvectiveTermForm.UPWIND, geometry=geometry
        )

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
            incorporated_bc=True,
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
            stream_function=sf,
            temperature=u,
            time=time,
        )
        self._stream_function = self._solve_stream_function(
            sf_nm1=sf,
            vorticity=self._temp_vorticity,
            time=time,
        )
        calculate_vorticity_from_sf(
            sf=self._stream_function,
            result=self._vorticity,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
        )

        return self._stream_function, self._vorticity

    def _solve_vorticity(
        self,
        old_vorticity: np.ndarray,
        stream_function: np.ndarray,
        temperature: np.ndarray,
        time: float,
    ) -> np.ndarray:
        return self.vorticity_solver.solve(
            w=old_vorticity,
            sf=stream_function,
            u=temperature,
            time=time,
        )

    def _solve_stream_function(
        self,
        sf_nm1: np.ndarray,
        vorticity: np.ndarray,
        time: float,
    ) -> np.ndarray:
        rho = self.vorticity_solver.rho
        c_ind = self.vorticity_solver.c_ind

        tau = self.geometry.dt * self.parameters.v / self.geometry.length_scale
        c = 0.5 * tau * (c_ind + rho / self.parameters.reynolds_number)
        f = (
            -vorticity
            - 0.5 * tau * (c_ind + rho / self.parameters.reynolds_number) * sf_nm1
        )
        return self.stream_function_solver.solve(
            initial_guess=sf_nm1, c=c, f=f, time=time
        )
