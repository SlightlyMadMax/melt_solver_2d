from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.solvers.stream_function_solvers import *
from src.fluid_dynamics.solvers.vorticity_solvers import *
from src.fluid_dynamics.utils import compute_velocity_from_sf
from src.geometry import DomainGeometry


class NavierStokesSolver:
    def __init__(
        self,
        vorticity_solver_name: VorticitySolverName,
        stream_function_solver_name: StreamFunctionSolverName,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        sf_top_bc: BoundaryCondition,
        sf_right_bc: BoundaryCondition,
        sf_bottom_bc: BoundaryCondition,
        sf_left_bc: BoundaryCondition,
        sf_max_iters: int = 50,
        sf_stopping_criteria: float = 1e-6,
        implicit_lin_max_iters: int = 5,
        implicit_lin_stopping_criteria: float = 1e-6,
        implicit_lin_urf: float = 0.5,
        vorticity_bc_order: int = 2,
    ):
        self.geometry = geometry
        self.implicit_lin_max_iters = implicit_lin_max_iters
        self.implicit_lin_stopping_criteria = implicit_lin_stopping_criteria
        self.implicit_lin_urf = implicit_lin_urf

        if vorticity_bc_order < 1 or vorticity_bc_order > 2:
            raise NotImplementedError(
                "Only 1st and 2nd order accuracy boundary conditions are available at the moment for the vorticity "
                "transport equation."
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
            max_iters=sf_max_iters,
            stopping_criteria=sf_stopping_criteria,
            bc_order=vorticity_bc_order,
        )
        self.stream_function_solver = stream_function_solver_class(
            geometry=geometry,
            top_bc=sf_top_bc,
            right_bc=sf_right_bc,
            bottom_bc=sf_bottom_bc,
            left_bc=sf_left_bc,
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        v_x, v_y = compute_velocity_from_sf(
            sf=self._stream_function,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
        )
        return self._stream_function, self._vorticity, v_x, v_y

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
            initial_guess=initial_guess, rhs=vorticity, time=time
        )
