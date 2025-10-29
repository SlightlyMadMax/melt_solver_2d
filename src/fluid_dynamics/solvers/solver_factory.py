import numpy as np

from typing import Tuple
from numpy.typing import NDArray


from src.convective_operators import (
    ConvectiveTermForm,
    StreamFunctionBasedConvectiveOperator,
)
from src.core.boundary_conditions import BoundaryConditions
from src.fluid_dynamics.solvers.stream_function_solvers import *
from src.fluid_dynamics.solvers.vorticity_solvers import *

from src.parameters.config import ExperimentConfig


class IterativeNavierStokesSolver:
    def __init__(
        self,
        cfg: ExperimentConfig,
        sf_bcs: BoundaryConditions,
        vorticity_solver_name: VorticitySolverName = VorticitySolverName.PEACEMAN_RACHFORD,
        stream_function_solver_name: StreamFunctionSolverName = StreamFunctionSolverName.MATRIX_SWEEP,
        convective_term_form: ConvectiveTermForm = ConvectiveTermForm.UPWIND,
        sf_max_iters: int = 1000,
        sf_stopping_criteria: float = 1e-6,
        max_iters: int = 5,
        tolerance: float = 1e-6,
        urf: float = 0.5,
        bc_order: int = 2,
    ):
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.urf = urf
        convective_operator = StreamFunctionBasedConvectiveOperator(
            form=convective_term_form, cfg=cfg
        )
        n_y, n_x = cfg.geometry.n_y, cfg.geometry.n_x

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
            cfg=cfg,
            convective_operator=convective_operator,
            bc_order=bc_order,
        )
        self.stream_function_solver = stream_function_solver_class(
            cfg=cfg,
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
        delta: float,
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
                delta=delta,
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
        initial_guess: np.ndarray,
        vorticity: np.ndarray,
        time: float,
    ) -> None:
        self._stream_function[:, :] = self.stream_function_solver.solve(
            initial_guess=initial_guess, rhs=-vorticity, time=time
        )
