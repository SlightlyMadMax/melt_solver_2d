import numpy as np
from numpy.typing import NDArray

from src.convective_operators import (
    ConvectiveTermForm,
    VorticityTransportOperator,
)
from src.core.boundary_conditions import BoundaryConditions
from src.heat_transfer.coefficient_smoothing.coefficients import StepScheme, DeltaScheme
from src.heat_transfer.solvers.heat_transfer_solvers import *
from src.parameters.config import ExperimentConfig


class HeatTransferSolver:
    def __init__(
        self,
        cfg: ExperimentConfig,
        bcs: BoundaryConditions,
        solver_name: HeatTransferSolverName = HeatTransferSolverName.PEACEMAN_RACHFORD,
        convective_term_form: ConvectiveTermForm = ConvectiveTermForm.UPWIND,
        max_iters: int = 1,
        tolerance: float = 1e-6,
        urf: float = 1.0,
        bc_order: int = 1,
        step_scheme: StepScheme = StepScheme.ERF,
        delta_scheme: DeltaScheme = DeltaScheme.GAUSS,
    ):
        if bc_order not in (1, 2):
            raise NotImplementedError(
                "Only 1st and 2nd order accuracy BCs are supported for the heat equation."
            )

        solver_class = HeatTransferSolverRegistry.get_solver_class(solver_name)

        self.convective_operator = VorticityTransportOperator(
            form=convective_term_form, cfg=cfg
        )

        self.solver = solver_class(
            cfg=cfg,
            convective_operator=self.convective_operator,
            bcs=bcs,
            max_iters=max_iters,
            tolerance=tolerance,
            urf=urf,
            bc_order=bc_order,
            step_scheme=step_scheme,
            delta_scheme=delta_scheme,
        )

    def solve(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        delta: float,
        time: float = 0.0,
    ):
        return self.solver.solve(u=u, sf=sf, delta=delta, time=time)
