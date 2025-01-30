import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryConditions
from src.convective_operator import ConvectiveTermForm, ConvectionOperator
from src.geometry import DomainGeometry
from src.heat_transfer.parameters import ThermalParameters
from src.heat_transfer.solvers.heat_transfer_solvers import *


class HeatTransferSolver:
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: ThermalParameters,
        bcs: BoundaryConditions,
        solver_name: HeatTransferSolverName = HeatTransferSolverName.PEACEMAN_RACHFORD,
        convective_term_form: ConvectiveTermForm = ConvectiveTermForm.UPWIND,
        fixed_delta: bool = False,
        implicit_lin_max_iters: int = 5,
        implicit_lin_stopping_criteria: float = 1e-6,
        implicit_lin_urf: float = 0.5,
    ):
        solver_class = HeatTransferSolverRegistry.get_solver_class(solver_name)

        self.convective_operator = ConvectionOperator(
            form=convective_term_form, geometry=geometry
        )

        self.solver = solver_class(
            geometry=geometry,
            parameters=parameters,
            convective_operator=self.convective_operator,
            bcs=bcs,
            fixed_delta=fixed_delta,
            implicit_lin_max_iters=implicit_lin_max_iters,
            implicit_lin_stopping_criteria=implicit_lin_stopping_criteria,
            implicit_lin_urf=implicit_lin_urf,
        )

    def solve(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        time: float = 0.0,
    ):
        return self.solver.solve(u, sf, time)
