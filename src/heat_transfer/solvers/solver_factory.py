import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition
from src.convective_operator import ConvectiveTermForm, ConvectionOperator
from src.geometry import DomainGeometry
from src.heat_transfer.parameters import ThermalParameters
from src.heat_transfer.solvers.heat_transfer_solvers import *


class HeatTransferSolver:
    def __init__(
        self,
        solver_name: HeatTransferSolverName,
        geometry: DomainGeometry,
        parameters: ThermalParameters,
        convective_term_form: ConvectiveTermForm,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
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
            top_bc=top_bc,
            right_bc=right_bc,
            bottom_bc=bottom_bc,
            left_bc=left_bc,
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
