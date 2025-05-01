import numpy as np
from numpy.typing import NDArray

from src.convective_operators import (
    ConvectiveTermForm,
    ConvectiveVorticityTransportOperator,
)
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.heat_transfer.solvers.heat_transfer_solvers import *
from src.parameters.thermal import ThermalParameters


class HeatTransferSolver:
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: ThermalParameters,
        bcs: BoundaryConditions,
        solver_name: HeatTransferSolverName = HeatTransferSolverName.PEACEMAN_RACHFORD,
        convective_term_form: ConvectiveTermForm = ConvectiveTermForm.UPWIND,
        fixed_delta: bool = False,
        max_iters: int = 5,
        tolerance: float = 1e-6,
        urf: float = 0.5,
    ):
        solver_class = HeatTransferSolverRegistry.get_solver_class(solver_name)

        self.convective_operator = ConvectiveVorticityTransportOperator(
            form=convective_term_form, geometry=geometry
        )

        self.solver = solver_class(
            geometry=geometry,
            parameters=parameters,
            convective_operator=self.convective_operator,
            bcs=bcs,
            fixed_delta=fixed_delta,
            max_iters=max_iters,
            tolerance=tolerance,
            urf=urf,
        )

    def solve(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        time: float = 0.0,
    ):
        return self.solver.solve(u, sf, time)
