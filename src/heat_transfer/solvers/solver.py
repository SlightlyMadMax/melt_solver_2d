import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition
from src.geometry import DomainGeometry
from src.heat_transfer.parameters import ThermalParameters
from src.heat_transfer.solvers.registry import (
    HeatTransferSchemeName,
    HeatTransferSchemeRegistry,
)
from src.heat_transfer.solvers.schemes import *  # noqa, automatically register all of the schemes


class HeatTransferSolver:
    def __init__(
        self,
        scheme: HeatTransferSchemeName,
        geometry: DomainGeometry,
        parameters: ThermalParameters,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
        fixed_delta: bool = False,
        implicit_lin_max_iters: int = 5,
        implicit_lin_stopping_criteria: float = 1e-6,
        implicit_lin_urf: float = 0.5,
    ):
        self.scheme = scheme
        scheme_class = HeatTransferSchemeRegistry.get_scheme_class(self.scheme)

        self.solver = scheme_class(
            geometry=geometry,
            parameters=parameters,
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
