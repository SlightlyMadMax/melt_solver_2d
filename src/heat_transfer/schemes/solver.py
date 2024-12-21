import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition
from src.geometry import DomainGeometry
from src.heat_transfer.parameters import ThermalParameters
from src.heat_transfer.schemes.registry import (
    HeatTransferSchemeName,
    HeatTransferSchemeRegistry,
)


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
        )

    def solve(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        time: float = 0.0,
        iters: int = 1,
    ):
        return self.solver.solve(u, sf, time, iters)
