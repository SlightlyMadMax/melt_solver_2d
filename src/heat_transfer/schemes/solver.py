import typing
import numpy as np
from enum import Enum
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition
from src.geometry import DomainGeometry
from src.heat_transfer.parameters import ThermalParameters
from src.heat_transfer.schemes.douglas_rachford import DouglasRachfordScheme
from src.heat_transfer.schemes.loc_one_dim import LocOneDimScheme
from src.heat_transfer.schemes.peaceman_rachford import PeacemanRachfordScheme


class HeatTransferSchemes(Enum):
    LOC_ONE_DIM = 1, "Locally one dimensional"
    DOUGLAS_RACHFORD = 2, "Douglas-Rachford"
    PEACEMAN_RACHFORD = 3, "Peaceman-Rachford"


class HeatTransferSolver:
    def __init__(
        self,
        scheme: HeatTransferSchemes,
        geometry: DomainGeometry,
        parameters: ThermalParameters,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
        fixed_delta: bool = False,
    ):
        self.scheme = scheme
        SchemeClass: typing.Type = self.get_scheme_class()
        self.solver = SchemeClass(
            geometry=geometry,
            parameters=parameters,
            top_bc=top_bc,
            right_bc=right_bc,
            bottom_bc=bottom_bc,
            left_bc=left_bc,
            fixed_delta=fixed_delta,
        )

    def get_scheme_class(self) -> typing.Type:
        if self.scheme == HeatTransferSchemes.LOC_ONE_DIM:
            return LocOneDimScheme
        elif self.scheme == HeatTransferSchemes.DOUGLAS_RACHFORD:
            return DouglasRachfordScheme
        elif self.scheme == HeatTransferSchemes.PEACEMAN_RACHFORD:
            return PeacemanRachfordScheme
        else:
            raise NotImplemented()

    def solve(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        time: float = 0.0,
        iters: int = 1,
    ):
        return self.solver.solve(u, sf, time, iters)
