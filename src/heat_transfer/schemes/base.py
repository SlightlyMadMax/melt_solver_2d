from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition
from src.geometry import DomainGeometry
from src.solver import Sweep2DScheme
from src.heat_transfer.parameters import ThermalParameters


class HeatTransferScheme(Sweep2DScheme):
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: ThermalParameters,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
        fixed_delta: bool = False,
    ):
        super().__init__(
            geometry=geometry,
            top_bc=top_bc,
            right_bc=right_bc,
            bottom_bc=bottom_bc,
            left_bc=left_bc,
        )
        self.fixed_delta = fixed_delta
        self.parameters = parameters

        # Pre-allocate arrays, used for calculations
        self._temp_u: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._iter_u: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._new_u: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )

    @abstractmethod
    def solve(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        time: float = 0.0,
        iters: int = 1,
    ) -> NDArray[np.float64]: ...
