from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray

from src.solver import Sweep2DScheme
from src.heat_transfer.parameters import ThermalParameters


class HeatTransferScheme(Sweep2DScheme):
    def __init__(
        self,
        parameters: ThermalParameters,
        fixed_delta: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.parameters = parameters
        self.fixed_delta = fixed_delta

        # Pre-allocate some arrays that will be used for calculations
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
