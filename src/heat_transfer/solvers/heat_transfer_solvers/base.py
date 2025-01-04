from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray

from src.base_solver import Sweep2DSolver
from src.heat_transfer.parameters import ThermalParameters


class HeatTransferSolver(Sweep2DSolver):
    def __init__(
        self,
        parameters: ThermalParameters,
        fixed_delta: bool = False,
        implicit_lin_max_iters: int = 5,
        implicit_lin_stopping_criteria: float = 1e-6,
        implicit_lin_urf: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.parameters = parameters
        self.fixed_delta = fixed_delta
        self.implicit_lin_max_iters = implicit_lin_max_iters
        self.implicit_lin_stopping_criteria = implicit_lin_stopping_criteria
        self.implicit_lin_urf = implicit_lin_urf

        # Pre-allocate some arrays that will be used in the calculations
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
        v_x: NDArray[np.float64],
        v_y: NDArray[np.float64],
        time: float = 0.0,
    ) -> NDArray[np.float64]: ...
