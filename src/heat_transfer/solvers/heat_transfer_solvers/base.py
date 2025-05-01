import numpy as np
from abc import abstractmethod
from typing import Optional
from numpy.typing import NDArray

from src.convective_operators import BaseConvectiveOperator
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.core.solvers.base_solver import BaseSolver
from src.core.solvers.mixins.iterative_solver import IterativeSolverMixin
from src.core.solvers.mixins.sweep_2d import Sweep2DMixin
from src.parameters.thermal import ThermalParameters


class BaseHeatTransferSolver(IterativeSolverMixin, BaseSolver):
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: ThermalParameters,
        convective_operator: BaseConvectiveOperator,
        bcs: Optional[BoundaryConditions] = None,
        fixed_delta: bool = False,
        max_iters: int = 5,
        tolerance: float = 1e-6,
        urf: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(geometry=geometry, bcs=bcs)

        self.parameters = parameters
        self.convective_operator = convective_operator
        self.fixed_delta = fixed_delta
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.urf = urf

        # Pre-allocate some arrays that will be used in the calculations
        self._iter_u: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._new_u: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )

    @abstractmethod
    def solve_linear(
        self, u: NDArray[np.float64], sf: NDArray[np.float64], time: float = 0.0
    ) -> None: ...


class ImplicitHeatTransferSolver(BaseHeatTransferSolver, Sweep2DMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._initialize_sweep_arrays()

        # Pre-allocate some arrays that will be used in the calculations
        self._temp_u: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )

    @abstractmethod
    def solve_linear(
        self, u: NDArray[np.float64], sf: NDArray[np.float64], time: float = 0.0
    ) -> None: ...


class ExplicitHeatTransferSolver(BaseHeatTransferSolver):
    @abstractmethod
    def solve_linear(
        self, u: NDArray[np.float64], sf: NDArray[np.float64], time: float = 0.0
    ) -> None: ...
