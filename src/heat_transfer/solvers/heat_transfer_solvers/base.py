import numpy as np

from abc import abstractmethod
from typing import Optional
from numba import njit
from numpy.typing import NDArray

from src.convective_operators import BaseConvectiveOperator
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.core.solvers.base_solver import BaseSolver
from src.core.solvers.mixins.iterative_solver import IterativeSolverMixin
from src.core.solvers.mixins.sweep_2d import Sweep2DMixin
from src.heat_transfer.coefficient_smoothing.coefficients import c_smoothed, k_smoothed
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
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        # Pre-allocate some arrays that will be used in the calculations
        self._iter_u: NDArray[np.float64] = np.empty(n_y, n_x)
        self._new_u: NDArray[np.float64] = np.empty(n_y, n_x)
        self._conv_x: NDArray[np.float64] = np.empty((n_y, n_x, 3))
        self._conv_y: NDArray[np.float64] = np.empty((n_y, n_x, 3))
        self._c_eff = np.empty((n_y, n_x))
        self._k_eff = np.empty((n_y, n_x))

    @abstractmethod
    def solve_linear(
        self, u: NDArray[np.float64], sf: NDArray[np.float64], time: float = 0.0
    ) -> None: ...

    @staticmethod
    @njit
    def compute_effective_properties(
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        u: NDArray[np.float64],
        u_ref: float,
        u_pt: float,
        delta_u: float,
        c_ref: float,
        c_solid: float,
        c_liquid: float,
        l_solid: float,
        k_ref: float,
        k_solid: float,
        k_liquid: float,
        delta: NDArray[np.float64],
        # delta: float,
    ) -> None:
        n_y, n_x = u.shape
        inv_c_ref = 1.0 / c_ref
        inv_k_ref = 1.0 / k_ref

        for j in range(n_y):
            for i in range(n_x):
                T = u[j, i] * delta_u + u_ref

                c_eff[j, i] = (
                    c_smoothed(
                        u=T,
                        u_pt=u_pt,
                        c_solid=c_solid,
                        c_liquid=c_liquid,
                        l_solid=l_solid,
                        # delta=delta,
                        delta=delta[j, i],
                    )
                    * inv_c_ref
                )

                k_eff[j, i] = (
                    k_smoothed(
                        u=T,
                        u_pt=u_pt,
                        k_solid=k_solid,
                        k_liquid=k_liquid,
                        # delta=delta,
                        delta=delta[j, i],
                    )
                    * inv_k_ref
                )


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
