from typing import Optional

import numpy as np
from numba import njit
from numpy.typing import NDArray
from abc import ABC, abstractmethod

from src.boundary_conditions import BoundaryConditions
from src.geometry import DomainGeometry


class BaseSolver(ABC):
    def __init__(
        self,
        geometry: DomainGeometry,
        bcs: Optional[BoundaryConditions] = None,
    ):
        self.geometry = geometry
        self.bcs = bcs

    @abstractmethod
    def solve(self, *args, **kwargs) -> NDArray[np.float64]: ...


class Sweep2DSolver(BaseSolver, ABC):
    def __init__(
        self,
        geometry: DomainGeometry,
        bcs: Optional[BoundaryConditions] = None,
    ):
        super().__init__(
            geometry=geometry,
            bcs=bcs,
        )

        # Pre-allocate some arrays that will be used for calculations
        self._a_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._b_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._c_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._a_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))
        self._b_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))
        self._c_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))
        self._rhs_x: NDArray[np.float64] = np.empty(self.geometry.n_x)
        self._rhs_y: NDArray[np.float64] = np.empty(self.geometry.n_y)

    @staticmethod
    @abstractmethod
    @njit
    def _compute_sweep_x(*args, **kwargs) -> NDArray[np.float64]: ...

    @staticmethod
    @abstractmethod
    @njit
    def _compute_sweep_y(*args, **kwargs) -> NDArray[np.float64]: ...
