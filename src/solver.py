from enum import Enum

import numba
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

from src.boundary_conditions import BoundaryCondition
from src.geometry import DomainGeometry


class SchemeType(Enum):
    EXPLICIT = 1, "Explicit"
    IMPLICIT = 2, "Implicit"
    SEMI_IMPLICIT = 3, "Semi-implicit"
    LINEARIZED_IMPLICIT = 4, "Linearized implicit"


class BaseScheme(ABC):
    def __init__(
        self,
        geometry: DomainGeometry,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
    ):
        self.geometry = geometry
        self.top_bc = top_bc
        self.right_bc = right_bc
        self.bottom_bc = bottom_bc
        self.left_bc = left_bc

    @abstractmethod
    def solve(self, *args, **kwargs) -> NDArray[np.float64]: ...


class Sweep2DScheme(BaseScheme, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pre-allocate some arrays that will be used for calculations
        self._a_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._b_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._c_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._a_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))
        self._b_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))
        self._c_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))

    @staticmethod
    @abstractmethod
    @numba.jit(nopython=True)
    def _compute_sweep_x(*args, **kwargs) -> NDArray[np.float64]: ...

    @staticmethod
    @abstractmethod
    @numba.jit(nopython=True)
    def _compute_sweep_y(*args, **kwargs) -> NDArray[np.float64]: ...
