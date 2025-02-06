from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.geometry import DomainGeometry


class ConvectiveTermForm(Enum):
    DIVERGENT_CENTRAL = "Divergent central"
    NON_DIVERGENT_CENTRAL = "Non-divergent central"
    SYMMETRIC = "Symmetric"
    UPWIND = "Upwind"


class BaseConvectiveOperator(ABC):
    @abstractmethod
    def __call__(
        self, *args, **kwargs
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def __init__(self, geometry: DomainGeometry, n_points: Optional[int] = 3):
        self.geometry = geometry

        self._result_x: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x, n_points)
        )
        self._result_y: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x, n_points)
        )

    @staticmethod
    @njit
    def _restrict(
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        u: NDArray[np.float64],
        u_pt: float,
    ):
        n_y, n_x = u.shape

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                if u[j, i] < u_pt:
                    conv_x[j, i, :] = 0.0
                    conv_y[j, i, :] = 0.0
