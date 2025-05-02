import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from numba import njit
from numpy.typing import NDArray

from src.core.geometry import DomainGeometry


class ConvectiveTermForm(Enum):
    DIVERGENT_CENTRAL = "Divergent central"
    NON_DIVERGENT_CENTRAL = "Non-divergent central"
    SYMMETRIC = "Symmetric"
    UPWIND = "Upwind"


class BaseConvectiveOperator(ABC):
    def __init__(self, geometry: DomainGeometry):
        self.geometry = geometry

    @abstractmethod
    def __call__(
        self,
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        **kwargs,
    ): ...

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
