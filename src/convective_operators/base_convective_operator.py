import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from numpy.typing import NDArray

from src.parameters.config import ExperimentConfig


class ConvectiveTermForm(Enum):
    DIVERGENT_CENTRAL = "Divergent central"
    NON_DIVERGENT_CENTRAL = "Non-divergent central"
    SYMMETRIC = "Symmetric"
    UPWIND = "Upwind"


class BaseConvectiveOperator(ABC):
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(
        self,
        conv_x: NDArray[np.float64],
        conv_y: NDArray[np.float64],
        **kwargs,
    ): ...
