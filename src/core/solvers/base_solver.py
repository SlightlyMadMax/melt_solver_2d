from typing import Optional

import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

from src.core.boundary_conditions import BoundaryConditions
from src.parameters.config import ExperimentConfig


class BaseSolver(ABC):
    def __init__(
        self,
        cfg: ExperimentConfig,
        bcs: Optional[BoundaryConditions] = None,
    ):
        self.cfg = cfg
        self.bcs = bcs

    @abstractmethod
    def solve(self, *args, **kwargs) -> NDArray[np.float64]: ...
