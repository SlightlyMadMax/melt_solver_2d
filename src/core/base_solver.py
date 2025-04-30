from typing import Optional

import numpy as np
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
