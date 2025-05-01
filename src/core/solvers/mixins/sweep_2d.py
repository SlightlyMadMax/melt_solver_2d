from abc import abstractmethod
from typing import Protocol, TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.core.geometry import DomainGeometry


class HasGeometry(Protocol):
    geometry: "DomainGeometry"


class Sweep2DMixin:
    """
    Mixin that provides pre-allocated coefficient arrays and abstract sweep methods
    for 2D splitting schemes (e.g. Peaceman–Rachford, Douglas–Rachford).

    Assumes the consuming class defines a `.geometry` attribute.
    """

    _a_x: NDArray[np.float64]
    _b_x: NDArray[np.float64]
    _c_x: NDArray[np.float64]
    _a_y: NDArray[np.float64]
    _b_y: NDArray[np.float64]
    _c_y: NDArray[np.float64]
    _rhs_x: NDArray[np.float64]
    _rhs_y: NDArray[np.float64]

    def _initialize_sweep_arrays(self: HasGeometry) -> None:
        """Preallocate arrays used for Thomas algorithm in X and Y directions."""
        self._a_x = np.empty(self.geometry.n_x - 1)
        self._b_x = np.empty(self.geometry.n_x - 1)
        self._c_x = np.empty(self.geometry.n_x - 1)
        self._a_y = np.empty(self.geometry.n_y - 1)
        self._b_y = np.empty(self.geometry.n_y - 1)
        self._c_y = np.empty(self.geometry.n_y - 1)
        self._rhs_x = np.empty(self.geometry.n_x)
        self._rhs_y = np.empty(self.geometry.n_y)

    @staticmethod
    @abstractmethod
    @njit
    def _compute_sweep_x(*args, **kwargs) -> NDArray[np.float64]:
        """Override in subclass with X-direction sweep implementation."""
        ...

    @staticmethod
    @abstractmethod
    @njit
    def _compute_sweep_y(*args, **kwargs) -> NDArray[np.float64]:
        """Override in subclass with Y-direction sweep implementation."""
        ...
