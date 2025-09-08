from abc import abstractmethod
from typing import Protocol, TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.utils.thomas import solve_tridiagonal

if TYPE_CHECKING:
    from src.parameters.config import ExperimentConfig


class HasConfig(Protocol):
    cfg: "ExperimentConfig"


class Sweep2DMixin:
    """
    Mixin that provides pre-allocated coefficient arrays and abstract sweep methods
    for 2D splitting schemes (e.g. Peaceman–Rachford, Douglas–Rachford).

    Assumes the consuming class defines a `.cfg` attribute.
    """

    _a_x: NDArray[np.float64]
    _b_x: NDArray[np.float64]
    _c_x: NDArray[np.float64]
    _a_y: NDArray[np.float64]
    _b_y: NDArray[np.float64]
    _c_y: NDArray[np.float64]
    _rhs_x: NDArray[np.float64]
    _rhs_y: NDArray[np.float64]

    def _initialize_sweep_arrays(self: HasConfig) -> None:
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        self._a_x = np.empty((n_y, n_x))
        self._b_x = np.empty((n_y, n_x))
        self._c_x = np.empty((n_y, n_x))
        self._a_y = np.empty((n_x, n_y))
        self._b_y = np.empty((n_x, n_y))
        self._c_y = np.empty((n_x, n_y))
        self._rhs_x = np.empty((n_y, n_x))
        self._rhs_y = np.empty((n_x, n_y))

    @abstractmethod
    def _compute_sweep_x_coeffs(self, *args, **kwargs) -> None:
        """
        Populate the coefficient and RHS arrays for the x-direction tridiagonal sweep.

        Fill the preallocated arrays `self._a_x`, `self._b_x`, `self._c_x` and `self._rhs_x`
        for the interior grid nodes so that a tridiagonal solve along x (at fixed y)
        can be performed by `_solve_sweep_x`.
        """
        ...

    @abstractmethod
    def _compute_sweep_y_coeffs(self, *args, **kwargs) -> None:
        """
        Populate the coefficient and RHS arrays for the y-direction tridiagonal sweep.

        Fill the preallocated arrays `self._a_y`, `self._b_y`, `self._c_y` and `self._rhs_y`
        for the interior grid nodes so that a tridiagonal solve along y (at fixed x)
        can be performed by `_solve_sweep_y`.
        """
        ...

    @abstractmethod
    def _apply_boundary_conditions_x(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def _apply_boundary_conditions_y(self, *args, **kwargs) -> None: ...

    @staticmethod
    def apply_dirichlet(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        rhs: np.ndarray,
        value: np.ndarray,
        side: int,
    ) -> None:
        """
        Enforce a Dirichlet boundary condition along one edge of a 2D tridiagonal coefficient grid
        (either left/right for x-sweep or top/bottom for y-sweep).
        :param a: sub-diagonal coefficients, shape (n_rows, n_cols) for x-sweep or (n_cols, n_rows) for y-sweep.
        :param b: diagonal coefficients, same shape as a.
        :param c: super-diagonal coefficients, same shape as a.
        :param rhs: right-hand side, same shape as a.
        :param value: Dirichlet values to impose along the boundary; should have length n_rows for x-sweep
        (one value per row) or length n_cols for y-sweep.
        :param side: boundary side: 0 for the "left" or "bottom" boundary (index 0), 1 for the "right" or "top" boundary (last index).
        :return: None
        """
        n, m = b.shape
        idx = 0 if side == 0 else m - 1
        a[:, idx] = 0.0
        b[:, idx] = 1.0
        c[:, idx] = 0.0
        rhs[:, idx] = value

    @staticmethod
    def apply_neumann_first_order(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        rhs: np.ndarray,
        flux: np.ndarray,
        side: int,
    ) -> None:
        """
        Enforce a Neumann boundary condition along one edge of a 2D tridiagonal coefficient grid
        (either left/right for x-sweep or top/bottom for y-sweep).
        This function uses first-order accuracy approximation.
        :param a: sub-diagonal coefficients, shape (n_rows, n_cols) for x-sweep or (n_cols, n_rows) for y-sweep.
        :param b: diagonal coefficients, same shape as a.
        :param c: super-diagonal coefficients, same shape as a.
        :param rhs: right-hand side, same shape as a.
        :param flux: flux values to impose along the boundary; should have length n_rows for x-sweep (one value per row) or length n_cols for y-sweep.
        :param side: boundary side: 0 for the "left" or "bottom" boundary (index 0), 1 for the "right" or "top" boundary (last index).
        :return: None
        """
        n, m = b.shape
        if side == 0:
            a[:, 0] = 1.0
            b[:, 0] = -1.0
            c[:, 0] = 0.0
            rhs[:, 0] = flux
        else:
            a[:, m - 1] = 0.0
            b[:, m - 1] = 1.0
            c[:, m - 1] = -1.0
            rhs[:, m - 1] = flux

    @staticmethod
    @njit
    def _solve_sweep_x(
        n: int,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
        result: NDArray[np.float64],
    ) -> None:
        for j in range(1, n - 1):
            solve_tridiagonal(
                a=a[j, :],
                b=b[j, :],
                c=c[j, :],
                f=rhs[j, :],
                result=result[j, :],
            )

    @staticmethod
    @njit
    def _solve_sweep_y(
        n: int,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
        result: NDArray[np.float64],
    ) -> None:
        for i in range(1, n - 1):
            solve_tridiagonal(
                a=a[i, :],
                b=b[i, :],
                c=c[i, :],
                f=rhs[i, :],
                result=result[:, i],
            )
