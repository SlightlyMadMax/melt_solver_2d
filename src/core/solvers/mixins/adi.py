from abc import abstractmethod
from typing import Protocol, TYPE_CHECKING, Optional

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.core.solvers.tridiagonal_solver import solve_tridiagonal

if TYPE_CHECKING:
    from src.parameters.config import ExperimentConfig


class HasConfig(Protocol):
    cfg: "ExperimentConfig"


class ADIMixin:
    """
    Mixin that provides complete ADI functionality for 2D splitting schemes
    (e.g. Peaceman–Rachford, Douglas–Rachford).

    Provides:
    - Pre-allocated coefficient arrays
    - Abstract methods for computing coefficients and applying BCs
    - Tridiagonal solve methods for x and y directions
    - Full sweep orchestration with optional direction alternation

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
        """Initialize coefficient and RHS arrays for both sweep directions."""
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
    def _apply_boundary_conditions_x(self, *args, **kwargs) -> None:
        """Apply boundary conditions to x-direction sweep."""
        ...

    @abstractmethod
    def _apply_boundary_conditions_y(self, *args, **kwargs) -> None:
        """Apply boundary conditions to y-direction sweep."""
        ...

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
        Enforce Dirichlet boundary condition along one edge.

        :param a: sub-diagonal coefficients
        :param b: diagonal coefficients
        :param c: super-diagonal coefficients
        :param rhs: right-hand side
        :param value: Dirichlet values to impose along the boundary
        :param side: 0 for left/bottom boundary, 1 for right/top boundary
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
        Enforce Neumann boundary condition along one edge (first-order accurate).

        :param a: sub-diagonal coefficients
        :param b: diagonal coefficients
        :param c: super-diagonal coefficients
        :param rhs: right-hand side
        :param flux: flux values to impose along the boundary
        :param side: 0 for left/bottom boundary, 1 for right/top boundary
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
        """Solve tridiagonal systems along x-direction for all y rows."""
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
        """Solve tridiagonal systems along y-direction for all x columns."""
        for i in range(1, n - 1):
            solve_tridiagonal(
                a=a[i, :],
                b=b[i, :],
                c=c[i, :],
                f=rhs[i, :],
                result=result[:, i],
            )

    def _execute_adi_step(
        self,
        result: NDArray[np.float64],
        time: float,
        coeff_kwargs: Optional[dict] = None,
        hook_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Execute the full ADI sweep sequence.

        :param result: Array to store the solution (modified in-place)
        :param time: Current physical time (for time-dependent BCs)
        :param coeff_kwargs: Arguments for _compute_sweep_*_coeffs methods
        :param hook_kwargs: Arguments for _between_sweeps_hook
        """
        n_y, n_x = result.shape
        coeff_kwargs = coeff_kwargs or {}
        hook_kwargs = hook_kwargs or {}

        # X-direction sweep
        self._compute_sweep_x_coeffs(**coeff_kwargs)
        self._apply_boundary_conditions_x(time=time)
        self._solve_sweep_x(
            n=n_y,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            result=result,
        )

        # Optional hook after the first sweep
        self._after_first_sweep(result=result, **hook_kwargs)

        # Y-direction sweep
        self._compute_sweep_y_coeffs(**coeff_kwargs)
        self._apply_boundary_conditions_y(time=time)
        self._solve_sweep_y(
            n=n_x,
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            result=result,
        )

        # Optional hook after the second sweep
        self._after_second_sweep(result=result, **coeff_kwargs)

    def _after_first_sweep(self, result: NDArray[np.float64], **kwargs) -> None:
        """
        Hook called after the first sweep. Override to add custom logic.
        """
        pass

    def _after_second_sweep(self, result: NDArray[np.float64], **kwargs) -> None:
        """
        Hook called after the second sweep. Override to add custom logic.
        """
        pass
