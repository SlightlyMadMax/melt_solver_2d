from typing import Protocol
import numpy as np
from numpy.typing import NDArray


class HasIterationParameters(Protocol):
    max_iters: int
    tolerance: float
    urf: float

    _iter_u: NDArray[np.float64]
    _new_u: NDArray[np.float64]

    def solve_linear(self, u: NDArray[np.float64], **kwargs) -> None: ...


class IterativeSolverMixin:
    """
    Mixin providing fixed-point iteration for nonlinear implicit solvers.

    Requirements:
    - `self.solve_linear(u: np.ndarray, **kwargs)`: fills `self._new_u`.
    - `self._iter_u`, `self._new_u`: arrays of shape (ny, nx).
    - Attributes:
        - max_iters: int
        - tolerance: float
        - urf: float
    """

    def solve(
        self: HasIterationParameters, u: NDArray[np.float64], **kwargs
    ) -> NDArray[np.float64]:
        """Runs fixed-point iterations until convergence or max iterations reached."""
        self._iter_u[:, :] = u  # start with previous solution
        last_diff = np.inf
        urf = self.urf

        for k in range(self.max_iters):
            self.solve_linear(u=u, **kwargs)

            # Under-relaxation
            self._iter_u[:, :] = urf * self._new_u + (1 - urf) * self._iter_u

            # Check for convergence
            norm_diff = np.linalg.norm(self._new_u - self._iter_u, ord=np.inf)
            if norm_diff < self.tolerance:
                break

            # Adaptive under-relaxation parameter
            if norm_diff > last_diff:
                urf = max(urf * 0.5, 1e-4)

            last_diff = norm_diff

        return self._new_u
