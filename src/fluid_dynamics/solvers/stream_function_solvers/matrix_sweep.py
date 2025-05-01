import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

from src.core.geometry import DomainGeometry
from src.core.boundary_conditions import BoundaryConditionType, BoundaryConditions
from src.core.solvers.base_solver import BaseSolver
from src.fluid_dynamics.solvers.stream_function_solvers.registry import (
    register_sf_solver,
    StreamFunctionSolverName,
)


@register_sf_solver(StreamFunctionSolverName.MATRIX_SWEEP)
class MatrixSweepPoissonSolver(BaseSolver):
    """
    A solver for the Poisson equation using the Matrix Sweep Algorithm (2D Thomas algorithm).

    This class extends the BaseSolver to provide functionality for solving the Poisson equation on a
    two-dimensional domain with specified boundary conditions.
    """

    def __init__(
        self,
        geometry: DomainGeometry,
        bcs: BoundaryConditions,
        *args,
        **kwargs,
    ):
        """
        Initialize the MatrixSweepPoissonSolver with domain geometry and boundary conditions.

        :param geometry: The computational domain's geometry.
        :param bcs: An object containing boundary conditions.
        """
        super().__init__(geometry=geometry, bcs=bcs)

        # Pre-allocate some arrays that will be used in the calculations
        self._result: np.ndarray = np.empty((self.geometry.n_y, self.geometry.n_x))
        self.alpha: list = self._init_alpha()

    def _init_alpha(self):
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        mu = (self.geometry.dx / self.geometry.dy) ** 2

        diagonals = [
            [-2 - 2 * mu] * (n_y - 2),
            [mu] * (n_y - 3),
            [mu] * (n_y - 3),
        ]
        a = diags(diagonals, offsets=[0, 1, -1], format="csc")

        alpha_m = diags([0] * (n_y - 2), format="csc")

        alpha: list = [alpha_m]

        for m in range(1, n_x - 1):
            alpha_m1 = spsolve(-a - alpha_m, np.eye(n_y - 2))
            alpha_m = csr_matrix(alpha_m1)
            alpha.append(alpha_m)

        return alpha

    def _matrix_sweep(
        self,
        f: np.ndarray,
        right_value: np.ndarray,
        left_value: np.ndarray,
        top_value: np.ndarray,
        bottom_value: np.ndarray,
    ):
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        mu = (self.geometry.dx / self.geometry.dy) ** 2
        beta_m = left_value[1:-1]
        beta_m_list = [beta_m]

        for m in range(1, n_x - 1):
            f_m = f[1:-1, m] * self.geometry.dx**2
            f_m[0] -= mu * top_value[m]
            f_m[-1] -= mu * bottom_value[m]

            beta_m1 = self.alpha[m] @ (beta_m - f_m)
            beta_m = beta_m1
            beta_m_list.append(beta_m)

        self._result[1:-1, n_x - 1] = right_value[1:-1]

        for m in range(n_x - 1, 0, -1):
            self._result[1:-1, m - 1] = (
                self.alpha[m - 1] @ self._result[1:-1, m] + beta_m_list[m - 1]
            )

        self._result[0, :] = top_value
        self._result[-1, :] = bottom_value

    def solve(
        self, initial_guess: np.ndarray, rhs: np.ndarray, time: float
    ) -> np.ndarray:
        """
        Solve the Poisson equation for a given initial guess and right-hand side.

        :param initial_guess: The initial guess for the solution.
        :param rhs: The right-hand side of the Poisson equation.
        :param time: The current time, used to calculate time-dependent boundary conditions.
        :return: The final solution as a 2D NumPy array.
        """
        self._result = np.copy(initial_guess)
        self._matrix_sweep(
            f=rhs,
            right_value=(
                self.bcs.right.get_value(t=time)
                if self.bcs.right.boundary_type == BoundaryConditionType.DIRICHLET
                else None
            ),
            left_value=(
                self.bcs.left.get_value(t=time)
                if self.bcs.left.boundary_type == BoundaryConditionType.DIRICHLET
                else None
            ),
            top_value=(
                self.bcs.top.get_value(t=time)
                if self.bcs.top.boundary_type == BoundaryConditionType.DIRICHLET
                else None
            ),
            bottom_value=(
                self.bcs.bottom.get_value(t=time)
                if self.bcs.bottom.boundary_type == BoundaryConditionType.DIRICHLET
                else None
            ),
        )

        return self._result
