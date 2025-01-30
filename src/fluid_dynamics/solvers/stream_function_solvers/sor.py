import numpy as np
from numba import njit

from src.base_solver import BaseSolver
from src.boundary_conditions import BoundaryConditions, BoundaryConditionType
from src.fluid_dynamics.solvers.stream_function_solvers.registry import (
    register_sf_solver,
    StreamFunctionSolverName,
)
from src.geometry import DomainGeometry


@register_sf_solver(StreamFunctionSolverName.SOR)
class SORPoissonSolver(BaseSolver):
    """
    A solver for the Poisson equation using the Successive Over-Relaxation (SOR) method.

    This class extends the BaseSolver to provide functionality for solving the Poisson equation on a
    two-dimensional domain with specified boundary conditions. The solver uses the SOR method, which
    accelerates the convergence of the iterative Gauss-Seidel solver_name by applying an over-relaxation parameter.
    """

    def __init__(
        self,
        geometry: DomainGeometry,
        bcs: BoundaryConditions,
        max_iters: int = 50,
        stopping_criteria: float = 1e-6,
    ):
        """
        Initialize the SORPoissonSolver with domain geometry and boundary conditions.

        :param geometry: The computational domain's geometry.
        :param bcs: An object containing boundary conditions.
        :param max_iters: Maximum number of iterations for convergence. Default is 50.
        :param stopping_criteria: Convergence criteria for the solver. Default is 1e-6.
        """
        super().__init__(geometry=geometry, bcs=bcs)
        self._optimal_omega = self.calculate_omega()
        self.max_iters = max_iters
        self.stopping_criteria = stopping_criteria

        # Pre-allocate some arrays that will be used in the calculations
        self._result: np.ndarray = np.empty((self.geometry.n_y, self.geometry.n_x))

    def calculate_omega(self) -> float:
        """
        Calculate the optimal over-relaxation parameter (omega) for the Successive Over-Relaxation (SOR) method.

        The calculation is based on the grid size of the domain, represented by `n_x` and `n_y`, and aims to
        minimize the number of iterations required for convergence (refer to Frankel S.P. (1950)).

        :return: The optimal over-relaxation parameter (omega) for the SOR method.
        """
        zeta = (
            (
                np.cos(np.pi / (self.geometry.n_x - 1))
                + np.cos(np.pi / (self.geometry.n_y - 1))
            )
            / 2.0
        ) ** 2
        omega_opt = 2.0 * (1.0 - np.sqrt(1.0 - zeta)) / zeta

        return omega_opt

    @staticmethod
    @njit
    def _solve(
        rhs: np.ndarray,
        result: np.ndarray,
        dx: float,
        dy: float,
        max_iters: int,
        stopping_criteria: float,
        right_value: np.ndarray,
        left_value: np.ndarray,
        top_value: np.ndarray,
        bottom_value: np.ndarray,
        omega: float = 1.0,
    ) -> None:
        n_y, n_x = result.shape
        beta = dx / dy
        factor = 0.5 * omega / (1.0 + beta * beta)

        result[0, :] = top_value
        result[n_y - 1, :] = bottom_value
        result[:, 0] = left_value
        result[:, n_x - 1] = right_value

        for iteration in range(max_iters):
            temp = np.copy(result)
            for i in range(1, n_x - 1):
                for j in range(1, n_y - 1):
                    result[j, i] = (
                        factor
                        * (
                            temp[j, i + 1]
                            + result[j, i - 1]
                            + beta * beta * temp[j + 1, i]
                            + beta * beta * result[j - 1, i]
                            - dx * dx * rhs[j, i]
                        )
                        + (1.0 - omega) * temp[j, i]
                    )
            diff = np.linalg.norm(temp - result, ord=2)
            if diff < stopping_criteria:
                break

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
        self._solve(
            rhs=rhs,
            result=self._result,
            omega=self._optimal_omega,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
            max_iters=self.max_iters,
            stopping_criteria=self.stopping_criteria,
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
