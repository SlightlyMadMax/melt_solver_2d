import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import cg, spilu, LinearOperator, norm, bicgstab

from src.base_solver import BaseSolver
from src.boundary_conditions import BoundaryConditions
from src.fluid_dynamics.solvers.stream_function_solvers.registry import (
    register_sf_solver,
    StreamFunctionSolverName,
)
from src.geometry import DomainGeometry


@register_sf_solver(StreamFunctionSolverName.CG)
class ConjugateGradientSolver(BaseSolver):
    """
    A solver for elliptic equations of the form Δu - c(x,y)u = -f(x,y) using the Conjugate Gradient method.

    This class extends the BaseSolver to provide functionality for solving elliptic equations on a
    two-dimensional domain with specified boundary conditions.
    """

    def __init__(
        self,
        geometry: DomainGeometry,
        bcs: BoundaryConditions,
        max_iters: int = 10000,
        stopping_criteria: float = 1e-6,
    ):
        """
        Initialize the ConjugateGradientSolver with domain geometry and boundary conditions.

        :param geometry: The computational domain's geometry.
        :param bcs: An object containing boundary conditions.
        :param max_iters: Maximum number of iterations for convergence. Default is 1000.
        :param stopping_criteria: Convergence criteria for the solver. Default is 1e-6.
        """
        super().__init__(geometry=geometry, bcs=bcs)
        self.max_iters = max_iters
        self.stopping_criteria = stopping_criteria

        # Pre-allocate some arrays that will be used in the calculations
        self._result: np.ndarray = np.empty((self.geometry.n_y, self.geometry.n_x))

    def _construct_rhs(
        self,
        f: np.ndarray,
        right_bc_value: np.ndarray,
        left_bc_value: np.ndarray,
        top_bc_value: np.ndarray,
        bottom_bc_value: np.ndarray,
    ) -> np.ndarray:
        dx2 = self.geometry.dx**2
        dy2 = self.geometry.dy**2
        rhs_inner = f[1:-1, 1:-1]
        rhs_inner[0, :] += top_bc_value[1:-1] / dy2
        rhs_inner[-1, :] += bottom_bc_value[1:-1] / dy2
        rhs_inner[:, 0] += left_bc_value[1:-1] / dx2
        rhs_inner[:, -1] += right_bc_value[1:-1] / dx2

        rhs_inner_flat = rhs_inner.flatten()

        return rhs_inner_flat

    def _spilu_preconditioner(self, A: csr_matrix):
        A_csc = A.tocsc()
        m_inv = spilu(A_csc, drop_tol=1e-4, fill_factor=20)
        m_in_op = LinearOperator(A.shape, m_inv.solve)
        return m_in_op

    def _jacobi_preconditioner(self, A):
        M_inv = diags(1 / A.diagonal())  # Inverse of diagonal elements
        return LinearOperator(A.shape, lambda x: M_inv @ x)

    def solve(
        self, A: csr_matrix, b: np.ndarray, initial_guess: np.ndarray, time: float
    ) -> np.ndarray:
        """
        Solve an elliptic equation of the form Δu - c(x,y)u = f(x,y) for a given initial guess, c(x,y) and f(x,y).

        :param initial_guess: The initial guess for the solution.
        :param b: The right-hand side of the equation.
        :param time: The current time, used to calculate time-dependent boundary conditions.
        :return: The final solution as a 2D NumPy array.
        """
        n_y, n_x = b.shape
        inner_n_y, inner_n_x = n_y - 2, n_x - 2

        right = self.bcs.right.get_value(t=time)
        left = self.bcs.left.get_value(t=time)
        top = self.bcs.top.get_value(t=time)
        bottom = self.bcs.bottom.get_value(t=time)

        rhs = self._construct_rhs(
            f=b,
            right_bc_value=right,
            left_bc_value=left,
            top_bc_value=top,
            bottom_bc_value=bottom,
        )
        # m = self._spilu_preconditioner(A=A)
        # m = self._incomp_chol_preconditioner(A=A)

        initial_guess_inner_flat = initial_guess[1:-1, 1:-1].flatten()

        solution_inner_flat, info = cg(
            A=A,
            b=rhs,
            # M=m,
            x0=initial_guess_inner_flat,
            maxiter=self.max_iters,
            rtol=self.stopping_criteria,
        )

        # solution_inner_flat, info = bicgstab(
        #     A=A,
        #     b=rhs,
        #     x0=initial_guess_inner_flat,
        #     M=m,
        #     maxiter=self.max_iters,
        #     rtol=self.stopping_criteria,
        # )

        if info != 0:
            raise RuntimeError(
                f"Conjugate Gradient did not converge. Info code: {info}"
            )

        solution_inner = solution_inner_flat.reshape((inner_n_y, inner_n_x))

        self._result[1:-1, 1:-1] = solution_inner
        self._result[:, 0] = left
        self._result[:, -1] = right
        self._result[0, :] = top
        self._result[-1, :] = bottom

        return self._result
