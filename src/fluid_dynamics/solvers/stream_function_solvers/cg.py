import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, spilu, cg

from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.core.solvers.base_solver import BaseSolver
from src.fluid_dynamics.solvers.stream_function_solvers.registry import (
    register_sf_solver,
    StreamFunctionSolverName,
)
from src.parameters.config import ExperimentConfig


@register_sf_solver(StreamFunctionSolverName.CG)
class ConjugateGradientSolver(BaseSolver):
    """
    A solver for elliptic equations of the form Δu - c(x,y)u = -f(x,y) using the Conjugate Gradient method.

    This class extends the BaseSolver to provide functionality for solving elliptic equations on a
    two-dimensional domain with specified boundary conditions.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        bcs: BoundaryConditions,
        max_iters: int = 10000,
        stopping_criteria: float = 1e-6,
    ):
        """
        Initialize the ConjugateGradientSolver with domain geometry and boundary conditions.

        :param cfg: The configuration of the experiment (domain geometry, material properties, etc.).
        :param bcs: An object containing boundary conditions.
        :param max_iters: Maximum number of iterations for convergence. Default is 10000.
        :param stopping_criteria: Convergence criteria for the solver. Default is 1e-6.
        """
        super().__init__(cfg=cfg, bcs=bcs)
        self.geometry: DomainGeometry = cfg.geometry
        self.max_iters = max_iters
        self.stopping_criteria = stopping_criteria

        # Pre-allocate some arrays that will be used in the calculations
        self._result: np.ndarray = np.empty((self.geometry.n_y, self.geometry.n_x))

    def _get_ilu_preconditioner(self, A):
        A_csc = A.tocsc()
        ilu = spilu(A_csc, drop_tol=1e-4, fill_factor=10)
        return LinearOperator(A.shape, ilu.solve)

    def _get_jacobi_preconditioner(self, A) -> LinearOperator:
        diag = A.diagonal()
        if np.any(diag == 0):
            raise ZeroDivisionError("Jacobi preconditioner: zero on diagonal of A")
        inv_diag = 1.0 / diag

        def matvec(x):
            return inv_diag * x

        return LinearOperator(A.shape, matvec=matvec)

    def solve(
        self,
        A: diags,
        b_flat: np.ndarray,
        initial_guess: np.ndarray,
        time: float,
        use_ilu: bool = False,
    ) -> np.ndarray:
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        inner_n_y, inner_n_x = n_y - 2, n_x - 2

        right = self.bcs.right.get_value(t=time)
        left = self.bcs.left.get_value(t=time)
        top = self.bcs.top.get_value(t=time)
        bottom = self.bcs.bottom.get_value(t=time)

        # Initial guess interior flattened
        x0 = initial_guess[1:-1, 1:-1].flatten()

        # Choose preconditioner
        if use_ilu:
            preconditioner = self._get_ilu_preconditioner(A)
        else:
            preconditioner = self._get_jacobi_preconditioner(A)

        # Solve A x = b
        solution_inner_flat, info = cg(
            A=A,
            b=b_flat,
            x0=x0,
            M=preconditioner,
            maxiter=self.max_iters,
            rtol=self.stopping_criteria,
        )

        if info > 0:
            raise RuntimeError(f"CG did not converge after {info} iterations")
        elif info < 0:
            raise RuntimeError(f"CG error: {info}")

        self._result[1:-1, 1:-1] = solution_inner_flat.reshape((inner_n_y, inner_n_x))
        self._result[:, 0] = left
        self._result[:, -1] = right
        self._result[0, :] = top
        self._result[-1, :] = bottom

        return self._result
