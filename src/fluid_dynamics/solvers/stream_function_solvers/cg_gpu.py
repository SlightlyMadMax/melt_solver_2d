import numpy as np
import cupy as cp

from cupyx.scipy.sparse.linalg import cg, LinearOperator

from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.core.solvers.base_solver import BaseSolver
from src.fluid_dynamics.solvers.stream_function_solvers.registry import (
    register_sf_solver,
    StreamFunctionSolverName,
)
from src.parameters.config import ExperimentConfig


@register_sf_solver(StreamFunctionSolverName.CG_GPU)
class ConjugateGradientGPUSolver(BaseSolver):
    """
    A solver for elliptic equations using the Conjugate Gradient method with GPU acceleration.

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
        A: np.ndarray,
        b_flat: np.ndarray,
        initial_guess: np.ndarray,
        time: float,
    ) -> np.ndarray:
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        inner_slice = (slice(1, -1), slice(1, -1))
        inner_n_y, inner_n_x = n_y - 2, n_x - 2

        self._result[:, 0] = self.bcs.left.get_value(t=time)
        self._result[:, -1] = self.bcs.right.get_value(t=time)
        self._result[0, :] = self.bcs.top.get_value(t=time)
        self._result[-1, :] = self.bcs.bottom.get_value(t=time)

        # Initial guess interior flattened
        x0 = initial_guess[inner_slice].ravel()

        # Convert numpy arrays into cupy arrays
        A_gpu = cp.sparse.csr_matrix(A)
        b_gpu = cp.array(b_flat)
        x0_gpu = cp.array(x0)

        preconditioner = self._get_jacobi_preconditioner(A_gpu)

        # Solve A x = b
        x_gpu, info = cg(
            A=A_gpu,
            b=b_gpu,
            x0=x0_gpu,
            M=preconditioner,
            maxiter=self.max_iters,
            tol=self.stopping_criteria,
        )
        solution_inner_flat = cp.asnumpy(x_gpu)

        if info > 0:
            raise RuntimeError(f"CG did not converge after {info} iterations")
        elif info < 0:
            raise RuntimeError(f"CG error: {info}")

        self._result[inner_slice] = solution_inner_flat.reshape((inner_n_y, inner_n_x))

        return self._result
