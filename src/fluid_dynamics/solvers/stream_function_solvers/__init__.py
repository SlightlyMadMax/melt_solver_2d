from .sor import SORPoissonSolver
from .matrix_sweep import MatrixSweepPoissonSolver
from .cg import ConjugateGradientSolver
# from .cg_gpu import ConjugateGradientGPUSolver
from .amg import AlgebraicMultigridSolver
from .registry import StreamFunctionSolverName, StreamFunctionSolverRegistry
