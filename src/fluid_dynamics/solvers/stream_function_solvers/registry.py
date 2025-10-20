import typing
from enum import Enum

from src.core.solvers.base_solver import BaseSolver


class StreamFunctionSolverName(Enum):
    SOR = "Successive Over-Relaxation"
    MATRIX_SWEEP = "Matrix Sweep (2D Thomas algorithm)"
    CG = "Conjugate Gradient"
    CG_GPU = "Conjugate Gradient with GPU Acceleration"
    AMG = "Algebraic Multigrid"


def register_sf_solver(solver_name: StreamFunctionSolverName):
    def decorator(solver_class):
        StreamFunctionSolverRegistry.register_solver(solver_name, solver_class)

        return solver_class

    return decorator


class StreamFunctionSolverRegistry:
    _registry = {}

    @classmethod
    def register_solver(
        cls, solver_name: StreamFunctionSolverName, solver_class: typing.Type
    ):
        cls._registry[solver_name] = solver_class

    @classmethod
    def get_solver_class(cls, solver_name: StreamFunctionSolverName):
        try:
            return cls._registry[solver_name]
        except KeyError:
            raise ValueError(
                f"Stream function equation solver {solver_name} is not registered."
            )
