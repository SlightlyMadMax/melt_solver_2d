import typing
from enum import Enum

from src.heat_transfer.solvers.heat_transfer_solvers.base import BaseHeatTransferSolver


class HeatTransferSolverName(Enum):
    LOC_ONE_DIM = "Locally one dimensional"
    DOUGLAS_RACHFORD = "Douglas-Rachford"
    PEACEMAN_RACHFORD = "Peaceman-Rachford"
    EXPLICIT = "Explicit"


def register_solver(solver_name: HeatTransferSolverName):
    def decorator(solver_class):
        HeatTransferSolverRegistry.register_solver(solver_name, solver_class)

        return solver_class

    return decorator


class HeatTransferSolverRegistry:
    _registry = {}

    @classmethod
    def register_solver(
        cls, solver_name: HeatTransferSolverName, solver_class: typing.Type
    ):
        cls._registry[solver_name] = solver_class

    @classmethod
    def get_solver_class(cls, solver_name: HeatTransferSolverName) -> typing.Type[BaseHeatTransferSolver]:
        try:
            return cls._registry[solver_name]
        except KeyError:
            raise ValueError(f"Heat transfer solver {solver_name} is not registered.")
