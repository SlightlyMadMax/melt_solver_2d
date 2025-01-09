import typing
from enum import Enum


class VorticitySolverName(Enum):
    EXPLICIT = "Explicit"
    DOUGLAS_RACHFORD = "Douglas-Rachford"
    PEACEMAN_RACHFORD = "Peaceman-Rachford"
    LOC_ONE_DIM = "Local one dimensional"


def register_solver(solver_name: VorticitySolverName):
    def decorator(solver_class):
        VorticitySolverRegistry.register_solver(solver_name, solver_class)

        return solver_class

    return decorator


class VorticitySolverRegistry:
    _registry = {}

    @classmethod
    def register_solver(
        cls, solver_name: VorticitySolverName, solver_class: typing.Type
    ):
        cls._registry[solver_name] = solver_class

    @classmethod
    def get_solver_class(cls, solver_name: VorticitySolverName) -> typing.Type:
        try:
            return cls._registry[solver_name]
        except KeyError:
            raise ValueError(
                f"Vorticity equation solver {solver_name} is not registered."
            )
