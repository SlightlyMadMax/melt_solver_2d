from functools import wraps

from src.fluid_dynamics.solver.registry import (
    NavierStokesSchemeName,
    NavierStokesSchemeRegistry,
)


def register_scheme(scheme: NavierStokesSchemeName):
    def decorator(scheme_class):
        NavierStokesSchemeRegistry.register_scheme(scheme, scheme_class)

        return scheme_class

    return decorator
