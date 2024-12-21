from functools import wraps

from src.fluid_dynamics.schemes.registry import (
    NavierStokesSchemeName,
    NavierStokesSchemeRegistry,
)


def register_scheme(scheme: NavierStokesSchemeName):
    def decorator(scheme_class):
        NavierStokesSchemeRegistry.register_scheme(scheme, scheme_class)

        @wraps(scheme_class)
        class WrappedScheme(scheme_class):
            pass

        return WrappedScheme

    return decorator
