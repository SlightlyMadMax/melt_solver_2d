from functools import wraps

from src.heat_transfer.schemes.registry import (
    HeatTransferSchemeName,
    HeatTransferSchemeRegistry,
)


def register_scheme(scheme: HeatTransferSchemeName):
    def decorator(scheme_class):
        HeatTransferSchemeRegistry.register_scheme(scheme, scheme_class)

        @wraps(scheme_class)
        class WrappedScheme(scheme_class):
            pass

        return WrappedScheme

    return decorator
