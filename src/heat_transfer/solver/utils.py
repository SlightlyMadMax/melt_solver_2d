from src.heat_transfer.solver.registry import (
    HeatTransferSchemeName,
    HeatTransferSchemeRegistry,
)


def register_scheme(scheme: HeatTransferSchemeName):
    def decorator(scheme_class):
        HeatTransferSchemeRegistry.register_scheme(scheme, scheme_class)

        return scheme_class

    return decorator
