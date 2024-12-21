import typing
from enum import Enum


class NavierStokesSchemeName(Enum):
    EXPLICIT_CENTRAL = "Explicit central differences"
    EXPLICIT_UPWIND = "Explicit upwind"
    DOUGLAS_RACHFORD = "Douglas-Rachford"
    PEACEMAN_RACHFORD = "Peaceman-Rachford"


class NavierStokesSchemeRegistry:
    _registry = {}

    @classmethod
    def register_scheme(cls, scheme: NavierStokesSchemeName, scheme_class: typing.Type):
        cls._registry[scheme] = scheme_class

    @classmethod
    def get_scheme_class(cls, scheme: NavierStokesSchemeName) -> typing.Type:
        try:
            return cls._registry[scheme.value]
        except KeyError:
            raise ValueError(f"Scheme {scheme} not registered.")
