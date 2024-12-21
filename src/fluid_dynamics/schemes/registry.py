import typing
from enum import Enum


class NavierStokesScheme(Enum):
    EXPLICIT_CENTRAL = 1, "Explicit central differences"
    EXPLICIT_UPWIND = 2, "Explicit upwind"
    DOUGLAS_RACHFORD = 3, "Douglas-Rachford"
    PEACEMAN_RACHFORD = 4, "Peaceman-Rachford"


class NavierStokesSchemeRegistry:
    _registry = {}

    @classmethod
    def register_scheme(cls, scheme: NavierStokesScheme, scheme_class: typing.Type):
        cls._registry[scheme] = scheme_class

    @classmethod
    def get_scheme_class(cls, scheme: NavierStokesScheme) -> typing.Type:
        try:
            return cls._registry[scheme]
        except KeyError:
            raise ValueError(f"Scheme {scheme} not registered.")
