import typing
from enum import Enum


class HeatTransferSchemeName(Enum):
    LOC_ONE_DIM = "Locally one dimensional"
    DOUGLAS_RACHFORD = "Douglas-Rachford"
    PEACEMAN_RACHFORD = "Peaceman-Rachford"


class HeatTransferSchemeRegistry:
    _registry = {}

    @classmethod
    def register_scheme(cls, scheme: HeatTransferSchemeName, scheme_class: typing.Type):
        cls._registry[scheme] = scheme_class

    @classmethod
    def get_scheme_class(cls, scheme: HeatTransferSchemeName) -> typing.Type:
        try:
            return cls._registry[scheme]
        except KeyError:
            raise ValueError(f"Scheme {scheme} not registered.")
