from abc import ABC, abstractmethod
from enum import Enum

from src.parameters.config import ExperimentConfig


class ConvectiveTermForm(Enum):
    DIVERGENT_CENTRAL = "Divergent Central"
    NON_DIVERGENT_CENTRAL = "Non-Divergent Central"
    SYMMETRIC = "Symmetric"
    UPWIND_FC = "Upwind Face-Centered"
    UPWIND_NC = "Upwind Node-Centered"
    DEFERRED_CORRECTION = "Deferred Correction"


class BaseConvectiveOperator(ABC):
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(
        self,
        *args,
        **kwargs,
    ): ...
