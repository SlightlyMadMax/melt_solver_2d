from abc import ABC

import numpy as np
from numpy.typing import NDArray

from src.core.geometry import DomainGeometry
from src.core.solvers.base_solver import BaseSolver
from src.convective_operators import BaseConvectiveOperator
from src.core.solvers.mixins.sweep_2d import Sweep2DMixin
from src.fluid_dynamics.utils import VorticityBCMixin
from src.parameters.fluid import FluidParameters


class BaseVorticitySolver(BaseSolver, VorticityBCMixin, ABC):
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        convective_operator: BaseConvectiveOperator,
        bc_order: int,
        *args,
        **kwargs,
    ):
        super().__init__(geometry=geometry)

        self.parameters = parameters
        self.convective_operator = convective_operator
        self.bc_order = bc_order

        # Pre-allocate some arrays that will be used in the calculations
        self._new_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._conv_x: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x, 3)
        )
        self._conv_y: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x, 3)
        )
        self.top_bc: NDArray[np.float64] = np.empty(self.geometry.n_x)
        self.right_bc: NDArray[np.float64] = np.empty(self.geometry.n_y)
        self.bottom_bc: NDArray[np.float64] = np.empty(self.geometry.n_x)
        self.left_bc: NDArray[np.float64] = np.empty(self.geometry.n_y)
        self.c_ind: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self.rho = self.calculate_rho()

    def calculate_rho(self):
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        dy, dx = self.geometry.dy, self.geometry.dx

        rho = np.zeros((n_y, n_x))

        rho[2 : n_y - 2, 1] = 2 * dx**-4
        rho[2 : n_y - 2, n_x - 2] = 2 * dx**-4

        rho[1, 2 : n_x - 2] = 2 * dy**-4
        rho[n_y - 2, 2 : n_x - 2] = 2 * dy**-4

        rho[1, 1] = 2 * (dx**-4 + dy**-4)
        rho[1, n_x - 2] = 2 * (dx**-4 + dy**-4)
        rho[n_y - 2, 1] = 2 * (dx**-4 + dy**-4)
        rho[n_y - 2, n_x - 2] = 2 * (dx**-4 + dy**-4)

        return rho


class ImplicitVorticitySolver(BaseVorticitySolver, Sweep2DMixin, ABC):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._initialize_sweep_arrays()

        # Pre-allocate some arrays that will be used in the calculations
        self._temp_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )


class ExplicitVorticitySolver(BaseVorticitySolver, ABC):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
