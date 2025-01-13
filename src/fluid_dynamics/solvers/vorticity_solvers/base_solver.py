from abc import ABC

import numpy as np
from numpy.typing import NDArray

from src.base_solver import Sweep2DSolver, BaseSolver
from src.convective_operator import ConvectionOperator
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.solvers.vorticity_solvers.bc_mixin import VorticityBCMixin
from src.geometry import DomainGeometry


class ImplicitVorticitySolver(Sweep2DSolver, VorticityBCMixin, ABC):
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        convective_operator: ConvectionOperator,
        bc_order: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            geometry=geometry,
        )

        self.parameters = parameters
        self.convective_operator = convective_operator
        self.bc_order = bc_order

        # Pre-allocate some arrays that will be used in the calculations
        self._temp_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._new_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self.top_bc: NDArray[np.float64] = np.empty(self.geometry.n_x)
        self.right_bc: NDArray[np.float64] = np.empty(self.geometry.n_y)
        self.bottom_bc: NDArray[np.float64] = np.empty(self.geometry.n_x)
        self.left_bc: NDArray[np.float64] = np.empty(self.geometry.n_y)
        self._c_ind: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )


class ExplicitVorticitySolver(BaseSolver, VorticityBCMixin, ABC):
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        convective_operator: ConvectionOperator,
        bc_order: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            geometry=geometry,
        )

        self.parameters = parameters
        self.convective_operator = convective_operator
        self.bc_order = bc_order

        # Pre-allocate some arrays that will be used in the calculations
        self._new_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self.top_bc: NDArray[np.float64] = np.empty(self.geometry.n_x)
        self.right_bc: NDArray[np.float64] = np.empty(self.geometry.n_y)
        self.bottom_bc: NDArray[np.float64] = np.empty(self.geometry.n_x)
        self.left_bc: NDArray[np.float64] = np.empty(self.geometry.n_y)
        self.c_ind: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
