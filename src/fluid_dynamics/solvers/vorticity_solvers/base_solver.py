from abc import ABC

import numpy as np
from numpy.typing import NDArray

from src.convective_operators import BaseConvectiveOperator
from src.core.solvers.base_solver import BaseSolver
from src.core.solvers.mixins.sweep_2d import Sweep2DMixin
from src.fluid_dynamics.utils import VorticityBCMixin
from src.parameters.config import ExperimentConfig


class BaseVorticitySolver(BaseSolver, VorticityBCMixin, ABC):
    def __init__(
        self,
        cfg: ExperimentConfig,
        convective_operator: BaseConvectiveOperator,
        bc_order: int,
        *args,
        **kwargs,
    ):
        super().__init__(cfg=cfg)

        self.convective_operator = convective_operator
        self.bc_order = bc_order

        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x

        # Pre-allocate some arrays that will be used in the calculations
        self._new_w: NDArray[np.float64] = np.empty((n_y, n_x))
        self._conv_x: NDArray[np.float64] = np.empty((n_y, n_x, 3))
        self._conv_y: NDArray[np.float64] = np.empty((n_y, n_x, 3))
        self.top_bc: NDArray[np.float64] = np.empty(n_x)
        self.right_bc: NDArray[np.float64] = np.empty(n_y)
        self.bottom_bc: NDArray[np.float64] = np.empty(n_x)
        self.left_bc: NDArray[np.float64] = np.empty(n_y)
        self.c_ind: NDArray[np.float64] = np.empty((n_y, n_x))


class ImplicitVorticitySolver(BaseVorticitySolver, Sweep2DMixin, ABC):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._initialize_sweep_arrays()

    def _apply_boundary_conditions_x(self, time: float) -> None:
        self.apply_dirichlet(
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            value=self.left_bc,
            side=0,
        )
        self.apply_dirichlet(
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            value=self.right_bc,
            side=1,
        )

    def _apply_boundary_conditions_y(self, time: float) -> None:
        self.apply_dirichlet(
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            value=self.bottom_bc,
            side=0,
        )
        self.apply_dirichlet(
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            value=self.top_bc,
            side=1,
        )


class ExplicitVorticitySolver(BaseVorticitySolver, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
