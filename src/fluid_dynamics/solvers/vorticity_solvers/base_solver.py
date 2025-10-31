from abc import ABC
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.convective_operators import BaseConvectiveOperator, StreamFunctionBasedConvectiveOperator
from src.core.constants import ABS_ZERO
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
        self.penalty_term: NDArray[np.float64] = np.empty((n_y, n_x))
        self.px_half: NDArray[np.float64] = np.empty((n_y, n_x - 1))
        self.py_half: NDArray[np.float64] = np.empty((n_y - 1, n_x))
        self.buoyancy_term: NDArray[np.float64] = np.empty((n_y, n_x))

    def _calculate_buoyancy_term(self, u: np.ndarray):
        dx_scaled, _, _ = self.cfg.scaled_grid_steps
        inv_re2 = 1.0 / self.cfg.reynolds_number**2
        inv_dx = 1.0 / dx_scaled
        gr = self.cfg.grashof_number
        delta_u = self.cfg.delta_u
        u_ref = self.cfg.u_ref
        beta = self.cfg.thermal_exp_coefficient_ref
        rho_ref = self.cfg.material_props.density_liquid
        interior = (slice(1, -1), slice(1, -1))

        u_k = u * delta_u + u_ref
        u_c = u_k + ABS_ZERO
        drhodu = (
            0.0673268037314653
            - 2 * 0.00894484552601798 * u_c
            + 3 * 8.78462866500416e-5 * u_c**2
            - 4 * 6.62139792627547e-7 * u_c**3
        )

        dudx = 0.5 * inv_dx * (u_k[1:-1, 2:] - u_k[1:-1, :-2])
        drhodx = drhodu[interior] * dudx
        self.buoyancy_term[interior] = gr * inv_re2 * drhodx / (delta_u * beta * rho_ref)

        # dudx = 0.5 * inv_dx * (u[1:-1, 2:] - u[1:-1, :-2])
        #
        # self.buoyancy_term[1:-1, 1:-1] = gr * inv_re2 * dudx

    def _calculate_penalty_term_at_faces(self):
        self.px_half[:, :] = 0.5 * (
            self.penalty_term[:, :-1] + self.penalty_term[:, 1:]
        )
        self.py_half[:, :] = 0.5 * (
            self.penalty_term[:-1, :] + self.penalty_term[1:, :]
        )

    def _calculate_penalty_term_coeff(self, u: np.ndarray, delta: float) -> None:
        u_pt = self.cfg.u_pt_nd
        eps = self.cfg.epsilon
        inv_eps2 = 1.0 / (eps * eps)
        diff_u = u - u_pt

        # --- Variant 1: sharp step ----------------------
        # self.penalty_term[:, :] = np.where(u <= u_pt, inv_eps2, 0.0)

        # --- Variant 2: error‐function form -------------------
        # f_l = 0.5 * (1.0 + erf(diff_u / (np.sqrt(2.0) * delta)))
        # self.penalty_term[:, :] = inv_eps2 * (1.0 - f_l) ** 2 / (f_l**3 + 1e-6)
        # self.penalty_term[:, :] = 0.5 * inv_eps2 * (1.0 - erf(diff_u / (np.sqrt(2.0) * delta)))

        # --- Variant 3: hyperbolic‐tangent form ---------------
        self.penalty_term[:, :] = 0.5 * inv_eps2 * (1.0 - np.tanh(diff_u / delta))

        # --- Variant 4: exponential form (one-sided smoothing) ----------------------
        # exp_term = np.exp((delta - diff_u) / delta)
        # temp = inv_eps2 * 0.5 * (2.0 + exp_term / (0.5 - exp_term))
        # self.penalty_term[:, :] = np.where(diff_u <= 0, temp, 0.0)

    def _prepare(
        self,
        sf: np.ndarray,
        u: np.ndarray,
        conv_w: Optional[np.ndarray] = None,
        delta: Optional[float] = None,
    ):
        dx_scaled, dy_scaled, _ = self.cfg.scaled_grid_steps

        if isinstance(self.convective_operator, StreamFunctionBasedConvectiveOperator):
            self.convective_operator(conv_x=self._conv_x, conv_y=self._conv_y, sf=sf)
        else:
            assert conv_w is not None
            self.convective_operator(conv_x=self._conv_x, conv_y=self._conv_y, w=conv_w)

        self._calculate_penalty_term_coeff(u=u, delta=delta or self.cfg.delta_nd)

        self._calculate_penalty_term_at_faces()

        self._calculate_buoyancy_term(u=u)

        self.calculate_boundary_conditions(
            sf=sf,
            top_bc=self.top_bc,
            right_bc=self.right_bc,
            bottom_bc=self.bottom_bc,
            left_bc=self.left_bc,
            order=self.bc_order,
            dx=dx_scaled,
            dy=dy_scaled,
        )


class ADIVorticitySolver(BaseVorticitySolver, Sweep2DMixin, ABC):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._initialize_sweep_arrays()

    def solve(
        self,
        w: np.ndarray,
        sf: np.ndarray,
        u: np.ndarray,
        conv_w: Optional[np.ndarray] = None,
        delta: Optional[float] = None,
        time: float = 0.0,
    ) -> np.ndarray:
        self._prepare(sf=sf, u=u, conv_w=conv_w, delta=delta)

        n_x, n_y = self.cfg.geometry.n_x, self.cfg.geometry.n_y
        dx_scaled, dy_scaled, dt_scaled = self.cfg.scaled_grid_steps

        self._compute_sweep_x_coeffs(
            w=w, sf=sf, dx=dx_scaled, dy=dy_scaled, dt=dt_scaled
        )

        self._apply_boundary_conditions_x(time=time)

        self._new_w[:, :] = w

        self._solve_sweep_x(
            n=n_y,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            result=self._new_w,
        )

        self._compute_sweep_y_coeffs(
            w=w, sf=sf, dx=dx_scaled, dy=dy_scaled, dt=dt_scaled
        )

        self._apply_boundary_conditions_y(time=time)

        self._solve_sweep_y(
            n=n_x,
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            result=self._new_w,
        )

        return self._new_w

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
