import numpy as np

from abc import abstractmethod, ABC
from typing import Optional, Callable

from numba import njit
from numpy.typing import NDArray

from src.convective_operators import BaseConvectiveOperator
from src.core.boundary_conditions import (
    BoundaryConditions,
    BoundaryCondition,
    BoundaryConditionType,
)
from src.core.solvers.base_solver import BaseSolver
from src.core.solvers.mixins.iterative_solver import IterativeSolverMixin
from src.core.solvers.mixins.sweep_2d import Sweep2DMixin
from src.heat_transfer.coefficient_smoothing.coefficients import (
    StepScheme,
    DeltaScheme,
    get_step_fn,
    get_delta_fn,
)
from src.parameters.config import ExperimentConfig


class BaseHeatSolver(IterativeSolverMixin, BaseSolver):
    def __init__(
        self,
        cfg: ExperimentConfig,
        convective_operator: BaseConvectiveOperator,
        bcs: Optional[BoundaryConditions] = None,
        fixed_delta: bool = False,
        max_iters: int = 5,
        tolerance: float = 1e-6,
        urf: float = 0.5,
        bc_order: int = 1,
        step_scheme: StepScheme = StepScheme.ERF,
        delta_scheme: DeltaScheme = DeltaScheme.GAUSS,
        *args,
        **kwargs,
    ):
        super().__init__(cfg=cfg, bcs=bcs)

        self.convective_operator = convective_operator
        self.fixed_delta = fixed_delta
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.urf = urf
        self.bc_order = bc_order
        self.step_scheme = step_scheme
        self.delta_scheme = delta_scheme
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x

        # Pre-allocate some arrays that will be used in the calculations
        self._iter_u: NDArray[np.float64] = np.empty((n_y, n_x))
        self._new_u: NDArray[np.float64] = np.empty((n_y, n_x))
        self._conv_x: NDArray[np.float64] = np.empty((n_y, n_x, 3))
        self._conv_y: NDArray[np.float64] = np.empty((n_y, n_x, 3))
        self._c_eff = np.empty((n_y, n_x))
        self._k_eff = np.empty((n_y, n_x))

    def _prepare(self, sf: np.ndarray, delta: tuple[float, float] | None = None):
        self.convective_operator(
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            sf=sf,
        )
        self.compute_effective_properties(
            c_eff=self._c_eff, k_eff=self._k_eff, u=self._iter_u, delta=delta
        )

    @abstractmethod
    def solve_linear(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        delta: tuple[float, float] | None = None,
        time: float = 0.0,
    ) -> None: ...

    def compute_k_eff(
        self, u: float, delta: tuple[float, float] | None = None
    ) -> float:
        props = self.cfg.material_props
        k_ref = self.cfg.thermal_conductivity_ref
        k_solid_nd = props.thermal_conductivity_solid / k_ref
        k_liquid_nd = props.thermal_conductivity_liquid / k_ref

        if delta is None:
            delta = self.cfg.delta_nd
        else:
            delta = max(delta[0], delta[1])

        u_0 = self.cfg.u_pt_nd
        if delta <= 0:
            return k_solid_nd if u <= u_0 else k_liquid_nd

        step_fn = get_step_fn(self.step_scheme)
        step_val = step_fn(u, u_0, delta)
        return k_solid_nd + (k_liquid_nd - k_solid_nd) * step_val

    def compute_effective_properties(
        self,
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        u: NDArray[np.float64],
        delta: Optional[tuple[float, float]] = None,
    ) -> None:
        step_fn = get_step_fn(self.step_scheme)
        delta_fn = get_delta_fn(self.delta_scheme)
        props = self.cfg.material_props
        c_ref = self.cfg.volumetric_heat_capacity_ref
        k_ref = self.cfg.thermal_conductivity_ref

        c_solid_nd = props.volumetric_heat_capacity_solid / c_ref
        c_liquid_nd = props.volumetric_heat_capacity_liquid / c_ref
        latent_heat_nd = 1.0 / self.cfg.stefan_number
        k_solid_nd = props.thermal_conductivity_solid / k_ref
        k_liquid_nd = props.thermal_conductivity_liquid / k_ref

        delta = (
            (self.cfg.delta_left_nd, self.cfg.delta_right_nd)
            if delta is None
            else delta
        )

        is_asymmetric = self.delta_scheme == DeltaScheme.GAUSS_ASYM

        if is_asymmetric:
            self._compute_effective_properties_asym(
                c_eff=c_eff,
                k_eff=k_eff,
                u=u,
                u_0=self.cfg.u_pt_nd,
                c_solid=c_solid_nd,
                c_liquid=c_liquid_nd,
                l_solid=latent_heat_nd,
                k_solid=k_solid_nd,
                k_liquid=k_liquid_nd,
                delta_left=delta[0],
                delta_right=delta[1],
                step_fn=step_fn,
                delta_fn=delta_fn,
            )
        else:
            self._compute_effective_properties_sym(
                c_eff=c_eff,
                k_eff=k_eff,
                u=u,
                u_0=self.cfg.u_pt_nd,
                c_solid=c_solid_nd,
                c_liquid=c_liquid_nd,
                l_solid=latent_heat_nd,
                k_solid=k_solid_nd,
                k_liquid=k_liquid_nd,
                delta_max=max(delta[0], delta[1]),
                step_fn=step_fn,
                delta_fn=delta_fn,
            )

    @staticmethod
    @njit
    def _compute_effective_properties_asym(
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        u: NDArray[np.float64],
        u_0: float,
        c_solid: float,
        c_liquid: float,
        l_solid: float,
        k_solid: float,
        k_liquid: float,
        delta_left: float,
        delta_right: float,
        step_fn: Callable,
        delta_fn: Callable,
    ) -> None:
        n_y, n_x = u.shape
        c_diff = c_liquid - c_solid
        k_diff = k_liquid - k_solid
        delta_max = max(delta_left, delta_right)

        for j in range(n_y):
            for i in range(n_x):
                if delta_left <= 0 and delta_right <= 0:
                    c_eff[j, i] = c_solid if u[j, i] <= u_0 else c_liquid
                    k_eff[j, i] = k_solid if u[j, i] <= u_0 else k_liquid
                    continue

                step_val = step_fn(u[j, i], u_0, delta_max)
                delta_val = delta_fn(u[j, i], u_0, delta_left, delta_right)

                c_eff[j, i] = c_solid + c_diff * step_val + l_solid * delta_val
                k_eff[j, i] = k_solid + k_diff * step_val

    @staticmethod
    @njit
    def _compute_effective_properties_sym(
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        u: NDArray[np.float64],
        u_0: float,
        c_solid: float,
        c_liquid: float,
        l_solid: float,
        k_solid: float,
        k_liquid: float,
        delta_max: float,
        step_fn: Callable,
        delta_fn: Callable,
    ) -> None:
        n_y, n_x = u.shape
        c_diff = c_liquid - c_solid
        k_diff = k_liquid - k_solid

        for j in range(n_y):
            for i in range(n_x):
                if delta_max <= 0:
                    c_eff[j, i] = c_solid if u[j, i] <= u_0 else c_liquid
                    k_eff[j, i] = k_solid if u[j, i] <= u_0 else k_liquid
                    continue

                step_val = step_fn(u[j, i], u_0, delta_max)
                delta_val = delta_fn(u[j, i], u_0, delta_max)

                c_eff[j, i] = c_solid + c_diff * step_val + l_solid * delta_val
                k_eff[j, i] = k_solid + k_diff * step_val


class ADIHeatSolver(BaseHeatSolver, Sweep2DMixin, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_sweep_arrays()

    def solve_linear(
        self,
        u: np.ndarray,
        sf: Optional[np.ndarray] = None,
        delta: tuple[float, float] | None = None,
        time: float = 0.0,
    ) -> None:
        """
        Advance the solution by one ADI time step.

        Performs an x-direction sweep followed by a y-direction sweep using
        the coefficients provided by `_compute_sweep_x_coeffs` and
        `_compute_sweep_y_coeffs`. Boundary conditions are applied between
        the sweeps.

        :param u: solution array at the beginning of the step (u^n).
        :param sf: stream function
        :param delta: phase change smoothing range
        :param time: current physical time.
        :return:
        """
        self._prepare(sf, delta)
        n_x, n_y = self.cfg.geometry.n_x, self.cfg.geometry.n_y
        dx_scaled, dy_scaled, dt_scaled = self.cfg.scaled_grid_steps()

        self._compute_sweep_x_coeffs(state=u, dx=dx_scaled, dy=dy_scaled, dt=dt_scaled)

        self._new_u = np.copy(u)

        self._apply_boundary_conditions_x(time=time)

        self._solve_sweep_x(
            n=n_y,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            result=self._new_u,
        )

        self._compute_sweep_y_coeffs(state=u, dx=dx_scaled, dy=dy_scaled, dt=dt_scaled)

        self._apply_boundary_conditions_y(time=time)

        self._solve_sweep_y(
            n=n_x,
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            result=self._new_u,
        )

    def _apply_boundary_conditions_x(self, time: float) -> None:
        self._apply_standard_bc(
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            bc=self.bcs.left,
            side=0,
            time=time,
            k_eff_slice=self._k_eff[:, 0],
        )
        self._apply_standard_bc(
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            bc=self.bcs.right,
            side=1,
            time=time,
            k_eff_slice=self._k_eff[:, -1],
        )

    def _apply_boundary_conditions_y(self, time: float) -> None:
        self._apply_standard_bc(
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            bc=self.bcs.bottom,
            side=0,
            time=time,
            k_eff_slice=self._k_eff[0, :],
        )
        self._apply_standard_bc(
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            bc=self.bcs.top,
            side=1,
            time=time,
            k_eff_slice=self._k_eff[-1, :],
        )

    def _apply_standard_bc(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        rhs: np.ndarray,
        bc: BoundaryCondition,
        side: int,
        time: float,
        k_eff_slice: np.ndarray,
    ) -> bool:
        """
        Common Dirichlet / first-order Neumann.
        a,b,c,rhs are the full 2D coefficient arrays for one sweep.
        bc is one of self.bcs.left/right or bottom/top.
        side is 0 or 1.
        k_eff_slice is the 1D array of k_eff on that boundary.
        """
        if bc.boundary_type == BoundaryConditionType.DIRICHLET:
            self.apply_dirichlet(
                a=a, b=b, c=c, rhs=rhs, value=bc.get_value(t=time), side=side
            )
        elif bc.boundary_type == BoundaryConditionType.NEUMANN:
            if self.bc_order == 1:
                # first-order ghost
                flux = bc.get_flux(t=time) / (
                    k_eff_slice * self.cfg.thermal_conductivity_ref
                )
                self.apply_neumann_first_order(
                    a=a, b=b, c=c, rhs=rhs, flux=flux, side=side
                )
            else:
                # signal to caller: we need second-order BC here
                return False
        else:
            raise NotImplementedError("BC type not supported")

        return True
