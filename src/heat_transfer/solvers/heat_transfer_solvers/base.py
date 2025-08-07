import numpy as np

from abc import abstractmethod
from typing import Optional
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
    c_smoothed,
    k_smoothed,
    StepScheme,
    DeltaScheme,
    get_step_fn,
    get_delta_fn,
)
from src.parameters.config import ExperimentConfig


class BaseHeatTransferSolver(IterativeSolverMixin, BaseSolver):
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

    @abstractmethod
    def solve_linear(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        delta: float,
        time: float = 0.0,
    ) -> None: ...

    def compute_k_eff(self, u: float, delta: float) -> float:
        step_fn = get_step_fn(self.step_scheme)
        k_eff = (
            k_smoothed(
                u=u,
                u_pt=self.cfg.u_pt_non_dim,
                k_solid=self.cfg.material_props.thermal_conductivity_solid,
                k_liquid=self.cfg.material_props.thermal_conductivity_liquid,
                delta=delta,
                step_fn=step_fn,
            )
            / self.cfg.thermal_conductivity_ref
        )
        return k_eff

    def compute_effective_properties(
        self,
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        u: NDArray[np.float64],
        delta: Optional[float] = None,
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

        delta = self.cfg.delta_nd if delta is None else delta
        u_0 = self.cfg.u_mid_nd if delta is None else self.cfg.u_pt_non_dim

        self._compute_effective_properties(
            c_eff=c_eff,
            k_eff=k_eff,
            u=u,
            u_0=u_0,
            c_solid=c_solid_nd,
            c_liquid=c_liquid_nd,
            l_solid=latent_heat_nd,
            k_solid=k_solid_nd,
            k_liquid=k_liquid_nd,
            delta=delta,
            step_fn=step_fn,
            delta_fn=delta_fn,
        )

    @staticmethod
    @njit
    def _compute_effective_properties(
        c_eff: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        u: NDArray[np.float64],
        u_0: float,
        c_solid: float,
        c_liquid: float,
        l_solid: float,
        k_solid: float,
        k_liquid: float,
        delta: float,
        step_fn: callable,
        delta_fn: callable,
    ) -> None:
        n_y, n_x = u.shape

        for j in range(n_y):
            for i in range(n_x):
                c_eff[j, i] = c_smoothed(
                    u=u[j, i],
                    u_pt=u_0,
                    c_solid=c_solid,
                    c_liquid=c_liquid,
                    l_solid=l_solid,
                    delta=delta,
                    delta_fn=delta_fn,
                    step_fn=step_fn,
                )

                k_eff[j, i] = k_smoothed(
                    u=u[j, i],
                    u_pt=u_0,
                    k_solid=k_solid,
                    k_liquid=k_liquid,
                    delta=delta,
                    step_fn=step_fn,
                )


class ImplicitHeatTransferSolver(BaseHeatTransferSolver, Sweep2DMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._initialize_sweep_arrays()

    @abstractmethod
    def solve_linear(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        delta: float,
        time: float = 0.0,
    ) -> None: ...

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


class ExplicitHeatTransferSolver(BaseHeatTransferSolver):
    @abstractmethod
    def solve_linear(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        delta: float,
        time: float = 0.0,
    ) -> None: ...
