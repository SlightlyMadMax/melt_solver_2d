from enum import IntEnum

import numpy as np

from abc import ABC
from typing import Optional, Callable

from numba import njit
from numpy.typing import NDArray

from src.convective_operators import BaseConvectiveOperator, ConvectiveTermForm
from src.core.boundary_conditions import (
    BoundaryConditions,
    BoundaryCondition,
    BoundaryConditionType,
)
from src.core.solvers.base_solver import BaseSolver
from src.core.solvers.mixins.sweep_2d import Sweep2DMixin
from src.heat_transfer.coefficient_smoothing.coefficients import (
    StepScheme,
    DeltaScheme,
    get_step_fn,
    get_delta_fn,
)
from src.parameters.config import ExperimentConfig


class KFaceMethod(IntEnum):
    ARITHMETIC = 0
    HARMONIC = 1
    FROM_TEMP = 2


class BaseHeatSolver(BaseSolver, ABC):
    def __init__(
        self,
        cfg: ExperimentConfig,
        convective_operator: BaseConvectiveOperator,
        max_iters: int,
        tolerance: float,
        urf: float,
        bc_order: int,
        step_scheme: StepScheme,
        delta_scheme: DeltaScheme,
        k_face_method: KFaceMethod,
        post_correction: bool,
        bcs: Optional[BoundaryConditions] = None,
        *args,
        **kwargs,
    ):
        super().__init__(cfg=cfg, bcs=bcs)

        self.convective_operator = convective_operator
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.urf = urf
        self.bc_order = bc_order
        self.step_scheme = step_scheme
        self.delta_scheme = delta_scheme
        self.k_face_method = k_face_method
        self.post_correction = post_correction
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x

        # Pre-allocate some arrays that will be used in the calculations
        self._u_new: NDArray[np.float64] = np.empty((n_y, n_x))
        self._conv_x: NDArray[np.float64] = np.zeros((n_y, n_x, 3))
        self._conv_y: NDArray[np.float64] = np.zeros((n_y, n_x, 3))
        self._correction_x: NDArray[np.float64] = np.zeros((n_y, n_x))
        self._correction_y: NDArray[np.float64] = np.zeros((n_y, n_x))
        self._c_eff = np.empty((n_y, n_x))
        self._k_eff = np.empty((n_y, n_x))

        # Effective conductivity at faces
        self._k_x = np.empty((n_y, n_x + 1))  # i = -1/2 ... n_x-1/2
        self._k_y = np.empty((n_y + 1, n_x))  # j = -1/2 ... n_y-1/2

        self._h = np.empty((n_y, n_x))

    def compute_effective_properties(
        self,
        u: NDArray[np.float64],
        delta: float,
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

        self._compute_effective_properties(
            c_eff=self._c_eff,
            k_eff=self._k_eff,
            u=u,
            u_0=self.cfg.u_pt_nd,
            c_solid=c_solid_nd,
            c_liquid=c_liquid_nd,
            latent_heat=latent_heat_nd,
            k_solid=k_solid_nd,
            k_liquid=k_liquid_nd,
            delta=delta,
            step_fn=step_fn,
            delta_fn=delta_fn,
        )

        self._compute_face_conductivities(
            u=u,
            k_eff=self._k_eff,
            k_x=self._k_x,
            k_y=self._k_y,
            u_0=self.cfg.u_pt_nd,
            k_solid=k_solid_nd,
            k_liquid=k_liquid_nd,
            delta=delta,
            step_fn=step_fn,
            method=self.k_face_method.value,
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
        latent_heat: float,
        k_solid: float,
        k_liquid: float,
        delta: float,
        step_fn: Callable,
        delta_fn: Callable,
    ) -> None:
        n_y, n_x = u.shape
        c_diff = c_liquid - c_solid
        k_diff = k_liquid - k_solid

        if delta > 0:
            for j in range(n_y):
                for i in range(n_x):
                    step_val = step_fn(u[j, i], u_0, delta)
                    delta_val = delta_fn(u[j, i], u_0, delta)

                    c_eff[j, i] = c_solid + c_diff * step_val + latent_heat * delta_val
                    k_eff[j, i] = k_solid + k_diff * step_val
        else:
            for j in range(n_y):
                for i in range(n_x):
                    c_eff[j, i] = c_solid if u[j, i] <= u_0 else c_liquid
                    k_eff[j, i] = k_solid if u[j, i] <= u_0 else k_liquid

    @staticmethod
    @njit
    def _compute_face_conductivities(
        u: NDArray[np.float64],
        k_eff: NDArray[np.float64],
        k_x: NDArray[np.float64],
        k_y: NDArray[np.float64],
        u_0: float,
        k_solid: float,
        k_liquid: float,
        delta: float,
        step_fn: Callable,
        method: int,
    ) -> None:
        n_y, n_x = k_eff.shape

        for j in range(n_y):
            # left boundary: i = -1/2
            k_x[j, 0] = k_eff[j, 0]

            for i in range(1, n_x):
                if method == 0:  # arithmetic
                    k_x[j, i] = 0.5 * (k_eff[j, i - 1] + k_eff[j, i])

                elif method == 1:  # harmonic
                    a = k_eff[j, i - 1]
                    b = k_eff[j, i]
                    s = a + b
                    if s == 0.0:
                        k_x[j, i] = 0.0
                    else:
                        k_x[j, i] = 2.0 * a * b / s

                else:  # from temperature
                    u_face = 0.5 * (u[j, i - 1] + u[j, i])
                    if delta <= 0:
                        k_x[j, i] = k_solid if u_face <= u_0 else k_liquid
                    else:
                        s = step_fn(u_face, u_0, delta)
                        k_x[j, i] = k_solid + (k_liquid - k_solid) * s

            # right boundary: i = n_x - 1/2
            k_x[j, n_x] = k_eff[j, n_x - 1]

        for i in range(n_x):
            # bottom boundary: j = -1/2
            k_y[0, i] = k_eff[0, i]

            for j in range(1, n_y):
                if method == 0:
                    k_y[j, i] = 0.5 * (k_eff[j - 1, i] + k_eff[j, i])

                elif method == 1:
                    a = k_eff[j - 1, i]
                    b = k_eff[j, i]
                    s = a + b
                    if s == 0.0:
                        k_y[j, i] = 0.0
                    else:
                        k_y[j, i] = 2.0 * a * b / s

                else:
                    u_face = 0.5 * (u[j - 1, i] + u[j, i])
                    if delta <= 0:
                        k_y[j, i] = k_solid if u_face <= u_0 else k_liquid
                    else:
                        s = step_fn(u_face, u_0, delta)
                        k_y[j, i] = k_solid + (k_liquid - k_solid) * s

            # top boundary: j = n_y - 1/2
            k_y[n_y, i] = k_eff[n_y - 1, i]

    @staticmethod
    @njit
    def _enthalpy_from_temperature(
        h_out: NDArray[np.float64],
        u: NDArray[np.float64],
        u_0: float,
        c_solid: float,
        c_liquid: float,
        latent_heat: float,
        delta: float,
    ) -> None:
        n_y, n_x = u.shape
        if delta > 0.0:
            u_s = u_0 - delta
            u_l = u_0 + delta
            c_diff = c_liquid - c_solid

            # h at Tl: evaluate integral through mushy zone
            # Hl = c_solid*Tl + c_diff*delta + latent_heat
            h_l = c_solid * u_l + c_diff * delta + latent_heat

            for j in range(n_y):
                for i in range(n_x):
                    uu = u[j, i]
                    if uu <= u_s:
                        h_out[j, i] = c_solid * uu
                    elif uu >= u_l:
                        h_out[j, i] = h_l + c_liquid * (uu - u_l)
                    else:
                        # mushy branch:
                        # cp_app(T) = c_solid + (c_diff/(2*delta))*(T - Ts) + latent_heat/(2*delta)
                        # h = c_solid*T + (c_diff/(4*delta))*(T - Ts)^2 + (latent_heat/(2*delta))*(T - Ts)
                        y = uu - u_s
                        h_out[j, i] = (
                            c_solid * uu
                            + (c_diff / (4.0 * delta)) * (y * y)
                            + (latent_heat / (2.0 * delta)) * y
                        )
        else:
            # sharp jump at u_0
            for j in range(n_y):
                for i in range(n_x):
                    uu = u[j, i]
                    if uu <= u_0:
                        h_out[j, i] = c_solid * uu
                    else:
                        h_out[j, i] = c_liquid * uu + latent_heat

    @staticmethod
    @njit
    def _temperature_from_enthalpy(
        u_out: NDArray[np.float64],
        h: NDArray[np.float64],
        u_0: float,
        c_solid: float,
        c_liquid: float,
        latent_heat: float,
        delta: float,
    ) -> None:
        n_y, n_x = h.shape
        if delta > 0.0:
            u_s = u_0 - delta
            u_l = u_0 + delta
            c_diff = c_liquid - c_solid

            h_s = c_solid * u_s
            h_l = c_solid * u_l + c_diff * delta + latent_heat

            # coefficients for quadratic in mushy zone:
            # h - Hs = a*y + b*y^2  with y = T - Ts
            # a = c_solid + latent_heat/(2*delta)
            # b = c_diff/(4*delta)
            a = c_solid + (latent_heat / (2.0 * delta))
            b = c_diff / (4.0 * delta)

            for j in range(n_y):
                for i in range(n_x):
                    hh = h[j, i]
                    if hh <= h_s:
                        # solid branch: T = h / c_solid
                        u_out[j, i] = hh / c_solid
                    elif hh >= h_l:
                        # liquid branch: T = Tl + (h - Hl)/c_liquid
                        u_out[j, i] = u_l + (hh - h_l) / c_liquid
                    else:
                        # mushy: solve b*y^2 + A*y - (h - Hs) = 0 for y >= 0
                        rhs = hh - h_s
                        if b == 0.0:
                            # degenerates to linear
                            y = rhs / a
                        else:
                            disc = a * a + 4.0 * b * rhs
                            if disc < 0.0:
                                disc = 0.0
                            sqrt_disc = np.sqrt(disc)
                            y = (-a + sqrt_disc) / (2.0 * b)
                            # numerical safety
                            if y < 0.0:
                                y = 0.0
                            if y > 2.0 * delta:
                                y = 2.0 * delta
                        u_out[j, i] = u_s + y
        else:
            h_s = c_solid * u_0
            for j in range(n_y):
                for i in range(n_x):
                    hh = h[j, i]
                    if hh <= h_s:
                        u_out[j, i] = hh / c_solid
                    else:
                        u_out[j, i] = (hh - latent_heat) / c_liquid

    @staticmethod
    @njit
    def _update_enthalpy(
        h_new: NDArray[np.float64],
        h_old: NDArray[np.float64],
        c_eff: NDArray[np.float64],
        u_star: NDArray[np.float64],
        u_old: NDArray[np.float64],
    ) -> None:
        """
        Vectorized elementwise update:
        h_new = h_old + c_eff * (u_star - u_old)
        """
        n_y, n_x = h_old.shape
        for j in range(n_y):
            for i in range(n_x):
                h_new[j, i] = h_old[j, i] + c_eff[j, i] * (u_star[j, i] - u_old[j, i])

    def posterior_correction(
        self,
        u_old: NDArray[np.float64],
        u_star: NDArray[np.float64],
        delta: float,
        time: float,
    ) -> None:
        props = self.cfg.material_props
        c_ref = self.cfg.volumetric_heat_capacity_ref
        c_solid_nd = props.volumetric_heat_capacity_solid / c_ref
        c_liquid_nd = props.volumetric_heat_capacity_liquid / c_ref
        latent_heat_nd = 1.0 / self.cfg.stefan_number
        u0 = self.cfg.u_pt_nd

        # compute h_old = h(u_old)
        self._enthalpy_from_temperature(
            h_out=self._h,
            u=u_old,
            u_0=u0,
            c_solid=c_solid_nd,
            c_liquid=c_liquid_nd,
            latent_heat=latent_heat_nd,
            delta=delta,
        )

        # compute h_new = h_old + c_eff_old * (u_star - u_old)
        self._update_enthalpy(
            h_new=self._h,
            h_old=self._h,
            c_eff=self._c_eff,
            u_star=u_star,
            u_old=u_old,
        )

        # invert h_new -> u_new
        self._temperature_from_enthalpy(
            u_out=self._u_new,
            h=self._h,
            u_0=u0,
            c_solid=c_solid_nd,
            c_liquid=c_liquid_nd,
            latent_heat=latent_heat_nd,
            delta=delta,
        )

        # enforce Dirichlet BCs (keep exact prescribed BCs after projection)
        # left/right
        if self.bcs.left.boundary_type == BoundaryConditionType.DIRICHLET:
            val = self.bcs.left.get_value(t=time)
            self._u_new[:, 0] = val
        if self.bcs.right.boundary_type == BoundaryConditionType.DIRICHLET:
            val = self.bcs.right.get_value(t=time)
            self._u_new[:, -1] = val
        # bottom/top
        if self.bcs.bottom.boundary_type == BoundaryConditionType.DIRICHLET:
            val = self.bcs.bottom.get_value(t=time)
            self._u_new[0, :] = val
        if self.bcs.top.boundary_type == BoundaryConditionType.DIRICHLET:
            val = self.bcs.top.get_value(t=time)
            self._u_new[-1, :] = val


class ADIHeatSolver(BaseHeatSolver, Sweep2DMixin, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_sweep_arrays()

    def solve(
        self,
        u: NDArray[np.float64],
        delta: float,
        sf: NDArray[np.float64],
        time: float = 0.0,
    ) -> NDArray[np.float64]:
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
        n_x, n_y = self.cfg.geometry.n_x, self.cfg.geometry.n_y
        self.compute_effective_properties(u=u, delta=delta)

        self.convective_operator(
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            correction_x=self._correction_x,
            correction_y=self._correction_y,
            convected_quantity=u,
            sf=sf,
        )

        self._compute_sweep_x_coeffs(u=u)

        self._u_new[:, :] = u

        self._apply_boundary_conditions_x(time=time)

        self._solve_sweep_x(
            n=n_y,
            a=self._a_x,
            b=self._b_x,
            c=self._c_x,
            rhs=self._rhs_x,
            result=self._u_new,
        )

        if self.convective_operator.form == ConvectiveTermForm.DEFERRED_CORRECTION:
            self.convective_operator(
                conv_x=self._conv_x,
                conv_y=self._conv_y,
                correction_x=self._correction_x,
                correction_y=self._correction_y,
                convected_quantity=self._u_new,
                sf=sf,
                recalculate_velocity=False,
            )

        self._compute_sweep_y_coeffs(u=u)

        self._apply_boundary_conditions_y(time=time)

        self._solve_sweep_y(
            n=n_x,
            a=self._a_y,
            b=self._b_y,
            c=self._c_y,
            rhs=self._rhs_y,
            result=self._u_new,
        )

        if self.post_correction:
            self.posterior_correction(u_old=u, u_star=self._u_new, delta=delta, time=time)

        return self._u_new

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
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        rhs: NDArray[np.float64],
        bc: BoundaryCondition,
        side: int,
        time: float,
        k_eff_slice: NDArray[np.float64],
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
                flux = (
                    bc.get_flux(t=time)
                    * self.cfg.l
                    / (
                        k_eff_slice
                        * self.cfg.thermal_conductivity_ref
                        * self.cfg.delta_u
                    )
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
