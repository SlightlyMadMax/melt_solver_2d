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
from src.core.solvers.mixins.adi import ADIMixin
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
        self._f_l = np.empty((n_y, n_x))

    def calculate_liquid_fraction(self, u):
        """
        Vectorized implementation of liquid fraction calculation.
        """
        n_y, n_x = u.shape
        f_l = self._f_l
        h_x, h_y, _ = self.cfg.scaled_grid_steps

        u_0 = self.cfg.u_pt_nd
        # Create grid coordinates
        x_centers = np.arange(n_x) * h_x
        y_centers = np.arange(n_y) * h_y
        X, Y = np.meshgrid(x_centers, y_centers)

        # Control volume boundaries
        X_left = X - h_x / 2
        X_right = X + h_x / 2
        Y_bottom = Y - h_y / 2
        Y_top = Y + h_y / 2

        # For interior nodes
        for i in range(1, n_y - 1):
            for j in range(1, n_x - 1):
                u_center = u[i, j]

                if u_center == u_0:
                    f_l[i, j] = 0.5
                    continue

                # Check neighbors for interface crossing
                neighbors = [(i - 1, j, 'north', Y[i, j]),  # north
                             (i + 1, j, 'south', Y[i, j]),  # south
                             (i, j - 1, 'west', X[i, j]),  # west
                             (i, j + 1, 'east', X[i, j])]  # east

                intersections = []

                for ni, nj, direction, center_pos in neighbors:
                    u_neighbor = u[ni, nj]

                    if (u_center - u_0) * (u_neighbor - u_0) < 0:
                        # Linear interpolation
                        t = (u_0 - u_center) / (u_neighbor - u_center)

                        if direction == 'north':
                            x = X[i, j]
                            y = Y[i, j] - t * h_y
                            if Y_bottom[i, j] < y < Y_top[i, j]:
                                intersections.append((x, y))

                        elif direction == 'south':
                            x = X[i, j]
                            y = Y[i, j] + t * h_y
                            if Y_bottom[i, j] < y < Y_top[i, j]:
                                intersections.append((x, y))

                        elif direction == 'west':
                            x = X[i, j] - t * h_x
                            y = Y[i, j]
                            if X_left[i, j] < x < X_right[i, j]:
                                intersections.append((x, y))

                        elif direction == 'east':
                            x = X[i, j] + t * h_x
                            y = Y[i, j]
                            if X_left[i, j] < x < X_right[i, j]:
                                intersections.append((x, y))

                # Calculate liquid fraction
                if len(intersections) == 2:
                    # Two intersections - interface crosses control volume
                    liquid_area = self.calculate_liquid_area(
                        intersections[0], intersections[1],
                        (X_left[i, j], X_right[i, j], Y_bottom[i, j], Y_top[i, j]),
                        u_center > u_0
                    )
                    f_l[i, j] = liquid_area / (h_x * h_y)

                elif len(intersections) == 0:
                    # No interface crossing
                    f_l[i, j] = 1.0 if u_center > u_0 else 0.0

                else:
                    # 1 or more than 2 intersections
                    # Use temperature at cell corners to estimate
                    corner_u = [
                        u[i, j], u[i - 1, j], u[i + 1, j], u[i, j - 1], u[i, j + 1],
                        u[i - 1, j - 1], u[i - 1, j + 1], u[i + 1, j - 1], u[i + 1, j + 1]
                    ]
                    liquid_corners = np.sum([1 for val in corner_u if val > u_0])
                    f_l[i, j] = liquid_corners / len(corner_u)

    def calculate_liquid_area(self, p1, p2, cv_bounds, center_is_liquid):
        x_left, x_right, y_bottom, y_top = cv_bounds

        # Control volume corners
        corners = [
            (x_left, y_bottom),  # SW
            (x_left, y_top),  # NW
            (x_right, y_top),  # NE
            (x_right, y_bottom)  # SE
        ]

        # Determine which corners are on the liquid side
        # Using line equation through p1 and p2
        x1, y1 = p1
        x2, y2 = p2

        # Line equation: (y2 - y1)(x - x1) - (x2 - x1)(y - y1) = 0
        # Sign indicates which side of the line a point is on
        liquid_corners = []

        for corner in corners:
            cx, cy = corner
            value = (y2 - y1) * (cx - x1) - (x2 - x1) * (cy - y1)

            # Determine if corner is liquid based on which side of the line
            # and whether center is liquid
            # This is a simplification - you might need to adjust based on your coordinate system
            is_liquid = (value > 0) == center_is_liquid

            if is_liquid:
                liquid_corners.append(corner)

        # Create polygon from liquid corners and intersection points
        polygon_points = liquid_corners + [p1, p2]

        # Remove duplicates and ensure proper order
        polygon_points = list(dict.fromkeys(polygon_points))

        if len(polygon_points) < 3:
            # Degenerate case - use simple area calculation
            if center_is_liquid:
                return (x_right - x_left) * (y_top - y_bottom) / 2
            else:
                return (x_right - x_left) * (y_top - y_bottom) / 2

        # Calculate polygon area using shoelace formula
        area = 0
        n = len(polygon_points)
        for k in range(n):
            x1, y1 = polygon_points[k]
            x2, y2 = polygon_points[(k + 1) % n]
            area += x1 * y2 - x2 * y1

        return abs(area) / 2

    @staticmethod
    def _calculate_effective_heat_capacity(
        c_eff,
        h,
        u,
        u_0,
        c_solid,
        c_liquid,
        dx,
        dy,
    ):
        dT_dx = np.zeros_like(u)
        dT_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        dT_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx  # forward
        dT_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx  # backward

        dT_dy = np.zeros_like(u)
        dT_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
        dT_dy[0, :] = (u[1, :] - u[0, :]) / dy  # forward
        dT_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy  # backward

        dH_dx = np.zeros_like(h)
        dH_dx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2 * dx)
        dH_dx[:, 0] = (h[:, 1] - h[:, 0]) / dx
        dH_dx[:, -1] = (h[:, -1] - h[:, -2]) / dx

        dH_dy = np.zeros_like(h)
        dH_dy[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2 * dy)
        dH_dy[0, :] = (h[1, :] - h[0, :]) / dy
        dH_dy[-1, :] = (h[-1, :] - h[-2, :]) / dy

        numerator = dH_dx ** 2 + dH_dy ** 2
        denominator = dT_dx ** 2 + dT_dy ** 2
        epsilon = 1e-12
        denominator_safe = denominator + epsilon
        c_eff[:] = np.sqrt(numerator / denominator_safe)

        zero_grad_mask = denominator < epsilon
        c_eff[zero_grad_mask] = np.where(
            u[zero_grad_mask] < u_0,
            c_solid,
            c_liquid
        )

    def calculate_enthalpy(self, u: NDArray[np.float64]):
        props = self.cfg.material_props
        c_ref = self.cfg.volumetric_heat_capacity_ref

        c_solid_nd = props.volumetric_heat_capacity_solid / c_ref
        c_liquid_nd = props.volumetric_heat_capacity_liquid / c_ref
        latent_heat_nd = 1.0 / self.cfg.stefan_number
        dx, dy, _ = self.cfg.scaled_grid_steps

        u_0 = self.cfg.u_pt_nd
        self._h[:] = c_solid_nd * (u - u_0) + self._f_l * (latent_heat_nd + (c_liquid_nd - c_solid_nd) * (u - u_0))

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


class ADIHeatSolver(BaseHeatSolver, ADIMixin, ABC):
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

        :param u: solution array at the beginning of the step (u^n).
        :param sf: stream function
        :param delta: phase change smoothing range
        :param time: current physical time.
        :return:
        """
        self.compute_effective_properties(u=u, delta=delta)

        self.convective_operator(
            conv_x=self._conv_x,
            conv_y=self._conv_y,
            correction_x=self._correction_x,
            correction_y=self._correction_y,
            convected_quantity=u,
            sf=sf,
        )
        self._u_new[:, :] = u

        self._execute_adi_step(
            result=self._u_new,
            time=time,
            coeff_kwargs={"u": u},
            hook_kwargs={"sf": sf},
        )

        return self._u_new

    def _after_first_sweep(self, result: NDArray[np.float64], **kwargs) -> None:
        """
        Recalculate convective term after first sweep if using deferred correction.
        """
        if self.convective_operator.form == ConvectiveTermForm.DEFERRED_CORRECTION:
            sf = kwargs.get("sf")
            if sf is None:
                raise ValueError("sf is required for deferred correction")

            self.convective_operator(
                conv_x=self._conv_x,
                conv_y=self._conv_y,
                correction_x=self._correction_x,
                correction_y=self._correction_y,
                convected_quantity=result,
                sf=sf,
                recalculate_velocity=False,
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
