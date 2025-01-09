import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.solvers.vorticity_solvers.registry import (
    VorticitySolverName,
    register_solver,
)
from src.geometry import DomainGeometry
from src.base_solver import BaseSolver
from src.fluid_dynamics.utils import get_indicator_function as c_ind


@register_solver(VorticitySolverName.EXPLICIT_CENTRAL)
class ExpCentralNavierStokesSolver(BaseSolver):
    def __init__(
        self,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        *args,
        **kwargs,
    ):
        super().__init__(
            geometry=geometry,
        )

        self.parameters = parameters

        # Pre-allocate some arrays that will be used in the calculations
        self._new_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )

    @staticmethod
    @njit
    def _compute_vorticity(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        result: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        reynolds_number: float,
        grashof_number: float,
        u_ref: float,
        u_pt_ref: float,
        epsilon: float,
    ) -> NDArray[np.float64]:
        n_y, n_x = w.shape
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        inv_dy = 1.0 / dy
        inv_dy2 = inv_dy * inv_dy

        inv_re = 1.0 / reynolds_number
        inv_re2 = inv_re * inv_re

        result[0, :] = -2.0 * inv_dy2 * sf[1, :]
        result[n_y - 1, :] = -2.0 * inv_dy2 * sf[n_y - 2, :]
        result[:, 0] = -2.0 * inv_dx2 * sf[:, 1]
        result[:, n_x - 1] = -2.0 * inv_dx2 * sf[:, n_x - 2]

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                result[j, i] = w[j, i] + dt * (
                    grashof_number
                    * inv_re2
                    * 0.5
                    * inv_dx
                    * (u[j, i + 1] - u[j, i - 1])
                    + inv_re * inv_dx2 * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    + inv_re * inv_dy2 * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j + 1, i - 1] - sf[j - 1, i - 1])
                    * w[j, i - 1]
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j - 1, i + 1] - sf[j + 1, i + 1])
                    * w[j, i + 1]
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j - 1, i - 1] - sf[j - 1, i + 1])
                    * w[j - 1, i]
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j + 1, i + 1] - sf[j + 1, i - 1])
                    * w[j + 1, i]
                    # + inv_re * c_ind(u=u[j, i], u_pt_ref=u_pt_ref, eps=epsilon) * sf[j, i]
                )

        return result

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ) -> (NDArray[np.float64], NDArray[np.float64]):
        self._new_w = np.copy(w)
        self._compute_vorticity(
            w=w,
            sf=sf,
            u=u,
            result=self._new_w,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
            dt=self.geometry.dt * self.parameters.v / self.geometry.length_scale,
            u_ref=self.parameters.u_ref,
            u_pt_ref=self.parameters.u_pt_ref,
            reynolds_number=self.parameters.reynolds_number,
            grashof_number=self.parameters.grashof_number,
            epsilon=self.parameters.epsilon,
        )

        return self._new_w
