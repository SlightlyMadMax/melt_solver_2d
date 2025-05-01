import numpy as np

from typing import Optional, Tuple
from numba import njit
from numpy.typing import NDArray

from src.convective_operators.base_convective_operator import (
    BaseConvectiveOperator,
    ConvectiveTermForm,
)
from src.core.geometry import DomainGeometry


class ConvectiveVorticityTransportOperator(BaseConvectiveOperator):
    def __init__(self, form: ConvectiveTermForm, geometry: DomainGeometry):
        super().__init__(geometry=geometry, n_points=3)
        self.form = form
        self._v_x: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._v_y: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )

    def __call__(
        self,
        sf: NDArray[np.float64],
        u: Optional[NDArray[np.float64]] = None,
        u_pt: Optional[float] = None,
        *args,
        **kwargs
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        self.compute_velocity_from_sf(
            sf=sf,
            v_x=self._v_x,
            v_y=self._v_y,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
        )
        if self.form == ConvectiveTermForm.UPWIND:
            self._compute_upwind_components(
                v_x=self._v_x,
                v_y=self._v_y,
                result_x=self._result_x,
                result_y=self._result_y,
                dx=self.geometry.dx / self.geometry.length_scale,
                dy=self.geometry.dy / self.geometry.length_scale,
            )
        elif self.form == ConvectiveTermForm.DIVERGENT_CENTRAL:
            self._compute_div_components(
                v_x=self._v_x,
                v_y=self._v_y,
                result_x=self._result_x,
                result_y=self._result_y,
                dx=self.geometry.dx / self.geometry.length_scale,
                dy=self.geometry.dy / self.geometry.length_scale,
            )
        elif self.form == ConvectiveTermForm.NON_DIVERGENT_CENTRAL:
            self._compute_non_div_components(
                v_x=self._v_x,
                v_y=self._v_y,
                result_x=self._result_x,
                result_y=self._result_y,
                dx=self.geometry.dx / self.geometry.length_scale,
                dy=self.geometry.dy / self.geometry.length_scale,
            )
        elif self.form == ConvectiveTermForm.SYMMETRIC:
            self._compute_div_components(
                v_x=self._v_x,
                v_y=self._v_y,
                result_x=self._result_x,
                result_y=self._result_y,
                dx=self.geometry.dx / self.geometry.length_scale,
                dy=self.geometry.dy / self.geometry.length_scale,
            )
            temp_x, temp_y = np.copy(self._result_x), np.copy(self._result_y)
            self._compute_non_div_components(
                v_x=self._v_x,
                v_y=self._v_y,
                result_x=self._result_x,
                result_y=self._result_y,
                dx=self.geometry.dx / self.geometry.length_scale,
                dy=self.geometry.dy / self.geometry.length_scale,
            )
            self._result_x = 0.5 * (temp_x + self._result_x)
            self._result_y = 0.5 * (temp_y + self._result_y)
        else:
            raise NotImplementedError

        if u is not None and u_pt is not None:
            self._restrict(conv_x=self._result_x, conv_y=self._result_y, u=u, u_pt=u_pt)

        return self._result_x, self._result_y

    @staticmethod
    @njit
    def compute_velocity_from_sf(
        sf: NDArray[np.float64],
        v_x: NDArray[np.float64],
        v_y: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> None:
        """
        Compute the velocity components (v_x and v_y) from the stream function.

        This function calculates the velocity field using the relationship between
        the stream function and velocity components in 2D incompressible flow:
            v_x = ∂(stream function) / ∂y
            v_y = -∂(stream function) / ∂x

        The velocity is computed using central differences for interior points.
        At the boundary points, the velocity components are set to zero to enforce the no-slip boundary condition.
        :param sf: A 2D numpy array representing the stream function values
                   over the computational grid.
        :param v_x: A 2D array where the x-component of the velocity field will be stored.
        :param v_y: A 2D array where the y-component of the velocity field will be stored.
        :param dx: Grid spacing in the x-direction.
        :param dy: Grid spacing in the y-direction.
        :return: None
        """
        n_y, n_x = sf.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        v_x[:, :] = 0.0
        v_y[:, :] = 0.0

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                v_x[j, i] = (sf[j + 1, i] - sf[j - 1, i]) * 0.5 * inv_dy
                v_y[j, i] = -(sf[j, i + 1] - sf[j, i - 1]) * 0.5 * inv_dx

    @staticmethod
    @njit
    def _compute_upwind_components(
        v_x: NDArray[np.float64],
        v_y: NDArray[np.float64],
        result_x: NDArray[np.float64],
        result_y: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> None:
        n_y, n_x = v_x.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                v_x_p = 0.5 * (v_x[j, i] + v_x[j, i + 1])
                v_x_m = 0.5 * (v_x[j, i] + v_x[j, i - 1])
                v_y_p = 0.5 * (v_y[j, i] + v_y[j + 1, i])
                v_y_m = 0.5 * (v_y[j, i] + v_y[j - 1, i])

                result_x[j, i, 0] = 0.5 * inv_dx * (v_x_p - abs(v_x_p))
                result_x[j, i, 1] = inv_dx * (
                    0.5 * (v_x_p + abs(v_x_p)) - 0.5 * (v_x_m - abs(v_x_m))
                )
                result_x[j, i, 2] = -0.5 * inv_dx * (v_x_m + abs(v_x_m))

                result_y[j, i, 0] = 0.5 * inv_dy * (v_y_p - abs(v_y_p))
                result_y[j, i, 1] = inv_dy * (
                    0.5 * (v_y_p + abs(v_y_p)) - 0.5 * (v_y_m - abs(v_y_m))
                )
                result_y[j, i, 2] = -0.5 * inv_dy * (v_y_m + abs(v_y_m))

    @staticmethod
    @njit
    def _compute_div_components(
        v_x: NDArray[np.float64],
        v_y: NDArray[np.float64],
        result_x: NDArray[np.float64],
        result_y: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> None:
        n_y, n_x = v_x.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                if i == 1:
                    result_x[j, i, 0] = inv_dx * v_x[j, i + 1]
                    result_x[j, i, 1] = inv_dx * v_x[j, i]
                    result_x[j, i, 2] = 0.0
                elif i == n_x - 2:
                    result_x[j, i, 0] = 0.0
                    result_x[j, i, 1] = inv_dx * v_x[j, i]
                    result_x[j, i, 2] = -inv_dx * v_x[j, i - 1]
                else:
                    result_x[j, i, 0] = 0.5 * inv_dx * v_x[j, i + 1]
                    result_x[j, i, 1] = 0.0
                    result_x[j, i, 2] = -0.5 * inv_dx * v_x[j, i - 1]
                # if j == 1:
                #     result_y[j, i, 0] = inv_dy * v_y[j + 1, i]
                #     result_y[j, i, 1] = inv_dy * v_y[j, i]
                #     result_y[j, i, 2] = 0.0
                # elif j == n_y - 1:
                #     result_y[j, i, 0] = 0.0
                #     result_y[j, i, 1] = inv_dy * v_y[j, i]
                #     result_y[j, i, 2] = -inv_dy * v_y[j - 1, i]
                # else:
                result_y[j, i, 0] = 0.5 * inv_dy * v_y[j + 1, i]
                result_y[j, i, 1] = 0.0
                result_y[j, i, 2] = -0.5 * inv_dy * v_y[j - 1, i]

    @staticmethod
    @njit
    def _compute_non_div_components(
        v_x: NDArray[np.float64],
        v_y: NDArray[np.float64],
        result_x: NDArray[np.float64],
        result_y: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> None:
        n_y, n_x = v_x.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                result_x[j, i, 0] = 0.5 * inv_dx * v_x[j, i]
                result_x[j, i, 1] = 0.0
                result_x[j, i, 2] = -0.5 * inv_dx * v_x[j, i]

                result_y[j, i, 0] = 0.5 * inv_dy * v_y[j, i]
                result_y[j, i, 1] = 0.0
                result_y[j, i, 2] = -0.5 * inv_dy * v_y[j, i]
