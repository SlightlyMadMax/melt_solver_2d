import numpy as np

from typing import Optional, Tuple
from numba import njit
from numpy.typing import NDArray

from src.convective_operators.base_convective_operator import BaseConvectiveOperator
from src.core.geometry import DomainGeometry


class EffectiveSFTransportOperator(BaseConvectiveOperator):
    def __init__(self, geometry: DomainGeometry):
        super().__init__(geometry=geometry, n_points=3)
        self._dw_dx: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._dw_dy: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )

    def __call__(
        self,
        w: NDArray[np.float64],
        u: Optional[NDArray[np.float64]] = None,
        u_pt: Optional[float] = None,
        *args,
        **kwargs
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        self._compute_vorticity_first_derivatives(
            w=w,
            dw_dx=self._dw_dx,
            dw_dy=self._dw_dy,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
        )
        self._compute_convective_operator(
            dw_dx=self._dw_dx,
            dw_dy=self._dw_dy,
            result_x=self._result_x,
            result_y=self._result_y,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
        )

        if u is not None and u_pt is not None:
            self._restrict(conv_x=self._result_x, conv_y=self._result_y, u=u, u_pt=u_pt)

        return self._result_x, self._result_y

    @staticmethod
    @njit
    def _compute_vorticity_first_derivatives(
        w: NDArray[np.float64],
        dw_dx: NDArray[np.float64],
        dw_dy: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> None:
        n_y, n_x = w.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        dw_dx[:, :] = 0.0
        dw_dy[:, :] = 0.0

        for j in range(n_y):
            for i in range(n_x):
                if j != n_y - 1 and j != 0:
                    dw_dy[j, i] = (w[j + 1, i] - w[j - 1, i]) * 0.5 * inv_dy
                if i != n_x - 1 and i != 0:
                    dw_dx[j, i] = (w[j, i + 1] - w[j, i - 1]) * 0.5 * inv_dx

    @staticmethod
    @njit
    def _compute_convective_operator(
        dw_dx: NDArray[np.float64],
        dw_dy: NDArray[np.float64],
        dx: float,
        dy: float,
        result_x: NDArray[np.float64],
        result_y: NDArray[np.float64],
    ):
        n_y, n_x = dw_dx.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                result_x[j, i, 0] = 0.25 * inv_dx * (dw_dy[j, i] + dw_dy[j, i + 1])
                result_x[j, i, 1] = 0.0
                result_x[j, i, 2] = -0.25 * inv_dx * (dw_dy[j, i] + dw_dy[j, i - 1])

                result_y[j, i, 0] = -0.25 * inv_dy * (dw_dx[j, i] + dw_dx[j + 1, i])
                result_y[j, i, 1] = 0.0
                result_y[j, i, 2] = 0.25 * inv_dy * (dw_dx[j, i] + dw_dx[j - 1, i])
