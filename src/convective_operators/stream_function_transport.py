import numpy as np

from typing import Optional
from numpy.typing import NDArray
from pydantic import ValidationError, BaseModel

from src.convective_operators.base_convective_operator import BaseConvectiveOperator
from src.core.geometry import DomainGeometry


class SFTransportArgs(BaseModel):
    w: np.ndarray
    u: Optional[np.ndarray] = None
    u_pt: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


class EffectiveSFTransportOperator(BaseConvectiveOperator):
    def __init__(self, geometry: DomainGeometry):
        super().__init__(geometry=geometry)
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        self._dw_dx: NDArray[np.float64] = np.empty((n_y, n_x))
        self._dw_dy: NDArray[np.float64] = np.empty((n_y, n_x))

    def __call__(self, conv_x, conv_y, **kwargs) -> None:
        try:
            parsed = SFTransportArgs(**kwargs)
        except ValidationError as e:
            raise ValueError(f"Invalid arguments for EffectiveSFTransportOperator: {e}")

        w = parsed.w
        u = parsed.u
        u_pt = parsed.u_pt

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
            result_x=conv_x,
            result_y=conv_y,
            dx=self.geometry.dx / self.geometry.length_scale,
            dy=self.geometry.dy / self.geometry.length_scale,
        )

        if u is not None and u_pt is not None:
            self._restrict(conv_x=conv_x, conv_y=conv_y, u=u, u_pt=u_pt)

    @staticmethod
    def _compute_vorticity_first_derivatives(
        w: NDArray[np.float64],
        dw_dx: NDArray[np.float64],
        dw_dy: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> None:
        """
        Compute first derivatives of vorticity using second-order accurate
        3-point central differences in the interior and 3-point one-sided
        (forward/backward) differences at the boundaries.
        """
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        dw_dx[:, :] = 0.0
        dw_dy[:, :] = 0.0

        dw_dy[1:-1, 1:-1] = (w[2:, 1:-1] - w[:-2, 1:-1]) * inv_2dy
        dw_dx[1:-1, 1:-1] = (w[1:-1, 2:] - w[1:-1, :-2]) * inv_2dx

        dw_dy[0, :] = (-3.0 * w[0, :] + 4.0 * w[1, :] - w[2, :]) * inv_2dy
        dw_dy[-1, :] = (3.0 * w[-1, :] - 4.0 * w[-2, :] + w[-3, :]) * inv_2dy

        dw_dx[:, 0] = (-3.0 * w[:, 0] + 4.0 * w[:, 1] - w[:, 2]) * inv_2dx
        dw_dx[:, -1] = (3.0 * w[:, -1] - 4.0 * w[:, -2] + w[:, -3]) * inv_2dx

    @staticmethod
    def _compute_convective_operator(
        dw_dx: NDArray[np.float64],
        dw_dy: NDArray[np.float64],
        dx: float,
        dy: float,
        result_x: NDArray[np.float64],
        result_y: NDArray[np.float64],
    ):
        inv_4dx = 1.0 / (4.0 * dx)
        inv_4dy = 1.0 / (4.0 * dy)

        jm, jp = slice(1, -1), slice(2, None)
        im, ip = slice(1, -1), slice(2, None)

        result_x[jm, im, 0] = -inv_4dx * (dw_dy[jm, im] + dw_dy[jm, ip])
        result_x[jm, im, 1] = 0.0
        result_x[jm, im, 2] = inv_4dx * (dw_dy[jm, im] + dw_dy[jm, slice(None, -2)])

        result_y[jm, im, 0] = inv_4dy * (dw_dx[jm, im] + dw_dx[jp, im])
        result_y[jm, im, 1] = 0.0
        result_y[jm, im, 2] = -inv_4dy * (dw_dx[jm, im] + dw_dx[slice(None, -2), im])
