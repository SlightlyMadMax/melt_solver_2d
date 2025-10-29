import numpy as np

from numpy.typing import NDArray
from pydantic import ValidationError, BaseModel

from src.convective_operators.base_convective_operator import BaseConvectiveOperator
from src.parameters.config import ExperimentConfig


class VorticityBasedArgs(BaseModel):
    w: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class VorticityBasedConvectiveOperator(BaseConvectiveOperator):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg=cfg)
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        self._dw_dx: NDArray[np.float64] = np.empty((n_y, n_x))
        self._dw_dy: NDArray[np.float64] = np.empty((n_y, n_x))

    def __call__(self, conv_x, conv_y, **kwargs) -> None:
        try:
            parsed = VorticityBasedArgs(**kwargs)
        except ValidationError as e:
            raise ValueError(
                f"Invalid arguments for VorticityBasedConvectiveOperator: {e}"
            )

        w = parsed.w

        self._compute_vorticity_first_derivatives(w=w)
        self._compute_convective_operator(result_x=conv_x, result_y=conv_y)

    def _compute_vorticity_first_derivatives(self, w: NDArray[np.float64]) -> None:
        """
        Compute first derivatives of vorticity using second-order accurate central differences in the interior and
        3-point one-sided (forward/backward) differences at the boundaries.
        """
        dx, dy, _ = self.cfg.scaled_grid_steps
        dw_dx = self._dw_dx
        dw_dy = self._dw_dy

        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        dw_dx[:, 1:-1] = (w[:, 2:] - w[:, :-2]) * inv_2dx
        dw_dy[1:-1, :] = (w[2:, :] - w[:-2, :]) * inv_2dy

        dw_dx[:, 0] = (-3.0 * w[:, 0] + 4.0 * w[:, 1] - w[:, 2]) * inv_2dx
        dw_dx[:, -1] = (3.0 * w[:, -1] - 4.0 * w[:, -2] + w[:, -3]) * inv_2dx

        dw_dy[0, :] = (-3.0 * w[0, :] + 4.0 * w[1, :] - w[2, :]) * inv_2dy
        dw_dy[-1, :] = (3.0 * w[-1, :] - 4.0 * w[-2, :] + w[-3, :]) * inv_2dy

    def _compute_convective_operator(
        self,
        result_x: NDArray[np.float64],
        result_y: NDArray[np.float64],
    ):
        dx, dy, _ = self.cfg.scaled_grid_steps
        dw_dx = self._dw_dx
        dw_dy = self._dw_dy
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
