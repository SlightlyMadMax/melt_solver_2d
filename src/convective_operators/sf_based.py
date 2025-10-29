import numpy as np

from typing import Optional
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError

from src.convective_operators.base_convective_operator import (
    BaseConvectiveOperator,
    ConvectiveTermForm,
)
from src.parameters.config import ExperimentConfig


class SFBasedArgs(BaseModel):
    sf: np.ndarray
    u: Optional[np.ndarray] = None
    u_pt: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


class StreamFunctionBasedConvectiveOperator(BaseConvectiveOperator):
    def __init__(self, form: ConvectiveTermForm, cfg: ExperimentConfig):
        super().__init__(cfg=cfg)
        self.form = form
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        self._v_x: NDArray[np.float64] = np.empty((n_y, n_x))
        self._v_y: NDArray[np.float64] = np.empty((n_y, n_x))

    def __call__(self, conv_x, conv_y, **kwargs) -> None:
        try:
            parsed = SFBasedArgs(**kwargs)
        except ValidationError as e:
            raise ValueError(
                f"Invalid arguments for StreamFunctionBasedConvectiveOperator: {e}"
            )

        sf = parsed.sf
        u = parsed.u
        u_pt = parsed.u_pt
        dx, dy, _ = self.cfg.scaled_grid_steps

        conv_x[:] = 0.0
        conv_y[:] = 0.0

        self.compute_velocity_from_sf(sf=sf)

        if self.form == ConvectiveTermForm.UPWIND:
            self._compute_upwind_components(result_x=conv_x, result_y=conv_y)
        elif self.form == ConvectiveTermForm.DIVERGENT_CENTRAL:
            self._compute_div_components(result_x=conv_x, result_y=conv_y)
        elif self.form == ConvectiveTermForm.NON_DIVERGENT_CENTRAL:
            self._compute_non_div_components(result_x=conv_x, result_y=conv_y)
        elif self.form == ConvectiveTermForm.SYMMETRIC:
            tmp_x, tmp_y = np.zeros_like(conv_x), np.zeros_like(conv_y)
            self._compute_div_components(result_x=tmp_x, result_y=tmp_y)
            self._compute_non_div_components(result_x=conv_x, result_y=conv_y)
            conv_x[:] = 0.5 * (tmp_x + conv_x)
            conv_y[:] = 0.5 * (tmp_y + conv_y)
        else:
            raise NotImplementedError(f"ConvectiveTermForm {self.form} not supported")

        if u is not None and u_pt is not None:
            self._restrict(conv_x=conv_x, conv_y=conv_y, u=u, u_pt=u_pt)

    def compute_velocity_from_sf(self, sf: NDArray[np.float64]) -> None:
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        v_x[:, :] = 0.0
        v_y[:, :] = 0.0

        v_x[1:-1, 1:-1] = inv_2dy * (sf[2:, 1:-1] - sf[:-2, 1:-1])
        v_y[1:-1, 1:-1] = -inv_2dx * (sf[1:-1, 2:] - sf[1:-1, :-2])

    def _compute_upwind_components(
        self, result_x: NDArray[np.float64], result_y: NDArray[np.float64]
    ) -> None:
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        jm, jp = slice(1, -1), slice(2, None)
        im, ip = slice(1, -1), slice(2, None)
        im2, jm2 = slice(None, -2), slice(None, -2)

        vx_c = v_x[jm, im]
        vx_r = v_x[jm, ip]
        vx_l = v_x[jm, im2]
        vy_c = v_y[jm, im]
        vy_d = v_y[jp, im]
        vy_u = v_y[jm2, im]

        vxp = 0.5 * (vx_c + vx_r)
        vxm = 0.5 * (vx_c + vx_l)
        vyp = 0.5 * (vy_c + vy_d)
        vym = 0.5 * (vy_c + vy_u)

        result_x[jm, im, 0] = inv_2dx * (vxp - np.abs(vxp))
        result_x[jm, im, 1] = inv_2dx * ((vxp + np.abs(vxp)) - (vxm - np.abs(vxm)))
        result_x[jm, im, 2] = -inv_2dx * (vxm + np.abs(vxm))

        result_y[jm, im, 0] = inv_2dy * (vyp - np.abs(vyp))
        result_y[jm, im, 1] = inv_2dy * ((vyp + np.abs(vyp)) - (vym - np.abs(vym)))
        result_y[jm, im, 2] = -inv_2dy * (vym + np.abs(vym))

    def _compute_div_components(
        self, result_x: NDArray[np.float64], result_y: NDArray[np.float64]
    ) -> None:
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        jm, im = slice(1, -1), slice(1, -1)
        ip, im2 = slice(2, None), slice(None, -2)
        jp, jm2 = slice(2, None), slice(None, -2)

        result_x[jm, im, 0] = inv_2dx * v_x[jm, ip]  # v_x[j, i+1]
        result_x[jm, im, 1] = 0.0
        result_x[jm, im, 2] = -inv_2dx * v_x[jm, im2]  # -v_x[j, i-1]

        result_y[jm, im, 0] = inv_2dy * v_y[jp, im]  # v_y[j+1, i]
        result_y[jm, im, 1] = 0.0
        result_y[jm, im, 2] = -inv_2dy * v_y[jm2, im]  # -v_y[j-1, i]

    def _compute_non_div_components(
        self, result_x: NDArray[np.float64], result_y: NDArray[np.float64]
    ) -> None:
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        interior = (slice(1, -1), slice(1, -1))

        result_x[interior + (0,)] = inv_2dx * v_x[interior]
        result_x[interior + (1,)] = 0.0
        result_x[interior + (2,)] = -inv_2dx * v_x[interior]

        result_y[interior + (0,)] = inv_2dy * v_y[interior]
        result_y[interior + (1,)] = 0.0
        result_y[interior + (2,)] = -inv_2dy * v_y[interior]
