import numpy as np

from typing import Optional
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError

from src.convective_operators.base_convective_operator import (
    BaseConvectiveOperator,
    ConvectiveTermForm,
)
from src.core.geometry import DomainGeometry


class VorticityTransportArgs(BaseModel):
    sf: np.ndarray
    u: Optional[np.ndarray] = None
    u_pt: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


class VorticityTransportOperator(BaseConvectiveOperator):
    def __init__(self, form: ConvectiveTermForm, geometry: DomainGeometry):
        super().__init__(geometry=geometry)
        self.form = form
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        self._v_x: NDArray[np.float64] = np.empty((n_y, n_x))
        self._v_y: NDArray[np.float64] = np.empty((n_y, n_x))

    def __call__(self, conv_x, conv_y, **kwargs) -> None:
        try:
            parsed = VorticityTransportArgs(**kwargs)
        except ValidationError as e:
            raise ValueError(f"Invalid arguments for VorticityTransportOperator: {e}")

        sf = parsed.sf
        u = parsed.u
        u_pt = parsed.u_pt
        dx = self.geometry.dx / self.geometry.length_scale
        dy = self.geometry.dy / self.geometry.length_scale

        self.compute_velocity_from_sf(
            sf=sf,
            v_x=self._v_x,
            v_y=self._v_y,
            dx=dx,
            dy=dy,
        )
        if self.form == ConvectiveTermForm.UPWIND:
            self._compute_upwind_components(
                v_x=self._v_x,
                v_y=self._v_y,
                result_x=conv_x,
                result_y=conv_y,
                dx=dx,
                dy=dy,
            )
        elif self.form == ConvectiveTermForm.DIVERGENT_CENTRAL:
            self._compute_div_components(
                v_x=self._v_x,
                v_y=self._v_y,
                result_x=conv_x,
                result_y=conv_y,
                dx=dx,
                dy=dy,
            )
        elif self.form == ConvectiveTermForm.NON_DIVERGENT_CENTRAL:
            self._compute_non_div_components(
                v_x=self._v_x,
                v_y=self._v_y,
                result_x=conv_x,
                result_y=conv_y,
                dx=dx,
                dy=dy,
            )
        elif self.form == ConvectiveTermForm.SYMMETRIC:
            self._compute_div_components(
                v_x=self._v_x,
                v_y=self._v_y,
                result_x=conv_x,
                result_y=conv_y,
                dx=dx,
                dy=dy,
            )
            temp_x, temp_y = np.copy(conv_x), np.copy(conv_y)
            self._compute_non_div_components(
                v_x=self._v_x,
                v_y=self._v_y,
                result_x=conv_x,
                result_y=conv_y,
                dx=dx,
                dy=dy,
            )
            conv_x = 0.5 * (temp_x + conv_x)
            conv_y = 0.5 * (temp_y + conv_y)
        else:
            raise NotImplementedError

        if u is not None and u_pt is not None:
            self._restrict(conv_x=conv_x, conv_y=conv_y, u=u, u_pt=u_pt)

    @staticmethod
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
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        v_x[:] = 0.0
        v_y[:] = 0.0

        v_x[1:-1, 1:-1] = inv_2dy * (sf[2:, 1:-1] - sf[:-2, 1:-1])
        v_y[1:-1, 1:-1] = -inv_2dx * (sf[1:-1, 2:] - sf[1:-1, :-2])

    @staticmethod
    def _compute_upwind_components(
        v_x: NDArray[np.float64],
        v_y: NDArray[np.float64],
        result_x: NDArray[np.float64],
        result_y: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> None:
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

    @staticmethod
    def _compute_div_components(
        v_x: NDArray[np.float64],
        v_y: NDArray[np.float64],
        result_x: NDArray[np.float64],
        result_y: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> None:
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

    @staticmethod
    def _compute_non_div_components(
        v_x: NDArray[np.float64],
        v_y: NDArray[np.float64],
        result_x: NDArray[np.float64],
        result_y: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> None:
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        interior = (slice(1, -1), slice(1, -1))

        result_x[interior + (0,)] = inv_2dx * v_x[interior]
        result_x[interior + (1,)] = 0.0
        result_x[interior + (2,)] = -inv_2dx * v_x[interior]

        result_y[interior + (0,)] = inv_2dy * v_y[interior]
        result_y[interior + (1,)] = 0.0
        result_y[interior + (2,)] = -inv_2dy * v_y[interior]
