from typing import Any

import numpy as np
from decimal import Decimal

from pydantic import BaseModel, Field


class DomainGeometry(BaseModel):
    width: float = Field(..., gt=0.0, description="Width of the domain [m].")
    height: float = Field(..., gt=0.0, description="Height of the domain [m].")
    end_time: float = Field(..., gt=0.0, description="Terminate modelling time [s].")
    n_x: int = Field(..., gt=0, description="Number of grid points in X-direction.")
    n_y: int = Field(..., gt=0, description="Number of grid points in Y-direction")
    n_t: int = Field(..., gt=0, description="Number of time steps.")

    @property
    def dx(self) -> float:
        return self.width / (self.n_x - 1)

    @property
    def dy(self) -> float:
        return self.height / (self.n_y - 1)

    @property
    def dt(self) -> float:
        return self.end_time / self.n_t

    @property
    def max_dimension(self) -> float:
        return max(self.width, self.height)

    @property
    def mesh_grid(self) -> tuple[np.ndarray[Any, np.dtype], ...]:
        x = np.linspace(0, self.width, self.n_x)
        y = np.linspace(0, self.height, self.n_y)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def __str__(self):
        return (
            f"Domain geometry:\n"
            f"  Width: {self.width} m\n"
            f"  Height: {self.height} m\n"
            f"  Terminate Time: {int(self.end_time / 60)} min\n"
            f"  X-step = {Decimal(self.dx):.2E} m\n"
            f"  Y-step = {Decimal(self.dy):.2E} m\n"
            f"  Time Step = {round(self.dt, 4)} s\n"
        )
