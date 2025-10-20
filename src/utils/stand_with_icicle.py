from typing import Optional

import numpy as np

from src.core.geometry import DomainGeometry
from src.parameters.config import ExperimentConfig


def init_temperature_icicle(
    cfg: ExperimentConfig,
    liquid_temp: Optional[float] = None,
    solid_temp: Optional[float] = None,
    rect_width: float = 0.04,
    rect_height: float = 0.12,
    location: str = "bottom",
) -> np.ndarray:
    geometry: DomainGeometry = cfg.geometry
    u = np.full((geometry.n_y, geometry.n_x), liquid_temp)

    X, Y = geometry.mesh_grid

    radius = rect_width / 2
    center_x = geometry.width / 2

    half_width = rect_width / 2
    half_height = rect_height / 2

    if location == "bottom":
        center_y = rect_height / 2
        semicircle_center_y = center_y + half_height
    else:  # "top"
        center_y = geometry.height - rect_height / 2
        semicircle_center_y = center_y - half_height

    # Rectangle part
    rect_mask = (np.abs(X - center_x) <= half_width) & (
        np.abs(Y - center_y) <= half_height
    )

    # Semicircle part
    dx = X - center_x
    dy = Y - semicircle_center_y
    semicircle_mask = dx**2 + dy**2 <= radius**2

    # Combined mask
    mask = rect_mask | semicircle_mask

    u[mask] = solid_temp

    u = (u - cfg.u_ref) / cfg.delta_u

    return u
