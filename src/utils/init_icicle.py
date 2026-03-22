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

    # try:
    #     x_step = float(X[0, 1] - X[0, 0]) if geometry.n_x > 1 else 1.0
    #     y_step = float(Y[1, 0] - Y[0, 0]) if geometry.n_y > 1 else 1.0
    #     dist = edt(mask, sampling=(y_step, x_step))
    # except Exception:
    #     outside_idxs = np.column_stack(np.nonzero(~mask))
    #
    #     if outside_idxs.size == 0:
    #         dist = np.zeros_like(X)
    #         dist[mask] = 1.0
    #     else:
    #         xy_out = np.column_stack((X[~mask].ravel(), Y[~mask].ravel()))
    #         xy_in = np.column_stack((X[mask].ravel(), Y[mask].ravel()))
    #
    #         d2 = (xy_in[:, None, 0] - xy_out[None, :, 0]) ** 2 + (
    #             xy_in[:, None, 1] - xy_out[None, :, 1]
    #         ) ** 2
    #         min_dist_in = np.sqrt(d2.min(axis=1))
    #
    #         dist = np.zeros_like(X, dtype=float)
    #         dist[mask] = min_dist_in
    #
    # max_dist = float(dist[mask].max()) if np.any(mask) else 0.0
    # surface_temp = 273.15
    #
    # if max_dist <= 0:
    #     u[mask] = solid_temp
    # else:
    #     normalized = np.zeros_like(dist, dtype=float)
    #     normalized[mask] = dist[mask] / max_dist
    #     u[mask] = surface_temp + (solid_temp - surface_temp) * normalized[mask]

    u = (u - cfg.u_ref) / cfg.delta_u

    return u
