import numpy as np

from src.parameters.config import ExperimentConfig


def nusselt(u: np.ndarray, cfg: ExperimentConfig) -> float:
    dx, dy, _ = cfg.scaled_grid_steps
    du_dx = (-3.0 * u[:, 0] + 4 * u[:, 1] - u[:, 2]) / (2.0 * dx)
    nu = np.trapz(du_dx, dx=dy)
    return nu
