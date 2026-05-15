import numpy as np
from src.parameters.config import ExperimentConfig


def calculate_nusselt(
    u: np.ndarray, cfg: ExperimentConfig, wall: str = "left", order: int = 2
) -> float:
    """
    Compute the average Nusselt number on the specified wall.

    Parameters
    ----------
    u : np.ndarray
        Dimensionless temperature field (shape: [n_y, n_x])
    cfg : ExperimentConfig
        Experiment configuration (contains scaled grid steps dx, dy)
    wall : str
        Wall name: 'left', 'right', 'bottom', or 'top'
    order : int
        Approximation order for gradient (1 or 2)

    Returns
    -------
    float
        Average Nusselt number integrated along the wall
    """
    dx, dy, _ = cfg.scaled_grid_steps

    wall_config = {
        "left": {
            "grad_axis": 0,
            "idx": [0, 1, 2],
            "step": dx,
            "nu_sign": -1,
            "int_step": dy,
        },
        "right": {
            "grad_axis": 0,
            "idx": [-1, -2, -3],
            "step": dx,
            "nu_sign": 1,
            "int_step": dy,
        },
        "bottom": {
            "grad_axis": 1,
            "idx": [0, 1, 2],
            "step": dy,
            "nu_sign": -1,
            "int_step": dx,
        },
        "top": {
            "grad_axis": 1,
            "idx": [-1, -2, -3],
            "step": dy,
            "nu_sign": -1,
            "int_step": dx,
        },
    }

    cfg_wall = wall_config[wall.lower()]
    grad_axis = cfg_wall["grad_axis"]
    idx = cfg_wall["idx"]
    step = cfg_wall["step"]
    nu_sign = cfg_wall["nu_sign"]
    int_step = cfg_wall["int_step"]

    if order == 2:
        if grad_axis == 0:
            dudn = (-3.0 * u[:, idx[0]] + 4.0 * u[:, idx[1]] - u[:, idx[2]]) / (
                2.0 * step
            )
        else:
            dudn = (-3.0 * u[idx[0], :] + 4.0 * u[idx[1], :] - u[idx[2], :]) / (
                2.0 * step
            )
    else:
        if grad_axis == 0:
            dudn = (u[:, idx[1]] - u[:, idx[0]]) / step
        else:
            dudn = (u[idx[1], :] - u[idx[0], :]) / step

    nu_local = nu_sign * dudn
    nu_avg = np.trapz(nu_local, dx=int_step)

    return nu_avg
