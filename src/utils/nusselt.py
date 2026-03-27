import numpy as np

from src.parameters.config import ExperimentConfig


def nusselt(
    u: np.ndarray, cfg: ExperimentConfig, wall: str = "left", order: int = 2
) -> float:
    """
    Вычисление числа Нуссельта с выбором стенки.

    Параметры:
    wall : str — 'left', 'right', 'bottom', 'top'
    order : int — порядок аппроксимации (1 или 2)
    """
    dx, dy, _ = cfg.scaled_grid_steps

    wall_config = {
        "left": {"axis": 0, "idx": [0, 1, 2], "step": dx, "sign": -1},
        "right": {"axis": 0, "idx": [-1, -2, -3], "step": dx, "sign": +1},
        "bottom": {"axis": 1, "idx": [0, 1, 2], "step": dy, "sign": -1},
        "top": {"axis": 1, "idx": [-1, -2, -3], "step": dy, "sign": +1},
    }

    cfg_wall = wall_config[wall.lower()]
    axis = cfg_wall["axis"]
    idx = cfg_wall["idx"]
    step = cfg_wall["step"]
    sign = cfg_wall["sign"]

    # Вычисление градиента на границе
    if order == 2:
        if axis == 0:
            du_dn = (
                sign * 3.0 * u[:, idx[0]]
                - sign * 4.0 * u[:, idx[1]]
                + sign * 1.0 * u[:, idx[2]]
            ) / (2.0 * step)
        else:
            du_dn = (
                sign * 3.0 * u[idx[0], :]
                - sign * 4.0 * u[idx[1], :]
                + sign * 1.0 * u[idx[2], :]
            ) / (2.0 * step)
    else:
        if axis == 0:
            du_dn = sign * (u[:, idx[1]] - u[:, idx[0]]) / step
        else:
            du_dn = sign * (u[idx[1], :] - u[idx[0], :]) / step

    # Интегрирование вдоль стенки
    if axis == 0:
        nu_avg = np.trapz(du_dn, dx=dy)
    else:
        nu_avg = np.trapz(du_dn, dx=dx)

    return nu_avg
