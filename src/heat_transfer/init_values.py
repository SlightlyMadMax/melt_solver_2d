import numpy as np

from enum import Enum
from numpy.typing import NDArray
from typing import Tuple, Optional

from src.geometry import DomainGeometry
from src.heat_transfer.parameters import ThermalParameters


class DomainShape(Enum):
    LINEAR = "linear"
    CIRCLE = "circle"
    DOUBLE_CIRCLE = "double_circle"
    PACMAN = "pacman"
    RECTANGLE = "rectangle"
    UNIFORM_LIQUID = "uniform_liquid"
    UNIFORM_SOLID = "uniform_solid"


def init_temperature_with_interface(
    geom: DomainGeometry,
    thermal_parameters: ThermalParameters,
    f: NDArray[np.float64],
    liquid_region_height: float,
    liquid_temp: float,
    solid_temp: float,
) -> NDArray[np.float64]:
    """
    Initializes the temperature field based on the given interface f.

    :param geom: An object containing geometry information.
    :param thermal_parameters: Object containing thermal parameters (phase-transition temperature etc.).
    :param f: 1D array representing the interface position for the phase transition.
    :param liquid_region_height: Height of the liquid region.
    :param liquid_temp: Temperature of the liquid phase.
    :param solid_temp: Temperature of the solid phase.
    :return: A 2D array of nondimensionilized temperatures initialized based on the interface.
    """
    u = np.empty((geom.n_y, geom.n_x))

    for i in range(geom.n_x):
        for j in range(geom.n_y):
            if j * geom.dy < f[i]:
                u[j, i] = solid_temp + j * geom.dy * (
                    thermal_parameters.u_pt - solid_temp
                ) / (geom.height - liquid_region_height)
            elif j * geom.dy > f[i]:
                u[j, i] = liquid_temp
            else:
                u[j, i] = thermal_parameters.u_pt

    u = (u - thermal_parameters.u_ref) / thermal_parameters.delta_u

    return u


def init_temperature(
    geom: DomainGeometry,
    shape: DomainShape,
    thermal_parameters: ThermalParameters,
    liquid_temp: Optional[float] = None,
    solid_temp: Optional[float] = None,
    radius: float = 0.25,
    small_radius: float = 0.1,
    eye_radius: float = 0.05,
    eye_offset: float = 0.6,
    rect_width: float = 0.04,
    rect_height: float = 0.12,
) -> NDArray[np.float64]:
    """
    Initializes the temperature field based on a specified domain shape.

    :param geom: An object containing geometry information.
    :param thermal_parameters: Object containing thermal parameters (phase-transition temperature etc.).
    :param shape: The shape of the temperature distribution.
    :param liquid_temp: The temperature assigned to water regions.
    :param solid_temp: The temperature assigned to ice regions.
    :param radius: The radius used for circular shapes (default: 0.25).
    :param small_radius: A smaller radius for additional features in shapes (default: 0.1).
    :param eye_radius: The radius of the eye in the Pacman shape (default: 0.05).
    :param eye_offset: The offset for positioning the eye in the Pacman shape (default: 0.6).
    :param rect_width: The width of the rectangle filled with solid phase (default: 0.02).
    :param rect_height: The height of the rectangle filled with solid phase (default: 0.12).
    :return: A 2D array of nondimensionilized initialized based on the specified shape of the domain.
    """
    u = np.full((geom.n_y, geom.n_x), solid_temp)

    X, Y = geom.mesh_grid

    if shape == DomainShape.UNIFORM_LIQUID:
        assert (
            liquid_temp is not None
        ), f"liquid_temp must be specified when shape = {shape}."

    elif shape == DomainShape.UNIFORM_SOLID:
        assert (
            solid_temp is not None
        ), f"solid_temp must be specified when shape = {shape}."

    else:
        assert (
            liquid_temp is not None and solid_temp is not None
        ), f"Both liquid_temp and solid_temp must be specified when shape = {shape}."

    if shape == DomainShape.LINEAR:
        # Linear temperature gradient from bottom (solid phase) to top (liquid phase)
        u[:, :] = np.linspace(solid_temp, liquid_temp, geom.n_y).reshape(1, -1)

    elif shape == DomainShape.CIRCLE:
        # Single circle centered at domain center with radius threshold
        mask = (X - geom.width / 2) ** 2 + (Y - geom.height / 2) ** 2 < radius**2
        u[mask] = liquid_temp

    elif shape == DomainShape.DOUBLE_CIRCLE:
        # Two circles centered vertically with specified radius
        mask1 = (X - geom.width / 2) ** 2 + (
            Y - 0.75 * geom.height
        ) ** 2 < small_radius**2
        mask2 = (X - geom.width / 2) ** 2 + (
            Y - 0.25 * geom.height
        ) ** 2 < small_radius**2
        u[mask1 | mask2] = liquid_temp

    elif shape == DomainShape.PACMAN:
        for i in range(geom.n_x):
            for j in range(geom.n_y):
                if (i * geom.dx - geom.width / 2.0) ** 2 + (
                    j * geom.dy - geom.height / 2.0
                ) ** 2 < radius**2:
                    if i * geom.dx <= j * geom.dy <= -i * geom.dx + 1:
                        u[j, i] = solid_temp  # Pacman's mouth
                    elif (i * geom.dx - eye_offset) ** 2 + (
                        j * geom.dy - eye_offset
                    ) ** 2 < eye_radius**2:
                        u[j, i] = solid_temp  # Pacman's eye
                    else:
                        u[j, i] = liquid_temp  # Pacman's body
                else:
                    u[j, i] = solid_temp

    elif shape == DomainShape.RECTANGLE:
        half_width = rect_width / 2
        half_height = rect_height / 2

        # center
        mask = (np.abs(X - geom.width / 2) < half_width) & (np.abs(Y - geom.height / 2) < half_height)

        # top
        # mask = (np.abs(X - geom.width / 2) < half_width) & (
        #     Y > geom.height - rect_height
        # )

        # bottom
        # mask = (np.abs(X - geom.width / 2) < half_width) & (Y < rect_height)

        # left
        # mask = (X < rect_width) & (np.abs(Y - geom.height / 2) < half_height)

        u[mask] = liquid_temp

    elif shape == DomainShape.UNIFORM_LIQUID:
        u = np.ones(u.shape) * liquid_temp

    elif shape == DomainShape.UNIFORM_SOLID:
        u = np.ones(u.shape) * solid_temp

    else:
        raise Exception("Unknown shape")

    u = (u - thermal_parameters.u_ref) / thermal_parameters.delta_u

    return u


def init_temperature_lake(
    geom: DomainGeometry,
    thermal_parameters: ThermalParameters,
    lake_data: Tuple[NDArray[np.float64], NDArray[np.float64]],
    water_temp: float,
    ice_temp: float,
) -> NDArray[np.float64]:
    """
    Initialize temperature for a lake profile using preloaded thickness data.

    :param geom: An object containing geometry information.
    :param thermal_parameters: An object containing thermal parameters (phase transition temperature etc.).
    :param lake_data: Preloaded water and ice thickness grids.
    :param water_temp: The water temperature.
    :param ice_temp: The ice temperature.
    :return: A 2D nondimensionilized temperature field array.
    """
    water_th_grid, ice_th_grid = lake_data

    grid_x = water_th_grid[0]
    grid_step = grid_x[1] - grid_x[0]
    print(f"Grid step: {grid_step}")

    lake_width = grid_x[-1]
    print(f"Lake width: {lake_width}")

    new_x = [i * geom.dx for i in range(int(lake_width / geom.dx + 1))]
    print(new_x[-1], len(new_x))

    water_th_interp = np.interp(new_x, grid_x, water_th_grid[1])

    ice_th_interp = np.interp(new_x, grid_x, ice_th_grid[1])

    print(f"Max lake thickness {max(water_th_interp)}")

    u = np.empty((geom.n_y, geom.n_x))

    for i in range(geom.n_x):
        x = i * geom.dx
        ice_th_at_x, water_th_at_x = 0.0, 0.0

        if (geom.width - lake_width) / 2.0 <= x <= (geom.width + lake_width) / 2.0:
            water_th_at_x = water_th_interp[i + int((len(new_x) - geom.n_x) / 2)]
            ice_th_at_x = ice_th_interp[i + int((len(new_x) - geom.n_x) / 2)]

        for j in range(geom.n_y):
            y = geom.height - j * geom.dy
            if water_th_at_x > 0.0 and ice_th_at_x <= y <= ice_th_at_x + water_th_at_x:
                u[j, i] = water_temp
            else:
                u[j, i] = ice_temp + (j / geom.n_y) * (
                    thermal_parameters.u_pt - ice_temp
                )

    u = (u - thermal_parameters.u_ref) / thermal_parameters.delta_u

    return u
