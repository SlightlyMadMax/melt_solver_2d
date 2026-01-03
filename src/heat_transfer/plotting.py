import os
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional
from PIL import Image
from numpy.typing import NDArray

from src.core.constants import ABS_ZERO
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.config import ExperimentConfig


def _convert_temp_in_display_units(
    u: NDArray[np.float64],
    actual_temp_units: TemperatureUnit = TemperatureUnit.KELVIN,
    display_temp_units: TemperatureUnit = TemperatureUnit.CELSIUS,
) -> NDArray[np.float64]:
    """
    Convert temperature to the desired units.

    :param u: A 2D array of temperatures.
    :param actual_temp_units: The original units of measurement.
    :param display_temp_units: The desired units of measurement.
    :return: An array of temperatures in the desired units of measurement.
    """
    if actual_temp_units == display_temp_units:
        return u
    return (
        u + ABS_ZERO if display_temp_units == TemperatureUnit.CELSIUS else u - ABS_ZERO
    )


def plot_temperature(
    u: NDArray[np.float64],
    cfg: ExperimentConfig,
    graph_id: int,
    actual_temp_units: TemperatureUnit = TemperatureUnit.KELVIN,
    display_temp_units: TemperatureUnit = TemperatureUnit.CELSIUS,
    plot_boundary: bool = False,
    show_graph: bool = True,
    show_grid: bool = False,
    directory: str = "../graphs/temperature/",
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    equal_aspect: Optional[bool] = True,
    invert_xaxis: Optional[bool] = False,
    invert_yaxis: Optional[bool] = False,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    """
    Plot and save a temperature field visualization for a 2D domain.

    This function visualizes the temperature distribution in a 2D domain using
    a filled contour plot. It supports various customization options, including
    displaying phase transition boundaries, grid points, axis inversion, and
    aspect ratio adjustment.

    :param u: A 2D numpy array of temperature values across the computational domain.
    :param cfg: An object containing experiment parameters (geometry, material properties, etc.).
    :param graph_id: Unique identifier for the graph, used in the saved file name.
    :param actual_temp_units: The unit of the temperature values in `u` (default is Kelvin).
    :param display_temp_units: The unit to display in the plot (default is Celsius).
    :param plot_boundary: Whether to plot the phase transition boundary (default is False).
    :param show_graph: Whether to display the graph interactively (default is True).
                       If False, the graph is saved without display.
    :param show_grid: Whether to display the grid points on the plot (default is False).
    :param directory: Directory path where the plot will be saved (default is "../graphs/temperature/").
    :param min_temp: Minimum temperature value for the color scale (default is None, auto-scaled).
    :param max_temp: Maximum temperature value for the color scale (default is None, auto-scaled).
    :param equal_aspect: Whether to enforce an equal aspect ratio for the plot (default is True).
    :param invert_xaxis: Whether to invert the x-axis (default is False).
    :param invert_yaxis: Whether to invert the y-axis (default is False).
    :param x_min: Minimum x-coordinate to display (default: full domain).
    :param x_max: Maximum x-coordinate to display (default: full domain).
    :param y_min: Minimum y-coordinate to display (default: full domain).
    :param y_max: Maximum y-coordinate to display (default: full domain).

    :return: None. The function saves the plot to the specified directory and optionally
             displays it if `show_graph` is True.
    """

    X, Y = cfg.geometry.mesh_grid

    # plotting region
    x_min_plot = x_min if x_min is not None else 0.0
    x_max_plot = x_max if x_max is not None else cfg.geometry.width
    y_min_plot = y_min if y_min is not None else 0.0
    y_max_plot = y_max if y_max is not None else cfg.geometry.height

    # Ensure bounds are within domain
    x_min_plot = max(x_min_plot, 0.0)
    x_max_plot = min(x_max_plot, cfg.geometry.width)
    y_min_plot = max(y_min_plot, 0.0)
    y_max_plot = min(y_max_plot, cfg.geometry.height)

    # Find indices corresponding to the subregion
    x_indices = np.where((X[0, :] >= x_min_plot) & (X[0, :] <= x_max_plot))[0]
    y_indices = np.where((Y[:, 0] >= y_min_plot) & (Y[:, 0] <= y_max_plot))[0]

    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Specified region is outside the computational domain.")

    # Slice the arrays
    X_sub = X[np.ix_(y_indices, x_indices)]
    Y_sub = Y[np.ix_(y_indices, x_indices)]
    u_sub = u[np.ix_(y_indices, x_indices)]

    if not show_graph:
        import matplotlib

        matplotlib.use("Agg")

    plt.figure(figsize=(8, 6))

    ax = plt.axes(
        xlim=(x_min_plot, x_max_plot),
        ylim=(y_min_plot, y_max_plot),
        xlabel="x, м",
        ylabel="y, м",
    )

    if show_grid:
        plt.plot(X_sub, Y_sub, marker=".", markersize=0.5, color="k", linestyle="none")

    disp_u = _convert_temp_in_display_units(
        u_sub, actual_temp_units, display_temp_units
    )

    if min_temp is None or max_temp is None:
        min_temp = disp_u.min()
        max_temp = disp_u.max()

    levels = np.linspace(min_temp, max_temp, 101)
    contour = plt.contourf(
        X_sub,
        Y_sub,
        disp_u,
        levels=levels,
        cmap="Blues",
        extend="both",
    )
    cbar = plt.colorbar(contour)
    cbar.set_ticks(np.linspace(min_temp, max_temp, num=7))
    cbar.set_label("Температура, °С", rotation=270, labelpad=15)

    if plot_boundary:
        X_b_list, Y_b_list = get_phase_trans_boundary(u=u, cfg=cfg)
        X_b_full, Y_b_full = np.asarray(X_b_list), np.asarray(Y_b_list)
        mask = (
            (X_b_full >= x_min_plot)
            & (X_b_full <= x_max_plot)
            & (Y_b_full >= y_min_plot)
            & (Y_b_full <= y_max_plot)
        )
        X_b = X_b_full[mask]
        Y_b = Y_b_full[mask]
        if len(X_b) > 0:
            plt.scatter(X_b, Y_b, s=1, linewidths=0.1, color="k", label="Граница ф.п.")
            ax.legend()

    if invert_xaxis:
        ax.invert_xaxis()

    if invert_yaxis:
        ax.invert_yaxis()

    if equal_aspect:
        ax.set_aspect("equal")

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{directory}T_{graph_id}.jpg", dpi=300)

    if show_graph:
        plt.show()
    else:
        plt.close()


def create_gif_from_images(
    output_filename: str,
    source_directory: str = "../graphs/temperature/",
    output_directory: str = "../graphs/animations/",
    duration: int = 100,
    loop: int = 0,
) -> None:
    """
    Creates a GIF animation from PNG images in a specified folder.

    :param output_filename: Filename of the resulting GIF file
    :param source_directory: Path to the folder containing PNG images.
    :param output_directory: Path to save the resulting GIF file.
    :param duration: Duration of each frame in milliseconds. Default is 100 ms.
    :param loop: Number of loops. 0 means infinite looping. Default is 0.
    """
    # Sort files by the numeric part of the filename (e.g., T_1.png -> 1)
    image_files = sorted(
        [file for file in os.listdir(source_directory) if file.endswith(".jpg")],
        key=lambda x: int(
            x.split("_")[1].split(".")[0]
        ),  # Extract the number after 'T_' and before '.png'
    )

    if not image_files:
        raise ValueError("No PNG files found in the specified folder.")

    images = [Image.open(os.path.join(source_directory, file)) for file in image_files]

    output_path = output_directory + output_filename + ".gif"

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        format="GIF",
    )
    print(f"GIF created and saved to {output_path}")
