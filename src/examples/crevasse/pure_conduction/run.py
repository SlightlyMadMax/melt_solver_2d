import logging
import sys
import time

import numpy as np

from src.core.boundary_conditions import BoundaryConditions
from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.heat_transfer.coefficient_smoothing.coefficients import StepScheme, DeltaScheme
from src.heat_transfer.init_values import (
    init_temperature,
    init_temperature_with_interface,
)
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from src.utils.boundary_conditions import (
    const_dirichlet_condition,
    const_neumann_condition,
)
from src.utils.time_utils import get_remaining_time

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
    logger.info(cfg)
    geometry: DomainGeometry = cfg.geometry
    dt = geometry.dt
    n_x, n_y, n_t = geometry.n_x, geometry.n_y, geometry.n_t
    min_temp = 263.15
    max_temp = 278.15

    material_props: MaterialProperties = cfg.material_props

    delta_u = cfg.delta_u
    u_ref = cfg.u_ref

    # Temperature boundary conditions
    u_bcs = BoundaryConditions(
        top=const_dirichlet_condition(n_x, value=(max_temp - u_ref) / delta_u),
        right=const_neumann_condition(n_y, value=0.0),
        bottom=const_dirichlet_condition(n_x, value=(min_temp - u_ref) / delta_u),
        left=const_neumann_condition(n_y, value=0.0),
    )

    b_lim = (
        (
            material_props.thermal_conductivity_liquid
            * material_props.specific_latent_heat
            / material_props.density_solid
        )
        * (max_temp + ABS_ZERO)
        / (
            (
                material_props.thermal_conductivity_liquid
                * material_props.specific_latent_heat
                / material_props.density_solid
            )
            * (max_temp + ABS_ZERO)
            + (
                material_props.thermal_conductivity_solid
                * material_props.specific_latent_heat
                / material_props.density_solid
            )
            * abs(min_temp + ABS_ZERO)
        )
    )
    print(f"Theoretical terminal boundary position: {(1.0 - b_lim) * geometry.height}")

    water_thickness = 0.025
    crevasse_width = 0.025
    crevasse_depth = 0.15
    f = np.empty(n_x)

    for i in range(n_x):
        x = i * geometry.dx
        if abs(x - geometry.width / 2) <= crevasse_width / 2:
            f[i] = geometry.height - water_thickness - crevasse_depth
        else:
            f[i] = geometry.height - water_thickness

    u = init_temperature_with_interface(
        cfg=cfg,
        liquid_temp=max_temp,
        solid_temp=min_temp,
        f=f,
        liquid_region_height=0.0,
    )

    plot_temperature(
        u=u * cfg.delta_u + cfg.u_ref,
        cfg=cfg,
        graph_id=0,
        plot_boundary=True,
        show_graph=True,
        min_temp=min_temp + ABS_ZERO,
        max_temp=max_temp + ABS_ZERO,
        invert_yaxis=False,
        actual_temp_units=TemperatureUnit.KELVIN,
        display_temp_units=TemperatureUnit.CELSIUS,
        directory="./graphs/temperature/",
    )

    heat_transfer_solver = HeatTransferSolver(
        cfg=cfg,
        bcs=u_bcs,
        max_iters=1,
        tolerance=1e-6,
        urf=1.0,
        solver_name=HeatTransferSolverName.LOC_ONE_DIM,
        step_scheme=StepScheme.ERF,
        delta_scheme=DeltaScheme.GAUSS,
        k_face_method=KFaceMethod.FROM_TEMP,
    )

    pt_arr = [0.05]

    start_time = time.perf_counter()

    for n in range(1, geometry.n_t + 1):
        t = n * geometry.dt
        u[:, :] = heat_transfer_solver.solve(
            u, sf=np.zeros_like(u), time=t, delta=cfg.delta_nd
        )
        if n % 900 == 0:
            print(
                f"ВРЕМЯ ВЫПОЛНЕНИЯ: {(time.perf_counter() - start_time) / 60:.2f} мин., "
                f"ОСТАЛОСЬ: {get_remaining_time(n=n, n_t=geometry.n_t, start_time=start_time) / 60:.2f} мин."
            )
            plot_temperature(
                u=u * cfg.delta_u + cfg.u_ref,
                cfg=cfg,
                graph_id=n,
                plot_boundary=True,
                show_graph=False,
                min_temp=min_temp + ABS_ZERO,
                max_temp=max_temp + ABS_ZERO,
                invert_yaxis=False,
                actual_temp_units=TemperatureUnit.KELVIN,
                display_temp_units=TemperatureUnit.CELSIUS,
                directory="./graphs/temperature/",
            )

    pt_a = (1.0 - b_lim) * geometry.height
    pt_num = pt_arr[-1]
    print(f"Calculated final location of the boundary: {pt_num}")
    print(
        f"Absolute error: {abs(pt_num - pt_a)}, relative: {round(abs(pt_num - pt_a) * 100 / pt_a, 2)}%"
    )
