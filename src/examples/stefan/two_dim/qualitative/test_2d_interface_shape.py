import math
import time

import numpy as np


from src.core.constants import ABS_ZERO
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.heat_transfer.init_values import (
    init_temperature_with_interface,
    DomainShape,
    init_temperature,
)
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from src.utils.boundary_conditions import (
    const_dirichlet_condition,
    const_neumann_condition,
)
from src.utils.time_utils import get_remaining_time

max_temp = 278.15
min_temp = 268.15
reference_temperature = 0.5 * (min_temp + max_temp)

geometry = DomainGeometry(
    width=0.2,
    height=0.05,
    end_time=60.0 * 60.0 * 24.0,
    n_x=401,
    n_y=101,
    n_t=60 * 60 * 24,
)

print(geometry)

material_props = MaterialProperties(
    u_pt=273.15,
    specific_heat_liquid=4212,
    specific_heat_solid=2116,
    specific_latent_heat=335000,
    density_liquid=999.8,
    density_solid=916.8,
    thermal_conductivity_liquid=0.566,
    thermal_conductivity_solid=2.26,
    dynamic_viscosity=1.7888e-3,
    volumetric_thermal_exp=-6.733353e-05,
)

cfg = ExperimentConfig(
    geometry=geometry,
    material_props=material_props,
    u_ref=0.5 * (min_temp + max_temp),
    delta_u=0.5 * (max_temp - min_temp),
    l=0.05,
    delta=0.05,
    epsilon=1e-6,
)
n_x, n_y = geometry.n_x, geometry.n_y
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


# f = [
#     geometry.height / 2
#     - 0.2 * math.exp(-((i * geometry.dx - geometry.width / 4.0) ** 2) / 0.005)
#     + 0.2 * math.exp(-((i * geometry.dx - geometry.width / 1.5) ** 2) / 0.005)
#     for i in range(geometry.n_x)
# ]
#
# f = [geometry.height / 2 for i in range(geometry.n_x)]
# f = np.array(f)
#
# u = init_temperature_with_interface(
#     cfg=cfg,
#     liquid_temp=max_temp,
#     solid_temp=min_temp,
#     f=f,
#     liquid_region_height=0.0,
# )

bcs = BoundaryConditions(
    top=const_dirichlet_condition(n_x, value=(max_temp - cfg.u_ref) / cfg.delta_u),
    right=const_neumann_condition(n_y, value=0.0),
    bottom=const_dirichlet_condition(n_x, value=(min_temp - cfg.u_ref) / cfg.delta_u),
    left=const_neumann_condition(n_y, value=0.0),
)

u = init_temperature(
    cfg=cfg,
    bcs=bcs,
    shape=DomainShape.UNIFORM_SOLID,
    solid_temp=min_temp,
    liquid_temp=max_temp,
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
    directory="./results/",
)

heat_transfer_solver = HeatTransferSolver(
    cfg=cfg,
    solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
    bcs=bcs,
    k_face_method=KFaceMethod.FROM_TEMP,
    post_correction=False,
)

pt_arr = [0.05]

start_time = time.perf_counter()

for n in range(1, geometry.n_t + 1):
    t = n * geometry.dt
    u[:, :] = heat_transfer_solver.solve(
        u, sf=np.zeros_like(u), time=t, delta=cfg.delta_nd
    )
    if n % 60 == 0:
        print(
            f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {int(n / 60)} часов, "
            f"ВРЕМЯ ВЫПОЛНЕНИЯ: {(time.perf_counter() - start_time) / 60:.2f} мин., "
            f"ОСТАЛОСЬ: {get_remaining_time(n=n, n_t=geometry.n_t, start_time=start_time) / 60:.2f} мин."
        )
        i = int(geometry.n_x / 2)

        dy = geometry.dy
        u_pt = cfg.u_pt_nd
        diff = u - u_pt
        j = np.where(diff[:-1] * diff[1:] < 0)[0][0]
        u0, up1 = (u[j, i], u[j + 1, i])
        y0 = j * dy
        s_lin = y0 + dy * (u_pt - u0) / (up1 - u0)
        pt_arr.append(s_lin)

np.savez("../../../../data/wavy_surface/boundary/melting/stefan_boundary.npz", b=np.asarray(pt_arr))
pt_a = (1.0 - b_lim) * geometry.height
pt_num = pt_arr[-1]
print(f"Calculated final location of the boundary: {pt_num}")
print(
    f"Absolute error: {abs(pt_num - pt_a)}, relative: {round(abs(pt_num - pt_a) * 100 / pt_a, 2)}%"
)

# print("СОЗДАНИЕ АНИМАЦИИ...")
# create_gif_from_images(
#     output_filename="test_animation",
#     source_directory="./results/",
#     output_directory="./results/",
# )
