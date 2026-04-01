import os
import time

import numpy as np
from matplotlib import pyplot as plt

from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import BoundaryConditions
from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.heat_transfer.coefficient_smoothing.coefficients import StepScheme, DeltaScheme
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
)
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.config import ExperimentConfig
from src.utils.boundary_conditions import (
    const_neumann_condition,
    const_dirichlet_condition,
)
from src.utils.time_utils import get_remaining_time
from src.examples.stefan.two_dim.quantitative.analytical_solver import (
    StefanCornerSolver,
    StefanParameters,
)


if __name__ == "__main__":
    dir_name = input("Enter a directory name where the data will be stored: ")
    dir_path = f"./results/{dir_name}/"

    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass

    cfg: ExperimentConfig = ExperimentConfig.load_from_file("test_config_1.json")

    delta = input(
        "Enter the smoothing parameter delta or just press 'Enter' to use an adaptive one: "
    )

    if delta == "":
        delta = None
        fixed_delta = False
    else:
        delta = float(delta) / cfg.delta_u
        fixed_delta = True

    print(cfg)

    geometry: DomainGeometry = cfg.geometry

    n_x, n_y = geometry.n_x, geometry.n_y
    max_temp = 273.25
    min_temp = 268.15

    bcs = BoundaryConditions(
        top=const_neumann_condition(n_x, value=0.0),
        right=const_neumann_condition(n_y, value=0.0),
        bottom=const_dirichlet_condition(
            n_x, value=(min_temp - cfg.u_ref) / cfg.delta_u
        ),
        left=const_dirichlet_condition(n_y, value=(min_temp - cfg.u_ref) / cfg.delta_u),
    )

    heat_transfer_solver = HeatTransferSolver(
        cfg=cfg,
        solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
        convective_term_form=ConvectiveTermForm.NON_DIVERGENT_CENTRAL,
        bcs=bcs,
        max_iters=1,
        tolerance=1e-6,
        urf=1.0,
        step_scheme=StepScheme.CONST,
        delta_scheme=DeltaScheme.GAUSS,
    )

    u = np.ones((geometry.n_y, geometry.n_x)) * max_temp
    u[0, :] = min_temp
    u[:, 0] = min_temp
    u = (u - cfg.u_ref) / cfg.delta_u

    save_interval = 60
    start_time = time.perf_counter()
    for n in range(1, geometry.n_t + 1):
        t = n * geometry.dt
        if not fixed_delta:
            delta = get_mushy_zone_temperature_range(u, u_pt=cfg.u_pt_nd, n_nodes=1)
        u[:, :] = heat_transfer_solver.solve(
            u=u, sf=np.zeros_like(u), time=t, delta=delta
        )
        if n % save_interval == 0:
            print(
                f"Modelling Time: {int(t / 3600)} hours, "
                f"Elapsed Time: {(time.perf_counter() - start_time) / 60:.2f} min., "
                f"Estimated Remaining Time: {get_remaining_time(n=n, n_t=geometry.n_t, start_time=start_time) / 60:.2f} min."
            )

    print(f"Elapsed Time: {time.perf_counter() - start_time:.2f} s., ")

    u_dim = u * cfg.delta_u + cfg.u_ref

    plot_temperature(
        u=u_dim,
        cfg=cfg,
        graph_id=geometry.n_t,
        plot_boundary=True,
        show_graph=True,
        min_temp=min_temp + ABS_ZERO,
        max_temp=max_temp + ABS_ZERO,
        actual_temp_units=TemperatureUnit.KELVIN,
        display_temp_units=TemperatureUnit.CELSIUS,
        directory=dir_path,
    )

    x_i, y_i = get_phase_trans_boundary(u=u_dim, cfg=cfg)

    t = geometry.n_t * geometry.dt
    scale = np.sqrt(4 * cfg.material_props.thermal_diffusivity_solid * t)

    Tf = cfg.material_props.u_pt
    Tw = min_temp
    Ti = max_temp
    kL = cfg.material_props.thermal_conductivity_liquid
    kS = cfg.material_props.thermal_conductivity_solid

    beta = cfg.material_props.specific_latent_heat / (
        cfg.material_props.specific_heat_solid * (Tf - Tw)
    )
    Ti_star = (kL / kS) * (Ti - Tf) / (Tf - Tw)

    params = StefanParameters(beta=beta, Ti_star=Ti_star)
    solver = StefanCornerSolver(params)

    print()
    res = solver.solve()
    print()

    a = np.asarray(x_i) / scale
    x_star_i = a[a >= res["x0_star"]]
    y_star_analytic = solver.get_interface_position(x_star_i)

    yi_np = np.asarray(y_i)[a >= res["x0_star"]]
    errors = yi_np - y_star_analytic * scale
    print("Max abs error:", np.max(np.abs(errors)))
    print("RMS error:", np.sqrt(np.mean(errors**2)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, geometry.width], [0, geometry.width], "k--", alpha=0.5, label="y = x")
    ax.scatter(x_i, y_i, s=10, c="black", label="Numerical front ($T=T_{p.t.}$)")
    ax.scatter(
        x_star_i * scale, y_star_analytic * scale, s=10, c="red", label="Analytic front"
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, geometry.width)
    ax.set_ylim(0, geometry.height)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.grid(True)
    plt.savefig(dir_path)
    plt.show()
