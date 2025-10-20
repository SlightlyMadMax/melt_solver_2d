import numpy as np
from matplotlib import pyplot as plt

from src.fluid_dynamics.plotting import plot_velocity_field
from src.parameters.config import ExperimentConfig
from src.utils.water_convection_benchmark import (
    calculate_T_profile_X05,
    calculate_T_profile_Y05,
)

cfg: ExperimentConfig = ExperimentConfig.load_from_file(
    "../../parameter_sets/water/convection.json"
)
delta_u = cfg.delta_u
u_ref = cfg.u_ref
n_x, n_y = cfg.geometry.n_x, cfg.geometry.n_y
data = np.load("../../data/water_convection/conv.npz")
u = data["u"]
v_x, v_y = data["v_x"], data["v_y"]

plot_velocity_field(
    v_x=v_x,
    v_y=v_y,
    u_dim=u * cfg.delta_u,
    cfg=cfg,
    graph_id=123,
    show_graph=True,
    directory="../../graphs/velocity/",
    equal_aspect=False,
    stride=8,
)

# X = np.linspace(0, 1, n_x)
# u_true = calculate_T_profile_Y05(X)
#
# u_mid = u[n_y // 2, :]  # take middle column in x-direction
#
# plt.figure(figsize=(6, 5))
# plt.plot(X, u_true, label="Michalek T. 2005", linewidth=2)
# plt.plot(
#     X, u_mid, "--", label=r"Numerical solution", linewidth=2
# )
#
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xticks(np.linspace(0, 1, 11))
# plt.yticks(np.linspace(0, 1, 11))
# plt.grid(True, which="both", linestyle="--", alpha=0.7)
#
# plt.xlabel("X", fontsize=14)
# plt.ylabel(r"$\theta$", fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# # plt.show()
# plt.savefig("../../graphs/velocity/temp.jpg", dpi=300)
