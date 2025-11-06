import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from src.core.geometry import DomainGeometry
from src.fluid_dynamics.init_values import initialize_velocity
from src.fluid_dynamics.plotting import plot_velocity_field
from src.fluid_dynamics.utils import calculate_velocity_from_sf
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig

img = plt.imread("../../data/kowalewski.png")

cfg: ExperimentConfig = ExperimentConfig.load_from_file(
    "../../parameter_sets/water/freezing.json"
)
geometry: DomainGeometry = cfg.geometry

min_temp = 263.15
max_temp = 283.15
delta_u = cfg.delta_u
u_ref = cfg.u_ref
u_pt = cfg.material_props.u_pt
l = cfg.l
v = cfg.v

# fig, ax = plt.subplots()
# ax.imshow(img, extent=[0, 1.0, 0, 1.0])

data = np.load("../../data/water_freezing/after_freezing_151x151.npz")
u = data["u"]
sf = data["sf"]
w = data["w"]
u_dim = u * delta_u + u_ref
v_x, v_y = initialize_velocity(geometry=geometry)
calculate_velocity_from_sf(sf, v_x, v_y, cfg)

plot_velocity_field(
    v_x * v,
    v_y * v,
    u_dim,
    cfg,
    111025,
    True,
    True,
    equal_aspect=False,
    stride=8,
    directory="../../graphs/velocity/",
)

# X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u * cfg.delta_u + cfg.u_ref)
# for i in range(len(X_b)):
#     X_b[i] /= cfg.l
#     Y_b[i] /= cfg.l
# ax.plot(X_b, Y_b, linestyle="--", color="red", linewidth=2)
#
# legend_elements = [
#     mlines.Line2D(
#         [], [], linestyle="--", color="red", linewidth=2, label="Numerical solution"
#     ),
# ]
# ax.legend(handles=legend_elements, fontsize=12)
#
# ax.set_xlabel("X", fontsize=14)
# ax.set_ylabel("Y", fontsize=14)
# plt.tight_layout()
#
# plt.savefig(f"../../graphs/water_freezing_exp_vs_num.jpg", dpi=300)
#
# plt.show()
