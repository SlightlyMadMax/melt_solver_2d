import numpy as np
from matplotlib import pyplot as plt

from src.core.constants import ABS_ZERO
from src.parameters.config import ExperimentConfig

cfg_8c: ExperimentConfig = ExperimentConfig.load_from_file("parameters/8c.json")

cfg_4c: ExperimentConfig = ExperimentConfig.load_from_file("./parameters/4c.json")

cfg_5pt6c: ExperimentConfig = ExperimentConfig.load_from_file("./parameters/5pt6c.json")


# --------------------------------------------------
# Загрузка данных и перевод температуры в размерный вид
# --------------------------------------------------

# data_8c = np.load("./data/8c/checkpoint_750.npz")
# u_8c = data_8c["u"]
# u_8c_dim = u_8c * cfg_8c.delta_u + cfg_8c.u_ref + ABS_ZERO
#
# data_4c = np.load("./data/4c/checkpoint_1800.npz")
# u_4c = data_4c["u"]
# u_4c_dim = u_4c * cfg_4c.delta_u + cfg_4c.u_ref + ABS_ZERO

data_5pt6c = np.load("./data/5pt6c_thin/checkpoint_600.npz")
u_5pt6c = data_5pt6c["u"]
u_5pt6c_dim = u_5pt6c * cfg_5pt6c.delta_u + cfg_5pt6c.u_ref + ABS_ZERO

x = np.linspace(0.0, cfg_5pt6c.geometry.width, cfg_5pt6c.geometry.n_x)
y = np.linspace(0.0, cfg_5pt6c.geometry.height, cfg_5pt6c.geometry.n_y)


levels = np.linspace(u_5pt6c_dim.min(), u_5pt6c_dim.max(), 60)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, u_5pt6c_dim, levels=levels, cmap="Blues")
plt.show()
