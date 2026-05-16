import numpy as np
from matplotlib import pyplot as plt

b_conv_melt = np.load("./data/convection_boundary_48_96_hrs.npz")["b"]
# b_cond_melt = np.load("./data/conduction_boundary.npz")["b"]

t = np.arange(0, len(b_conv_melt)) * 10

plt.plot(t, b_conv_melt, linewidth=2, label="Convection")
# plt.plot(t, b_cond_melt, linewidth=2, label="Conduction")

plt.ylabel("Average interface position, m")
plt.xlabel("Time, min")
plt.legend()
plt.savefig("./graphs/avg_interface_pos_48_96_hrs.jpg", dpi=300)
