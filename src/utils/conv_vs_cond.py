import numpy as np
from matplotlib import pyplot as plt

b_conv = np.load("../../data/wavy_surface/convection_boundary.npz")["b"]
stef_conv = np.load("../../data/wavy_surface/stefan_boundary.npz")["b"]

time = []
for n in range(stef_conv.shape[0]):
    time.append(n)

plt.plot(time, stef_conv)
plt.show()