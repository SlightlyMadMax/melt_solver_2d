import numpy as np
from matplotlib import pyplot as plt

b_conv = np.load("../../data/wavy_surface/boundary/freezing/convection_boundary.npz")["b"]
b_stef = np.load("../../data/wavy_surface/boundary/freezing/stefan_boundary.npz")["b"]

time = []
for n in range(b_stef.shape[0]):
    time.append(n)

plt.plot(time, b_stef)
plt.plot(time, b_conv[:1201])
plt.show()