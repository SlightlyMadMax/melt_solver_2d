import numpy as np
from matplotlib import pyplot as plt

b_conv = np.load("../../data/wavy_surface/boundary/melting/convection_boundary.npz")["b"]
b_stef = np.load("../../data/wavy_surface/boundary/melting/stefan_boundary.npz")["b"]

time = []
for n in range(b_stef.shape[0]):
    time.append(n)

plt.plot(time, b_stef, label="Pure conduction")
plt.plot(time, b_conv[:1201], label="Convection")
plt.legend()
plt.show()
