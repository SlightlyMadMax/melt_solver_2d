import numpy as np
import math

from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.special import erf
from src.constants import K_ICE, K_WATER, C_ICE_VOL, C_WATER_VOL, L_VOL


g = -5.0
u_0 = 5.0
s_0 = 0.3

a_ice = (K_ICE / C_ICE_VOL) ** 0.5
a_water = (K_WATER / C_WATER_VOL) ** 0.5


def trans_eq(_gamma: float):
    lhs = K_ICE * g * math.exp(-(_gamma / (2.0 * a_ice)) ** 2) / (a_ice * erf(_gamma / (2.0 * a_ice)))
    rhs = -K_WATER * u_0 * math.exp(-(_gamma / (2.0 * a_water)) ** 2) / \
          (a_water * (1.0 - erf(_gamma / (2.0 * a_water)))) - \
          _gamma * L_VOL * math.pi ** 0.5 / 2
    return lhs - rhs


b_5 = np.load("results/delta_5_s_0_3/1d_2f_boundary.npz")["boundary"]

b_1 = np.load("results/delta_1_s_0_3/1d_2f_boundary.npz")["boundary"]

b_0_3 = np.load("results/delta_0_3_s_0_3/1d_2f_boundary.npz")["boundary"]

b_0_0_1 = np.load("results/delta_0_0_1_s_0_3/1d_2f_boundary.npz")["boundary"]

b_0_0_0_5 = np.load("results/delta_0_0_0_5_s_0_3/1d_2f_boundary.npz")["boundary"]

n = len(b_5)

gamma = fsolve(trans_eq, 0.0002)[0]
t_0 = (s_0 / gamma) ** 2

time = [i * 60. * 60. * 24.0 + t_0 for i in range(n)]

exact = [gamma * time[i] ** 0.5 for i in range(n)]

fig = plt.figure()
ax = plt.axes()

plt.plot(time, exact, linewidth=1.1, color='k', label='Аналитическое решение')

plt.plot(time, b_5, linewidth=1.1, color='orange', label='Δ = 5')
plt.plot(time, b_1, linewidth=1.1, color='r', label='Δ = 1')
plt.plot(time, b_0_3, linewidth=1.1, color='g', label='Δ = 0.3')
plt.plot(time, b_0_0_1, linewidth=1.1, color='b', label='Δ = 0.01')
plt.plot(time, b_0_0_0_5, linewidth=1.1, color='violet', label='Δ = 0.005')

ax.set_xlabel("Время, с")
ax.set_ylabel("Положение границы фазового перехода, м")
ax.legend()
plt.savefig(f"./all_deltas.png")
plt.show()
