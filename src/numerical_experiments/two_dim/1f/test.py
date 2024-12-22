import numpy as np
from src.plotting import animate

t = np.linspace(0, 1, 100)


def a_boundary(x, t):
    return 0.5 * x + 1.25 * t + 0.5


def a_temp(x, y, t):
    return np.exp(1.25 * t - y + 0.5 * x + 0.5) - 1.0


x = np.linspace(0, 1.0, 300)
y = np.linspace(0, 2.25, 300)
X, Y = np.meshgrid(x, y)

T_full = []

for i in t:
    T_full.append(a_temp(X, Y, i))

animate(T_full, t, t[1]-t[0], "analytic_2d", X, Y, 300, 300, 1.0, 2.25)
