import math
import time

import numpy as np

from src.geometry import DomainGeometry
from src.heat_transfer.solvers import solve
from src.plotting import plot_temperature, animate
from src.heat_transfer import init_temperature_2f_test
from src.constants import K_ICE, K_WATER, RHO_ICE, L

T_WATER = 5.0
T_ICE = -5.0
end_time = 60.0 * 60.0 * 24.0 * 250.0
n_t = 24*250
T_0 = 0.0

b_lim = (K_WATER * L / RHO_ICE) * T_WATER / ((K_WATER * L / RHO_ICE) * T_WATER + (K_ICE * L / RHO_ICE)*abs(T_ICE))
print(f"Theoretical boundary final position: {1.0 - b_lim}")

geom = DomainGeometry(
    width=1.0,
    height=1.0,
    end_time=end_time,
    n_x=500,
    n_y=500,
    n_t=n_t
)

print(geom)

F = [geom.height / 2 - 0.2 * math.exp(-(i * geom.dx - geom.width / 4.0) ** 2 / 0.005) + 0.2 * math.exp(-(i * geom.dx - geom.width / 1.5) ** 2 / 0.005) for i in range(geom.n_x)]
F = np.array(F)

T = init_temperature_2f_test(geom=geom, water_temp=T_WATER, ice_temp=T_ICE, F=F)

plot_temperature(
    T=T,
    geom=geom,
    time=0.0,
    graph_id=0,
    plot_boundary=True,
    show_graph=True,
    min_temp=T_ICE,
    max_temp=T_WATER,
    directory="./results/"
)

T_full = [T]
times = [0]

start_time = time.process_time()

for i in range(1, n_t+1):
    t = i * geom.dt
    T = solve(T,
              top_cond_type=1,
              right_cond_type=2,
              bottom_cond_type=1,
              left_cond_type=2,
              dx=geom.dx,
              dy=geom.dy,
              dt=geom.dt,
              time=t,
              fixed_delta=False
              )
    if i % 24 == 0:
        T_full.append(T)
        times.append(t)
        print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {i} ч, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")

for j in range(1, geom.n_y - 1):
    if (T[j, 250] - T_0) * (T[j + 1, 250] - T_0) < 0.0:
        y_0 = abs((T[j, 250] * (j + 1) * geom.dy - T[j + 1, 250] * j * geom.dy) / (T[j, 250] - T[j + 1, 250]))
        print(f"Calculated final location of the boundary: {y_0}")
        print(f"Absolute error: {abs(y_0 - 1 + b_lim)}, relative: {round(abs(y_0 - 1 + b_lim) * 100/ b_lim, 2)}%")
        break

print("СОЗДАНИЕ АНИМАЦИИ...")
animate(
    T_full=T_full,
    geom=geom,
    times=times,
    t_step=60*60*24,
    directory="./results/",
    filename="test_animation",
    min_temp=T_ICE,
    max_temp=T_WATER
)
