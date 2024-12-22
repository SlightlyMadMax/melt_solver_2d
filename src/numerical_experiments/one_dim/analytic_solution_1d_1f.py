import numpy as np
from scipy.special import erf
from scipy.optimize import fsolve


N_X = N_Y = 500
dx = 1.0 / (N_X - 1)
dy = 1.0 / (N_Y - 1)

spy = 60.*60.*24.*365.24
L = 3.34e5
rho_water = 999.84
rho_ice = 918.0
Ks = spy*2.1
cs = 2000.
ks = Ks/(rho_ice*cs)
Kl = spy*0.58
cl = 4217.
kl = Kl/(rho_water*cl)


Tm = 0.0
T_ = -10.0

s_0 = 0.3  # 0.30303030303030304


def boundary(lmbd, t):
    return 2*lmbd*t**.5


def get_B(lmbd):
    return (Tm-T_)/(erf(lmbd*ks**(-.5)))


def get_temperature(lmbd, x, t):
    return T_ + get_B(lmbd)*erf(x/((4*ks*t)**.5))


def trans_equation(lmbd):
    lhs = rho_ice*L*lmbd*np.pi**0.5*ks**0.5
    rhs = -Ks*get_B(lmbd)*np.exp(-lmbd*lmbd/ks)
    return lhs-rhs


lam = fsolve(trans_equation, 1.0)[0]  # 1.05

t_0 = (s_0 / (2.0 * lam))**2  # t_0 = 0.02040470144494864


analytic_T = np.empty((N_Y, N_X))

for j in range(N_Y):
    temp = get_temperature(lam, j*dy, t_0)
    if temp < 0.0:
        analytic_T[j, :] = temp
    else:
        analytic_T[j, :] = 0.0

np.savez_compressed("./num_vs_analytic/data/analytic_at_0_3", T=analytic_T)

from src.plotting import plot_temperature

plot_temperature(analytic_T, t_0, 0, False)

