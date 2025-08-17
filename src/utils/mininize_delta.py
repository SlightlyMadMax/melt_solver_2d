import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

# Data
x = np.array([0.075, 0.1, 0.15, 0.2, 0.25, 0.3])
y = np.array(
    [
        0.003656533,
        0.002913074,
        0.002871984,
        0.003714405,
        0.004844584,
        0.006339469,

    ]
)

# Fit cubic spline (s=0 means it passes exactly through the points)
spline = UnivariateSpline(x, y, k=3, s=0)

# Find minimum of spline in the range of x
res = minimize_scalar(spline, bounds=(x.min(), x.max()), method="bounded")

x_min = res.x
y_min = spline(x_min)

print(f"Global minimum: x = {x_min:.5f}, y = {y_min:.5f}")
